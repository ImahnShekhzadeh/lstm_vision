import logging
import os
from datetime import datetime as dt

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch import optim
from torchinfo import summary
from typeguard import typechecked

from evaluate import check_accuracy, get_confusion_matrix
from train import TrainingConfig, str__cuda_0, train_and_validate
from utils import (
    check_config_keys,
    cleanup,
    get_datasets,
    get_git_info,
    get_model,
    get_samplers_loaders,
    load_checkpoint,
    log_param_table,
    setup_ddp_if_needed,
)


@typechecked
def run(rank: int | torch.device, world_size: int, cfg: DictConfig) -> None:
    """
    Run LSTM on MNIST data.

    Args:
        rank: Rank of the current process. Can be `torch.device("cpu")` if no
            GPU is available.
        world_size: Number of processes participating in distributed training.
            If `world_size` is 1, no distributed training is used.
        cfg: Configuration dictionary from hydra containing keys and values.
    """

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if cfg.training.seed_number is not None:
        # set random seed, each process gets different seed
        torch.manual_seed(cfg.training.seed_number + rank)

    setup_ddp_if_needed(rank, world_size, cfg)

    train_dataset, val_dataset, test_dataset = get_datasets(
        channels_img=cfg.dataset.channels_img,
        train_split=cfg.dataset.train_split,
    )
    (
        train_sampler,
        val_sampler,
        train_loader,
        val_loader,
        test_loader,
    ) = get_samplers_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.dataloading.num_workers,
        pin_memory=cfg.dataloading.pin_memory,
        use_ddp=cfg.training.use_ddp,
        seed_number=cfg.training.seed_number,
    )

    seq_length = test_loader.dataset[0][0].shape[1]
    inp_size = test_loader.dataset[0][0].shape[2]
    num_classes = len(test_loader.dataset.classes)

    model = get_model(
        input_size=inp_size,
        num_layers=cfg.model.num_layers,
        hidden_size=cfg.model.hidden_size,
        num_classes=num_classes,
        sequence_length=seq_length,
        bidirectional=cfg.model.bidirectional,
        dropout_rate=cfg.model.dropout,
        device=rank,
        use_ddp=cfg.training.use_ddp,
    )

    saving_name_best_cp = "lstm_best_cp.pt"

    if rank in [0, torch.device("cpu")]:
        check_config_keys(cfg)

        wandb_logging = cfg.training.wandb__api_key is not None
        if wandb_logging:
            wandb.login(key=cfg.training.wandb__api_key)
            wandb.init(
                project="lstm_vision",
                name=dt.now().strftime("%dp%mp%Y_%Hp%Mp%S"),
                config=OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                ),
            )

        if torch.cuda.is_available():
            list_gpus = [
                torch.cuda.get_device_name(i) for i in range(world_size)
            ]
            logging.info(f"\nGPU(s): {list_gpus}\n")

        get_git_info()

        logging.info(
            f"# Train:val:test samples: {len(train_loader.dataset)}"
            f":{len(val_loader.dataset)}:{len(test_loader.dataset)}\n\n"
            f"{summary(model, (cfg.training.batch_size, seq_length, inp_size))}\n"
        )
        log_param_table(model)
    else:
        wandb_logging = False

    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=cfg.optim.learning_rate,
        betas=(cfg.optim.beta_1, cfg.optim.beta_2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )
    model.train()

    if cfg.model.loading_path is not None:
        if rank == torch.device("cpu"):
            map_location = {str__cuda_0: "cpu"}
        else:
            map_location = {str__cuda_0: f"cuda:{rank}"}

        load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint=torch.load(
                cfg.model.loading_path,
                map_location=map_location,
                weights_only=True,
            ),
        )

    if cfg.training.num_epochs > 0:
        training_config = TrainingConfig(
            num_epochs=cfg.training.num_epochs,
            num_grad_accum_steps=cfg.training.num_grad_accum_steps,
            world_size=world_size,
            use_amp=cfg.training.use_amp,
            max_norm=cfg.training.max_norm,
            label_smoothing=cfg.training.label_smoothing,
            num_additional_cps=cfg.training.num_additional_cps,
            saving_path=output_dir,
            saving_name_best_cp=None if rank > 0 else saving_name_best_cp,
            freq_output__train=cfg.training.freq_output__train,
            freq_output__val=cfg.training.freq_output__val,
            wandb_logging=wandb_logging,
        )

        train_and_validate(
            model=model,
            optimizer=optimizer,
            rank=rank,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=training_config,
            train_sampler=train_sampler,
        )

        # Load checkpoint with lowest validation loss for final evaluation.
        # It is necessary that all processes load the same checkpoint.
        # Use a `barrier()` to make sure that process 1 loads the model after
        # process 0 saves it
        if cfg.training.use_ddp:
            dist.barrier()
        if rank == torch.device("cpu"):
            map_location = {str__cuda_0: "cpu"}
        else:
            map_location = {str__cuda_0: f"cuda:{rank}"}
        load_checkpoint(
            model=model,
            checkpoint=torch.load(
                os.path.join(output_dir, saving_name_best_cp),
                map_location=map_location,
                weights_only=True,
            ),
        )

    train__num_correct, train__num_samples = check_accuracy(
        train_loader,
        model,
        use_amp=cfg.training.use_amp,
        mode="train",
        device=rank,
        use_ddp=cfg.training.use_ddp,
    )

    if cfg.training.use_ddp:
        cleanup()  # destroy process group, clean exit

    # TODO: refactor following code into func `evaluate_test_set()`
    if rank in [0, torch.device("cpu")]:
        test__num_correct, test__num_samples = check_accuracy(
            test_loader,
            model,
            use_amp=cfg.training.use_amp,
            mode="test",
            device=rank,
            use_ddp=False,
        )
        # TODO: split the following and put into func `check_accuracy`, rename
        # `check_accuracy` into `check_log_accuracy`.
        logging.info(
            f"\nTrain data: Got {train__num_correct}/{train__num_samples} with"
            f" accuracy {(100 * train__num_correct / train__num_samples):.2f} "
            f"%\nTest data: Got {test__num_correct}/{test__num_samples} with "
            f"accuracy {(100 * test__num_correct / test__num_samples):.2f} %"
        )

        get_confusion_matrix(
            num_classes,
            test_loader,
            model,
            use_amp=cfg.training.use_amp,
            saving_path=output_dir,
            device=rank,
        )

        if wandb_logging:
            wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="conf")
@typechecked
def main(cfg: DictConfig) -> None:
    """
    Main.

    Args:
        cfg: Configuration dictionary from hydra containing keys and values.
    """

    world_size = int(os.getenv("WORLD_SIZE", 1))  # num GPUs

    if cfg.training.use_ddp and world_size == 1:
        logging.warning(
            "Distributed Data Parallel (DDP) is enabled but only one GPU is "
            "available. Proceeding with training on a single GPU."
        )
        cfg.training.use_ddp = False

    if cfg.training.use_ddp and world_size > 1:
        run(rank=int(os.getenv("RANK", 0)), world_size=world_size, cfg=cfg)
    else:
        rank = 0 if torch.cuda.is_available() else torch.device("cpu")
        if (
            cfg.training.use_ddp
            or cfg.training.master_addr is not None
            or cfg.training.master_port is not None
        ):
            logging.warning(
                "Distributed Data Parallel (DDP) can only be used if at least "
                "two GPUs are available. Proceeding with training on "
                f"{torch.device(rank)}."
            )
            cfg.training.use_ddp = False
        run(rank=rank, world_size=world_size, cfg=cfg)


if __name__ == "__main__":
    main()
