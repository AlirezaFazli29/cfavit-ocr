import os
import torch
from transformers import (
    get_scheduler,
    TrOCRProcessor,
    PreTrainedModel,
)
from tqdm.auto import tqdm
from typing import Literal, Annotated


WarmupFraction = Annotated[float, "must be between 0 and 1"]


class TransformerTrainer:
    """
    Trainer class for PyTorch HuggingFace transformer models with train, validation, 
    checkpointing, learning rate scheduling, and HuggingFace model saving support.
    
    Attributes:
        model (PreTrainedModel): The transformer model to train.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        test__loader (DataLoader): DataLoader for test data.
        optimizer (torch.optim.Optimizer): AdamW optimizer.
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        history (dict): Dictionary storing training and validation loss per epoch.
        device (str): Device to train on ('cpu' or 'cuda').
        best_checkpoints_path (str): Path to save the best model checkpoint.
        last_checkpoints_path (str): Path to save the last model checkpoint.
    """

    def __init__(
            self,
            model: PreTrainedModel,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            test__loader: torch.utils.data.DataLoader,
            lr: float = 2e-5,
            weight_decay: float = 0.01,
            num_epochs: int = 10,
            warmup_steps_percentage: WarmupFraction = 0.1,
            device: Literal["cpu", "cuda"] = "cpu",
            checkpoints_path: str = "app/core/models/training-checkpoints",
    ):
        """
        Initialize the TransformerTrainer.
        
        Args:
            model (PreTrainedModel): Transformer model to train.
            train_loader (DataLoader): Training data loader.
            valid_loader (DataLoader): Validation data loader.
            test__loader (DataLoader): Test data loader.
            lr (float, optional): Learning rate. Defaults to 2e-5.
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 0.01.
            num_epochs (int, optional): Number of epochs to train. Defaults to 10.
            warmup_steps_percentage (float, optional): Fraction of total steps for warmup. Defaults to 0.1.
            device (str, optional): Device to train on ('cpu' or 'cuda'). Defaults to 'cpu'.
            checkpoints_path (str, optional): Directory to save checkpoints. Defaults to "app/core/models/training-checkpoints".
        
        Raises:
            ValueError: If warmup_steps_percentage is not between 0 and 1.
        """
        if not (0.0 < warmup_steps_percentage < 1.0):
            raise ValueError("warmup_steps_percentage must be between 0 and 1")
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test__loader = test__loader
        self.num_epochs = num_epochs
        self.device = device
        self.best_checkpoints_path = os.path.join(checkpoints_path, "best_checkpoints.pt")
        self.last_checkpoints_path = os.path.join(checkpoints_path, "last_checkpoints.pt")

        os.makedirs(checkpoints_path, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler: linear warmup + cosine decay
        num_training_steps = num_epochs * len(train_loader)
        num_warmup_steps = int(num_training_steps * warmup_steps_percentage)
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Training history dictionary
        self.history = {
            'train_loss' : [],
            'valid_loss': []
        }


    def fit(
            self,
    ):
        """
        Train the model for the specified number of epochs, updating training and validation loss.
        Saves best and last checkpoints after each epoch.
        
        Returns:
            dict: Dictionary containing 'train_loss' and 'valid_loss' lists.
        """
        best_valid_loss = float("inf")

        # tqdm progress bar for epochs
        with tqdm(
            range(self.num_epochs),
            total=self.num_epochs,
            desc=f'Epoch',
            position=0,
            leave=True
        ) as epoch_pbar:

            for epoch in epoch_pbar:
                self.model.train()

                # tqdm progress bar for trainloader batches within an epoch
                with tqdm(
                    iterable=self.train_loader,
                    total=len(self.train_loader),
                    desc="Train Batch",
                    position=1,
                    leave=False,
                ) as train_batch_pbar:
                    total_loss = 0
                    epoch_lrs = []
                    for batch in train_batch_pbar:
                        pixel_values, labels = batch
                        self.optimizer.zero_grad()
                        out = self.model(
                            pixel_values=pixel_values,
                            labels=labels,
                        )
                        loss: torch.Tensor = out.loss
                        loss.backward()
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        total_loss += loss.item()
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        epoch_lrs.append(current_lr)
                        train_batch_pbar.set_postfix_str(
                            f"Running Loss = {loss.item():.4f} | LR = {current_lr:.6f}"
                        )
                        # train_batch_pbar.update()
                    
                # Calculate average training loss
                average_train_loss = total_loss/len(self.train_loader)
                self.history["train_loss"].append(average_train_loss)
                average_epoch_lr = sum(epoch_lrs) / len(epoch_lrs)

                self.model.eval()

                with torch.no_grad():
                    # tqdm progress bar for validloader batches within an epoch
                    with tqdm(
                        iterable=self.valid_loader,
                        total=len(self.valid_loader),
                        desc="Validation Batch",
                        position=1,
                        leave=False,
                    ) as valid_batch_pbar:
                        total_loss = 0
                        for batch in valid_batch_pbar:
                            pixel_values, labels = batch
                            out = self.model( 
                                pixel_values=pixel_values,
                                labels=labels,
                            )
                            loss = out.loss
                            total_loss += loss.item()
                            valid_batch_pbar.set_postfix_str(
                                f"Running Loss = {loss.item():.4f}"
                            )
                            # valid_batch_pbar.update()

                average_valid_loss = total_loss/len(self.valid_loader)
                self.history["valid_loss"].append(average_valid_loss)

                if average_valid_loss < best_valid_loss:
                    best_valid_loss = average_valid_loss
                    torch.save({
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.lr_scheduler.state_dict(),
                        "history": self.history,
                    }, self.best_checkpoints_path)
                torch.save({
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.lr_scheduler.state_dict(),
                        "history": self.history,
                    }, self.last_checkpoints_path)

                # Progress bar update
                epoch_pbar.set_postfix_str(
                    f"Train Loss = {average_train_loss:.4f} | Valid Loss = {average_valid_loss:.4f} | Avg LR = {average_epoch_lr:.6f}"
                )

        return self.history


    def load_training_checkpoints(
            self,
            state: Literal["best", "last"] = "last"
    ):
        """
        Load saved training checkpoints (best or last).
        
        Args:
            state (str, optional): 'best' or 'last'. Defaults to 'last'.
        
        Returns:
            dict: Checkpoint contents including model state, optimizer state, lr_scheduler state,
                  start epoch, and training history.
        """
        match state:
            case "last":
                path = self.last_checkpoints_path
            case "best":
                path = self.best_checkpoints_path
            case _:
                raise ValueError(f"Unknown state: {state}")
        checkpoints = torch.load(
                path,
                map_location=self.device
            )
        self.model.load_state_dict(checkpoints["model_state"])
        self.optimizer.load_state_dict(checkpoints["optimizer_state"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)        
        self.lr_scheduler.load_state_dict(checkpoints["scheduler_state"])
        start_epoch = checkpoints["epoch"] + 1
        self.history = checkpoints["history"]

        return {
            "model_state_dict": checkpoints["model_state"],
            "optimizer_state_dict": checkpoints["optimizer_state"],
            "lr_scheduler_state_dict": checkpoints["scheduler_state"],
            "start_epoch": start_epoch,
            "history": self.history
        }
    

    def save_hf_model(
            self,
            save_dir: str,
            processor: TrOCRProcessor,
    ):
        """
        Save the model and processor in HuggingFace format for future inference or fine-tuning.
        
        Args:
            save_dir (str): Directory to save the model and processor.
            processor (TrOCRProcessor): Corresponding HuggingFace processor.
        """
        self.model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)