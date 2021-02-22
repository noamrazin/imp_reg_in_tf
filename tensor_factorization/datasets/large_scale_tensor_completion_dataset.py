import torch
import torch.utils.data


class LargeScaleTensorCompletionDataset(torch.utils.data.Dataset):

    def __init__(self, tensor_indices: torch.Tensor, targets: torch.Tensor, target_tensor_modes_dim: int,
                 additional_metadata: dict):
        self.tensor_indices = tensor_indices
        self.targets = targets
        self.target_tensor_modes_dim = target_tensor_modes_dim
        self.additional_metadata = additional_metadata

    def __getitem__(self, index: int):
        return self.tensor_indices[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.targets)

    def to_device(self, device):
        self.tensor_indices = self.tensor_indices.to(device)
        self.targets = self.targets.to(device)

    def save(self, path: str):
        state_dict = {
            "tensor_indices": self.tensor_indices.cpu(),
            "targets": self.targets.cpu(),
            "target_tensor_modes_dim": self.target_tensor_modes_dim,
            "additional_metadata": self.additional_metadata
        }
        torch.save(state_dict, path)

    @staticmethod
    def load(path: str, device=torch.device("cpu")):
        state_dict = torch.load(path, map_location=device)
        return LargeScaleTensorCompletionDataset(state_dict["tensor_indices"],
                                                 state_dict["targets"],
                                                 state_dict["target_tensor_modes_dim"],
                                                 state_dict["additional_metadata"])
