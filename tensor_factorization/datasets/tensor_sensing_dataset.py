import torch
import torch.utils.data


class TensorSensingDataset(torch.utils.data.Dataset):

    def __init__(self, sensing_tensors: torch.Tensor, target_tensor: torch.Tensor, target_cp_rank: int):
        self.sensing_tensors = sensing_tensors
        self.target_tensor = target_tensor
        self.targets = (sensing_tensors * target_tensor.unsqueeze(0)).view(sensing_tensors.shape[0], -1).sum(dim=1)
        self.target_cp_rank = target_cp_rank

    def __getitem__(self, index: int):
        return self.sensing_tensors[index], self.targets[index]

    def __len__(self) -> int:
        return self.sensing_tensors.shape[0]

    def to_device(self, device):
        self.sensing_tensors = self.sensing_tensors.to(device)
        self.target_tensor = self.target_tensor.to(device)
        self.targets = self.targets.to(device)

    def save(self, path: str):
        state_dict = {
            "sensing_tensors": self.sensing_tensors.cpu(),
            "target_tensor": self.target_tensor.cpu(),
            "target_cp_rank": self.target_cp_rank
        }
        torch.save(state_dict, path)

    @staticmethod
    def load(path: str, device=torch.device("cpu")):
        state_dict = torch.load(path, map_location=device)
        return TensorSensingDataset(sensing_tensors=state_dict["sensing_tensors"],
                                    target_tensor=state_dict["target_tensor"],
                                    target_cp_rank=state_dict["target_cp_rank"])
