import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file, load_file


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPTrainer:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        self.model = MLPRegressor(input_dim, hidden_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def fit(self, X, y, epochs: int = 20, batch_size: int = 32) -> None:
        """
        X: [N, input_dim]
        y: [N, 1] 或 [N]
        """
        self.model.train()
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        for _ in range(epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        """
        X: [input_dim] 或 [N, input_dim]
        返回 numpy 一维数组
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            out = self.model(x).squeeze(-1).cpu().numpy()
        return out

    def save_safetensor(self, path: str) -> None:
        save_file(self.model.state_dict(), path)

    def load_safetensor(self, path: str) -> None:
        state = load_file(path)
        self.model.load_state_dict(state)

