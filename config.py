from dataclasses import dataclass, field
import json
import yaml


@dataclass
class DataGenerateConfig:
    mean_load: float = 50.0  # Mean load of nodes' load
    var_load: float = 10.0  # Variance load of nodes' load
    iid_var: float = 1.0  # Variance for iid data
    theta: float = 0.9  # AR(1) parameter


@dataclass
class ARConfig:
    order: int = 5  # AR order


@dataclass
class LSTMConfig:
    hidden_size: int = 64  # Hidden size
    num_layers: int = 4  # Number of layers


@dataclass
class GATConfig:
    hidden_size: int = 32  # Hidden size
    num_heads: int = 8  # Number of attention heads
    num_gat_layers: int = 3  # Number of GAT layers


@dataclass
class GNNConfig:
    hidden_size: int = 32
    num_layers: int = 3


@dataclass
class Config:
    N: int = 10  # Number of nodes
    T: int = 11000  # Total time steps

    T_train_val: int = 10000  # Training and validation time steps
    train_ratio: float = 0.8  # Training ratio
    T_train: int = 8000  # Training time steps
    T_val: int = 2000  # Validation time steps

    T_test: int = 1000  # Test time steps
    data_type: str = 'ar1'  # 'iid' or 'ar1'

    batch_size: int = 64  # Batch size
    seq_length: int = 20  # Sequence length
    input_size: int = 10  # Input size
    output_size: int = 10  # Output size
    learning_rate: float = 0.001  # Learning rate
    num_epochs: int = 100  # Number of epochs
    num_workers: int = 24  # Number of workers for DataLoader
    device: str = 'cuda'  # Device
    mix_precision: bool = True  # Mixed precision training

    # Early stopping parameters
    patience_epochs: int = 6  # Stop training if no improvement for 'patience_epochs' epochs
    min_delta: float = 1e-2  # Minimum change to qualify as an improvement

    # Scheduler parameters
    mode: str = 'min'  # 'min' means the lower the better, 'max' means the higher the better
    factor: float = 0.1  # Scaling factor for learning rate scheduler
    patience_lr: int = 2  # Number of epochs without improvement before scaling down the learning rate
    min_lr: float = 1e-6  # Minimum learning rate
    threshold: float = 1e-2  # Minimum change to qualify as an improvement

    # Use default_factory for complex types
    dg_config: DataGenerateConfig = field(default_factory=DataGenerateConfig)
    ar_config: ARConfig = field(default_factory=ARConfig)
    lstm_config: LSTMConfig = field(default_factory=LSTMConfig)
    gat_config: GATConfig = field(default_factory=GATConfig)
    gnn_config: GNNConfig = field(default_factory=GNNConfig)

    def print_config_info(self):
        print('-----------------Config Info-----------------')
        self._recursive_print(vars(self))

    def _recursive_print(self, config_dict, indent=0):
        for key, value in config_dict.items():
            if isinstance(value, (DataGenerateConfig, ARConfig, LSTMConfig, GATConfig, GNNConfig)):
                print(" " * indent + f"{key}:")
                self._recursive_print(vars(value), indent + 4)
            else:
                print(" " * indent + f"{key}: {value}")

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_json(self, file_path):
        # 使用 asdict() 将 dataclass 转换为字典，并递归转换内部 dataclass 对象
        def convert_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {key: convert_to_dict(value) for key, value in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            else:
                return obj

        config_dict = convert_to_dict(self)
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, file_path):
        # 使用 asdict() 将 dataclass 转换为字典，并递归转换内部 dataclass 对象
        def convert_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {key: convert_to_dict(value) for key, value in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            else:
                return obj

        config_dict = convert_to_dict(self)
        with open(file_path, 'w') as f:
            yaml.safe_dump(config_dict, f, indent=4)

# Automatically export default config to config.json when config.py is executed
if __name__ == "__main__":
    default_config = Config()
    default_config.to_json('config.json')
    print("Default config exported to config.json")