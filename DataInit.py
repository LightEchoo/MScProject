#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset
#%%
class DataGenerator:
    def __init__(self, config: DictConfig, if_save: bool) -> None:
        """
        初始化 DataGenerate 类，接受一个 omegaconf 的 DictConfig 对象作为配置，并将
        所需的配置项提取为类属性。

        :param config: 配置文件，使用 omegaconf 加载的 DictConfig 对象
        """
        # 提取并保存配置项
        base_config = config.base
        data_gen_config = config.data_generation.load_data
        latency_gen_config = config.data_generation.latency_data

        self.N = base_config.N
        self.T = base_config.T
        self.T_train_val = base_config.T_train_val
        self.T_test = base_config.T_test

        self.node_load_mean_mean = data_gen_config.node_load_mean_mean
        self.node_load_mean_var = data_gen_config.node_load_mean_var
        self.node_load_iid_var = data_gen_config.node_load_iid_var
        self.node_load_ar1_theta = data_gen_config.node_load_ar1_theta

        self.node_latency_mean_mean = latency_gen_config.node_latency_mean_mean
        self.node_latency_mean_var = latency_gen_config.node_latency_mean_var
        self.node_latency_ar1_theta = latency_gen_config.node_latency_ar1_theta

        # 初始化其他属性
        self.means_loads = self._generate_means(self.node_load_mean_mean, self.node_load_mean_var)  # 生成节点的平均负载
        self.load_iid, self.load_mean_iid = self._generate_iid_data(self.node_load_iid_var, self.means_loads)  # 生成iid数据
        self.load_ar1, self.load_mean_ar1 = self._generate_ar1_data(self.node_load_ar1_theta, self.means_loads)  # 生成ar1数据

        self.means_latencies = self._generate_means(self.node_latency_mean_mean, self.node_latency_mean_var)  # 生成节点的平均延迟
        self.latency_iid, self.latency_mean_iid = self._generate_iid_data(self.node_latency_mean_mean, self.means_latencies, data_type='latency')  # 生成iid延迟数据
        self.latency_ar1, self.latency_mean_ar1 = self._generate_ar1_data(self.node_latency_ar1_theta, self.means_latencies, data_type='latency')  # 生成ar1延迟数据

        # 保存数据并打印信息
        self._save_data() if if_save else None
        self.print_data_generate_info()
        self.plot_original_means()
        self.plot_combined_data(0)
        self.plot_comparison()

    def _generate_means(self, mean: float, var: float) -> np.ndarray:
        """
        生成节点的平均负载或延迟数据。
        
        :return: 包含节点平均负载或延迟的 numpy 数组
        """
        return np.random.normal(mean, var, size=(self.N,))

    def _generate_iid_data(self, var: float, means: np.ndarray, data_type: str = 'load') -> tuple[np.ndarray, np.ndarray]:
        """
        生成 IID 数据。

        :return: 生成的 IID 数据和每个节点的均值
        """
        if data_type == 'load':
            loads = np.array([
                np.random.normal(
                    loc=means[i],
                    scale=var,
                    size=self.T
                ) for i in range(self.N)
            ])
        elif data_type == 'latency':
            loads = np.array([
                np.random.exponential(
                    scale=means[i],
                    size=self.T
                ) for i in range(self.N)
            ])
        return loads, np.mean(loads, axis=1)

    def _generate_ar1_data(self, theta: float, means: np.ndarray, data_type: str = 'load') -> tuple[np.ndarray, np.ndarray]:
        """
        生成 AR(1) 数据。

        :return: 生成的 AR(1) 数据和每个节点的均值
        """
        loads = np.zeros((self.N, self.T))

        for i in range(self.N):
            if data_type == 'load':
                # 生成 load 数据的 AR(1)
                ar1 = np.zeros(self.T)
                ar1[0] = means[i]
                for t in range(1, self.T):
                    ar1[t] = theta * ar1[t-1] + (1 - theta) * np.random.normal(means[i], np.sqrt(self.node_load_iid_var))
                loads[i] = ar1

            elif data_type == 'latency':
                # 生成 latency 数据的 AR(1)，加入不同的噪声项
                ar1 = np.zeros(self.T)
                ar1[0] = means[i]
                for t in range(1, self.T):
                    ar1[t] = theta * ar1[t-1] + (1 - theta) * np.random.exponential(means[i])
                loads[i] = ar1

        return loads, np.mean(loads, axis=1)

    def _save_data(self) -> None:
        """
        将生成的数据保存为 CSV 文件。
        """
        pd.DataFrame(self.load_iid).to_csv(data_path/'load_iid_data.csv', index=False)
        pd.DataFrame(self.load_ar1).to_csv(data_path/'load_ar1_data.csv', index=False)
        pd.DataFrame(self.latency_iid).to_csv(data_path/'latency_iid_data.csv', index=False)
        pd.DataFrame(self.latency_ar1).to_csv(data_path/'latency_ar1_data.csv', index=False)

    def print_data_generate_info(self) -> None:
        """
        打印生成的数据的基本信息。
        """
        print(f'---------- Data Generation Info ----------')
        print(f'Number of Nodes: {self.N}')
        print(f'Number of Time Steps: {self.T}')
        print(f'Number of Training and Validation Time Steps: {self.T_train_val}')
        print(f'Number of Testing Time Steps: {self.T_test}')
        print(f'Node Load Mean Mean: {self.node_load_mean_mean}')
        print(f'Node Load Mean Variance: {self.node_load_mean_var}')
        print(f'Node Load IID Variance: {self.node_load_iid_var}')
        print(f'Node Load AR1 Theta: {self.node_load_ar1_theta}')
        print(f'Node Latency Mean Mean: {self.node_latency_mean_mean}')
        print(f'Node Latency Mean Variance: {self.node_latency_mean_var}')
        print(f'Node Latency AR1 Theta: {self.node_latency_ar1_theta}')
        print(f'-----------------------------------------')


    def plot_original_means(self) -> None:
        """
        绘制生成的节点平均负载和延迟。
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.means_loads, marker='o', linestyle='-', color='b', label='means_load')
        plt.plot(self.means_latencies, marker='x', linestyle='-', color='r', label='means_latency')
        plt.title('Original Random Means of Nodes for Load and Latency')
        plt.xlabel('Node')
        plt.ylabel('Mean Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_combined_data(self, i: int) -> None:
        """
        Combines and plots the IID and AR1 data for load and latency for all nodes.
        The histograms in the two right columns are only for the i-th node, 
        while the AR1 time series line plots include all nodes. The column order is rearranged.
        
        Parameters:
        i (int): Index of the node to plot histograms for.
        """

        fig, axs = plt.subplots(2, 5, figsize=(25, 10))

        # KDE plot for each node's load IID data (left side)
        for node_index in range(self.N):
            sns.kdeplot(self.load_iid[node_index], ax=axs[0, 0], label=f'Node {node_index+1}')
        axs[0, 0].set_title("Load IID Data Distribution - All Nodes")
        axs[0, 0].legend()

        # Time series plot of load IID for all nodes
        axs[0, 1].plot(self.load_iid.T, alpha=0.6)
        axs[0, 1].set_title("Load IID Time Series - All Nodes")

        # Histogram of IID data for load (middle, only for the i-th node)
        axs[0, 2].hist(self.load_iid[i], bins=30, color='blue', alpha=0.7)
        axs[0, 2].set_title(f"Node {i+1} Load IID Histogram")

        # Time series plot of load AR1 for all nodes (next to the histogram)
        axs[0, 3].plot(self.load_ar1.T, alpha=0.6)
        axs[0, 3].set_title("Load AR1 Time Series - All Nodes")

        # Histogram of AR1 data for load (rightmost, only for the i-th node)
        axs[0, 4].hist(self.load_ar1[i], bins=30, color='orange', alpha=0.7)
        axs[0, 4].set_title(f"Node {i+1} Load AR1 Histogram")

        # KDE plot for each node's latency IID data (left side)
        for node_index in range(self.N):
            sns.kdeplot(self.latency_iid[node_index], ax=axs[1, 0], label=f'Node {node_index+1}')
        axs[1, 0].set_title("Latency IID Data Distribution - All Nodes")
        axs[1, 0].legend()

        # Time series plot of latency IID for all nodes
        axs[1, 1].plot(self.latency_iid.T, alpha=0.6)
        axs[1, 1].set_title("Latency IID Time Series - All Nodes")

        # Histogram of IID data for latency (middle, only for the i-th node)
        axs[1, 2].hist(self.latency_iid[i], bins=30, color='green', alpha=0.7)
        axs[1, 2].set_title(f"Node {i+1} Latency IID Histogram")

        # Time series plot of latency AR1 for all nodes (next to the histogram)
        axs[1, 3].plot(self.latency_ar1.T, alpha=0.6)
        axs[1, 3].set_title("Latency AR1 Time Series - All Nodes")

        # Histogram of AR1 data for latency (rightmost, only for the i-th node)
        axs[1, 4].hist(self.latency_ar1[i], bins=30, color='red', alpha=0.7)
        axs[1, 4].set_title(f"Node {i+1} Latency AR1 Histogram")

        plt.tight_layout()
        plt.savefig(f'Combined_Figure_Reordered_Node_{i+1}.png')
        plt.show()

    def plot_comparison(self) -> None:
        """
        绘制 self.means_loads, self.load_mean_iid, self.load_mean_ar1, 
        self.means_latencies, self.latency_mean_iid, self.latency_mean_ar1 的对比图。
        其中，latency 的曲线使用虚线，means 的、iid 的、ar1 的要有对应相似的表现。
        """

        plt.figure(figsize=(12, 8))

        # 绘制 Load 数据
        plt.plot(self.means_loads, marker='o', linestyle='-', color='blue', label='Load Means')
        plt.plot(self.load_mean_iid, marker='x', linestyle='-', color='cyan', label='Load IID Mean')
        plt.plot(self.load_mean_ar1, marker='s', linestyle='-', color='darkblue', label='Load AR1 Mean')

        # 绘制 Latency 数据 (使用虚线)
        plt.plot(self.means_latencies, marker='o', linestyle='--', color='red', label='Latency Means')
        plt.plot(self.latency_mean_iid, marker='x', linestyle='--', color='orange', label='Latency IID Mean')
        plt.plot(self.latency_mean_ar1, marker='s', linestyle='--', color='darkred', label='Latency AR1 Mean')

        # 图例和标签
        plt.title('Comparison of Means, IID Mean, and AR1 Mean for Load and Latency')
        plt.xlabel('Node')
        plt.ylabel('Mean Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
#%%
class DataManager:
    def __init__(self, config: DictConfig, data_type: str) -> None:
        """
        初始化 TrainValidManage 类，接受一个 omegaconf 的 DictConfig 对象作为配置，并将
        所需的配置项提取为类属性。

        :param config: 配置文件，使用 omegaconf 加载的 DictConfig 对象
        """
        # 提取并保存配置项
        base_config = config.base
        exp4_config = config.exp4

        # self.data_type = base_config.data_type  # 从 base_config 提取 data_type
        self.data_type = data_type
        self.device = exp4_config.device  # 从 exp4_config 提取设备信息
        self.batch_size = exp4_config.batch_size
        self.num_workers = exp4_config.num_workers
        self.N = base_config.N
        self.T = base_config.T
        self.T_train = base_config.T_train
        self.T_val = base_config.T_val
        self.train_ratio = base_config.train_ratio
        self.T_train_val = base_config.T_train_val
        self.T_test = base_config.T_test
        self.seq_length = exp4_config.seq_length

        # 加载数据
        self.load_iid = pd.read_csv(data_path/'load_iid_data.csv').values
        self.load_ar1 = pd.read_csv(data_path/'load_ar1_data.csv').values
        self.latency_iid = pd.read_csv(data_path/'latency_iid_data.csv').values
        self.latency_ar1 = pd.read_csv(data_path/'latency_ar1_data.csv').values

        # 根据数据类型选择数据
        match self.data_type:
            case 'load_iid':
                self.data_np = self.load_iid
            case 'load_ar1':
                self.data_np = self.load_ar1
            case 'latency_iid':
                self.data_np = self.latency_iid
            case 'latency_ar1':
                self.data_np = self.latency_ar1
            case _:
                raise ValueError(f"Unknown data_type: {self.data_type}")

        # 将数据转换为 PyTorch 张量
        self.data_tensor = torch.tensor(self.data_np, device=self.device, dtype=torch.float32)

        # 划分np.array的训练集、验证集和测试集
        self.train_val_data_np = self.data_np[:, :self.T_train_val]
        self.train_data_np = self.data_np[:, :self.T_train]
        self.val_data_np = self.data_np[:, self.T_train:self.T_train_val]
        self.test_data_np = self.data_np[:, self.T_train_val:]

        # 储存tensor的训练集、验证集和测试集
        self.train_val_data_tensor = torch.tensor(self.train_val_data_np, device=self.device, dtype=torch.float32)
        self.train_data_tensor = torch.tensor(self.train_data_np, device=self.device, dtype=torch.float32)
        self.val_data_tensor = torch.tensor(self.val_data_np, device=self.device, dtype=torch.float32)
        self.test_data_tensor = torch.tensor(self.test_data_np, device=self.device, dtype=torch.float32)

        # 创建训练集、验证集、训练验证集的序列数据
        self.train_val_x, self.train_val_y = self._create_sequences(self.data_np, 'train_val')
        self.train_x, self.train_y = self._create_sequences(self.data_np, 'train')
        self.val_x, self.val_y = self._create_sequences(self.data_np, 'val')

        # 创建TensorDataset，用于创建DataLoader
        self.train_val_dataset = TensorDataset(self.train_val_x, self.train_val_y)
        self.train_dataset = TensorDataset(self.train_x, self.train_y)
        self.val_dataset = TensorDataset(self.val_x, self.val_y)

        # 创建数据加载器
        self.train_val_dataloader = DataLoader(self.train_val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # 创建GNN的边索引
        self.edge_index_tensor = torch.tensor(
            np.array([(i, j) for i in range(self.N) for j in range(self.N)]).T,
            dtype=torch.long)  # 默认全连接图

        # 打印信息
        self.print_train_valid_info()

    def _create_sequences(self, data: np.ndarray, split_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据数据创建序列数据。
        
        :param data: 输入的numpy数组数据
        :param split_type: 数据的切分类型（'train', 'val', 'train_val'）
        :return: 生成的输入序列张量和目标序列张量
        """
        x, y = [], []

        if split_type == 'train':
            # 训练集，从1到8000 (用1-20预测21)
            for i in range(self.seq_length, self.T_train):
                x.append(data[:, i - self.seq_length:i].T)
                y.append(data[:, i])

        elif split_type == 'val':
            # 验证集，从8001到10000 (用7981-8000预测8001)
            for i in range(self.T_train, self.T_train_val):
                x.append(data[:, i - self.seq_length:i].T)
                y.append(data[:, i])

        elif split_type == 'train_val':
            # 训练验证集，从1到10000 (用1-20预测21)
            for i in range(self.seq_length, self.T_train_val):
                x.append(data[:, i - self.seq_length:i].T)
                y.append(data[:, i])

        return torch.tensor(np.array(x)), torch.tensor(np.array(y))

    def print_train_valid_info(self) -> None:
        """
        打印训练和验证数据的信息。
        """
        print(f'================= Data Info =================')
        print(f'----------------- Base Info-----------------')
        print(f'data_type: {self.data_type}')
        print(f'device: {self.device}')
        print(f'batch_size: {self.batch_size}')
        print(f'num_workers: {self.num_workers}')
        print(f'N: {self.N}')
        print(f'T: {self.T}')
        print(f'T_train: {self.T_train}')
        print(f'T_val: {self.T_val}')
        print(f'train_ratio: {self.train_ratio}')
        print(f'T_train_val: {self.T_train_val}')
        print(f'T_test: {self.T_test}')
        print(f'seq_length: {self.seq_length}')
        
        print(f'----------------- Data Info -----------------')
        print(f'load_iid.shape: {self.load_iid.shape}')
        print(f'load_ar1.shape: {self.load_ar1.shape}')
        print(f'latency_iid.shape: {self.latency_iid.shape}')
        print(f'latency_ar1.shape: {self.latency_ar1.shape}')
        print(f'data_np.shape: {self.data_np.shape}')
        print(f'data_tensor.shape: {self.data_tensor.shape}')

        print(f'----------------- Split Info -----------------')
        print('train_val_data_np.shape:', self.train_val_data_np.shape)
        print('train_data_np.shape:', self.train_data_np.shape)
        print('val_data_np.shape:', self.val_data_np.shape)
        print('test_data_np.shape:', self.test_data_np.shape)
        print('train_val_data_tensor.shape:', self.train_val_data_tensor.shape)
        print('train_data_tensor.shape:', self.train_data_tensor.shape)
        print('val_data_tensor.shape:', self.val_data_tensor.shape)
        print('test_data_tensor.shape:', self.test_data_tensor.shape)
        
        print(f'----------------- Sequence Info -----------------')
        print('train_val_x.shape:', self.train_val_x.shape)
        print('train_val_y.shape:', self.train_val_y.shape)
        print('train_x.shape:', self.train_x.shape)
        print('train_y.shape:', self.train_y.shape)
        print('val_x.shape:', self.val_x.shape)
        print('val_y.shape:', self.val_y.shape)
        
        print(f'----------------- DataLoader Info -----------------')
        self.print_dataloader_info(self.train_val_dataloader, 'Train-Val')
        self.print_dataloader_info(self.train_dataloader, 'Train')
        self.print_dataloader_info(self.val_dataloader, 'Val')
        
        print(f'----------------- Edge Index Info -----------------')
        print('edge_index_tensor.shape:', self.edge_index_tensor.shape)
        
        print(f'===================== End Info =====================')

    def print_dataloader_info(self, dataloader: DataLoader, title: str) -> None:
        """
        打印 DataLoader 的信息。
    
        :param dataloader: DataLoader 对象
        :param title: 信息标题
        """
        print(f'{"-"*10} {title} Dataloader Info {"-"*10}')

        # 打印头部行，展示 min/max 和 shape 信息
        print(f'{"Batch":<10} {"x.min":<8} {"x.max":<8} {"y.min":<8} {"y.max":<8} {"x.shape:torch.Size":<18} {"y.shape:torch.Size":<18}')

        # 打印每个批次的信息
        for i, (x, y) in enumerate(dataloader):
            if i % 30 == 0 or i == len(dataloader) - 1:
                x_shape_str = str(list(x.shape))  # 使用 list 来缩短 shape 显示
                y_shape_str = str(list(y.shape))

                print(f'{i + 1:>5}/{len(dataloader):<5} '
                      f'{x.min():<8.4f} {x.max():<8.4f} '
                      f'{y.min():<8.4f} {y.max():<8.4f} '
                      f'{x_shape_str:<18} {y_shape_str:<18}')

    def plot_range_data(self, data: np.ndarray, start: int = None, end: int = None, title: str = 'Load Data') -> None:
        """
        绘制指定范围内的数据。

        :param data: 输入数据
        :param start: 开始时间步，默认为0
        :param end: 结束时间步，默认为数据结束
        :param title: 图像标题
        """
        start = 0 if start is None else start
        end = data.shape[1] if end is None else end

        time_steps = np.arange(start, end)
        plt.figure(figsize=(12, 6))
        for i in range(data.shape[0]):
            plt.plot(time_steps, data[i, start:end], label=f'Node {i}')
        plt.title(f'{title} - Nodes {0}-{data.shape[0]}')
        plt.xlabel('Time')
        plt.ylabel('Load')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

#%%
def manage_and_save_data(config: DictConfig, data_type: str, plot_start_node: int, plot_end_node: int, data_path: Path) -> None:
    """
    生成数据，绘图，并保存数据管理对象。

    :param config: 配置对象
    :param data_type: 数据类型，例如 'load_iid', 'load_ar1', 'latency_iid', 'latency_ar1'
    :param data_path: 数据保存的路径
    :param plot_title: 绘图的标题
    """
    # 数据生成
    data_manager = DataManager(config, data_type)

    # 绘图
    data_manager.plot_range_data(data_manager.data_np[plot_start_node:plot_end_node, :], title=f'{data_type} Data')

    # 保存数据
    with open(data_path/f'{data_type}_data_manage.pkl', 'wb') as f:
        pickle.dump(data_manager, f)
#%%
if __name__ == '__main__':

    # 加载 config.yaml 文件
    config = OmegaConf.load("config/config.yaml")
    global_path = Path(config.path.global_path)
    data_path = global_path / 'Data'
    # 打印完整的配置内容
    print(f'---------- Config Info ----------')
    print(OmegaConf.to_yaml(config))
    # 打印全局路径和数据路径
    print(f'---------- Path Info ----------')
    print(f'Global Path: {global_path}')
    print(f'Data Path: {data_path}')

    # 数据生成
    data_generate = DataGenerator(config, if_save=True)

    # 数据管理
    manage_and_save_data(config, 'load_iid', 0, 3, data_path)
    manage_and_save_data(config, 'load_ar1', 0, 3, data_path)
    manage_and_save_data(config, 'latency_iid', 0, 3, data_path)
    manage_and_save_data(config, 'latency_ar1', 0, 3, data_path)
    
    
    
    # load_iid_data_manage = DataManager(config, 'load_iid')
    # load_iid_data_manage.plot_range_data(load_iid_data_manage.data_np[:3, :], title='iid Data')
    # with open(data_path/'load_iid_data_manage.pkl', 'wb') as f:
    #     pickle.dump(load_iid_data_manage, f)
    # 
    # load_ar1_data_manage = DataManager(config, 'load_ar1')
    # load_ar1_data_manage.plot_range_data(load_ar1_data_manage.data_np[:3, :], title='ar1 Data')
    # with open(data_path/'load_ar1_data_manage.pkl', 'wb') as f:
    #     pickle.dump(load_ar1_data_manage, f)
    # 
    # latency_iid_data_manage = DataManager(config, 'latency_iid')
    # latency_iid_data_manage.plot_range_data(latency_iid_data_manage.data_np[:3, :], title='iid Data')
    # with open(data_path/'latency_iid_data_manage.pkl', 'wb') as f:
    #     pickle.dump(latency_iid_data_manage, f)
    # 
    # latency_ar1_data_manage = DataManager(config, 'latency_ar1')
    # latency_ar1_data_manage.plot_range_data(latency_ar1_data_manage.data_np[:3, :], title='ar1 Data')
    # with open(data_path/'latency_ar1_data_manage.pkl', 'wb') as f:
    #     pickle.dump(latency_ar1_data_manage, f)
#%%
