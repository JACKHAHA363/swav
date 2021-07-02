"""
Generate Command
"""
import os
import socket
import subprocess

def find_free_network_port() -> int:
    """
    Finds a free port on localhost.
    It is useful in single-node training when we don't want to connect to a real master node but
    have to set the `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port

DATASET_PATH="/data/home/lyuchen/swav_exp/new_stl10"
EXPERIMENT_PATH="./experiments/stl/deepclusterv2_400ep_2x224_pretrain_4gpu_knn5"
#EXPERIMENT_PATH="./experiments/stl/debug"
os.makedirs(EXPERIMENT_PATH, exist_ok=True)
PORT = find_free_network_port()

# Change --nproc_per_node for single node multi-GPU
CMD = ["python", "-m", "torch.distributed.launch", "--nnodes", "1", "--nproc_per_node", "4"]
CMD += ["main_deepclusterv2.py",
"--nb_neighbor", "5",
"--data_path", DATASET_PATH,
"--nmb_crops", "2",
"--size_crops", "96",
"--min_scale_crops", "0.33",
"--max_scale_crops", "1.",
"--crops_for_assign", "0", "1",
"--temperature", "0.1",
"--hidden_mlp", "1024",
"--feat_dim", "128",
"--nmb_prototypes", "512",
"--epochs", "400",
"--batch_size", "64",
"--base_lr", "4.8",
"--final_lr", "0.0048",
"--freeze_prototypes_niters", "40000",
"--wd", "0.000001",
"--warmup_epochs", "10",
"--start_warmup", "0.3",
"--arch", "resnet50",
"--sync_bn", "pytorch",
"--dist_url", f"tcp://localhost:{PORT}",
"--dump_path", EXPERIMENT_PATH]
print(' '.join(CMD))

train_proc = subprocess.Popen(CMD)
train_proc.wait()
