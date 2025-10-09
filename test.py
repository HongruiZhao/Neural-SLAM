import torch
import torch.multiprocessing as mp
import time

def worker_process(rank, device_id):
    """A PyTorch worker function that runs on a specific GPU."""
    # Set the visible device for this process
    torch.cuda.set_device(device_id)

    # Create a dummy model and data
    model = torch.nn.Linear(10, 10).cuda()
    data = torch.randn(100, 10).cuda()
    
    start_time = time.time()
    # Perform a few operations
    for _ in range(1000):
        _ = model(data)
    
    end_time = time.time()
    print(f"Process {rank}: Finished in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    # Start the MPS daemon before running this script
    # e.g., in a separate terminal: nvidia-cuda-mps-control -d

    # Number of parallel processes to run
    num_processes = 4
    gpu_device_id = 0  # Assuming you want to use GPU 0

    # Start the processes
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=worker_process, args=(i, gpu_device_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()