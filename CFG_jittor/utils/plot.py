import matplotlib.pyplot as plt
import re
import os

def plot_loss(log_filepath="training_progress.log", output_image_filepath="training_loss_plot.png"):
    global_steps = []
    losses = []
    epoch_times = {}  

    full_log_line_pattern = re.compile(
        r".*?INFO - "
        r"Epoch\s+(?P<epoch>\d+)/(?P<epochs_total>\d+)\s+"
        r"Step\s+(?P<step_in_epoch>\d+)/(?P<steps_total_in_epoch>\d+)\s+"
        r"Loss\s+(?P<loss>[0-9.]+)\s+"
        r"Time\s+(?P<time>[0-9.]+)"
    )

    with open(log_filepath, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            match = full_log_line_pattern.search(line)
            if match:
                data = match.groupdict()
                epoch = int(data['epoch'])
                step_in_epoch = int(data['step_in_epoch'])
                total_steps_in_epoch = int(data['steps_total_in_epoch'])

                current_global_step = (epoch - 1) * total_steps_in_epoch + step_in_epoch

                loss = float(data['loss'])
                raw_time = float(data['time'])  

                global_steps.append(current_global_step)
                losses.append(loss)

                if epoch not in epoch_times:
                    epoch_times[epoch] = 0.0
                epoch_times[epoch] += raw_time

    if output_image_filepath:
        output_dir = os.path.dirname(os.path.abspath(output_image_filepath))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        loss_output_path = os.path.abspath(output_image_filepath) 
        plt.figure(figsize=(12, 6))  
        plt.plot(global_steps, losses, marker='.', linestyle='-', markersize=4, label='Loss')

        plt.title("Training Loss over Global Steps")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(loss_output_path)
        plt.close()  

        epoch_time_output_path =  "training_epoch_time_plot.png" 
        output_dir = os.path.dirname(os.path.abspath(output_image_filepath))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        epoch_time_output_path = os.path.join(output_dir, epoch_time_output_path) 

        sorted_epochs = sorted(epoch_times.keys())
        total_epoch_times = [epoch_times[e] for e in sorted_epochs]

        plt.figure(figsize=(12, 6))

        plt.plot(sorted_epochs, total_epoch_times, marker='o', linestyle='-', label='Total Time per Epoch')

        plt.title("Training Time per Epoch")
        plt.xlabel("Epoch Number")
        plt.ylabel("Total Time (seconds)")
        plt.legend()
        plt.grid(True)
        if len(sorted_epochs) > 0:
            plt.xticks(sorted_epochs)

        plt.tight_layout()

        plt.savefig(epoch_time_output_path)
        print(f"Total Time per Epoch is saved to: {epoch_time_output_path}")
        plt.close()
