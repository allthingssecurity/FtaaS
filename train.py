import gradio as gr
import os
import pandas as pd
import subprocess

def autotrain_llm(model_name, learning_rate, num_epochs, batch_size, block_size, trainer, warmup_ratio, weight_decay, gradient_accumulation, use_fp16, use_peft, use_int4, lora_r, lora_alpha, lora_dropout, push_to_hub, hf_token, repo_id, csv_file):
    # Save the uploaded CSV file
    if csv_file is None:
     return "No CSV file uploaded. Please upload a CSV file to continue."

# Check the size of the uploaded file
    df = pd.read_csv(csv_file.name, encoding='utf-8')
    print(df.to_string())
    # Write the DataFrame to data.csv
    csv_path = "/workspace/data/train.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    
    

    # Construct the command for training
    command = f"""
    autotrain llm --train --model {model_name} --project-name jainllama1 --data-path "/workspace/data/" --text-column text --lr {learning_rate} --batch-size {batch_size} --epochs {num_epochs} --block-size {block_size} --warmup-ratio {warmup_ratio} --lora-r {lora_r} --lora-alpha {lora_alpha} --lora-dropout {lora_dropout} --weight-decay {weight_decay} --gradient-accumulation {gradient_accumulation} {"--fp16" if use_fp16 else ""} {"--use-peft" if use_peft else ""} {"--use-int4" if use_int4 else ""} {"--push-to-hub --token " + hf_token + " --repo-id " + repo_id if push_to_hub else ""}
    """
    # Start the process
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Print each line of the standard output as it's produced
    for line in iter(process.stdout.readline, ''):
        print(line, end='')

# Wait for the process to complete
    process.wait()

# Check for errors
    if process.returncode != 0:
        print(f"The command exited with an error code: {process.returncode}")

    # Return the result
        if result.returncode == 0:
            return "Training complete!"
        else:
            return f"An error occurred: {result.stderr.decode()}"

# Define the Gradio interface
iface = gr.Interface(
    fn=autotrain_llm,
    inputs=[
        
        gr.Textbox(value="abhishek/llama-2-7b-hf-small-shards", label="Model Name"),
        gr.Slider(min=1e-5, max=1, step=1e-5, value=2e-4, label="Learning Rate"),
        gr.Slider(min=1, max=100, value=1, label="Number of Epochs"),
        gr.Slider(min=1, max=32, step=1, value=1, label="Batch Size"),
        gr.Slider(min=1, max=2048, value=8, label="Block Size"),
        gr.Dropdown(choices=["default", "sft"], label="Trainer", value="sft"),
        gr.Slider(min=0, max=1, step=0.01, value=0.1, label="Warmup Ratio"),
        gr.Slider(min=0, max=1, step=0.01, value=0.01, label="Weight Decay"),
        gr.Slider(min=1, max=32, value=4, label="Gradient Accumulation"),
        gr.Checkbox(value=True, label="Use FP16"),
        gr.Checkbox(value=True, label="Use PEFT"),
        gr.Checkbox(value=True, label="Use INT4"),
        gr.Slider(min=1, max=64, value=16, label="Lora R"),
        gr.Slider(min=1, max=64, value=32, label="Lora Alpha"),
        gr.Slider(min=0, max=1, step=0.01, value=0.05, label="Lora Dropout"),
        gr.Checkbox(value=False, label="Push to Hub"),
        gr.Textbox(value="hf_XXX", label="HF Token"),
        gr.Textbox(value="username/repo_name", label="Repo ID"),
        
        gr.inputs.File(label="Upload CSV"),
    ],
    outputs="text",

)

iface.launch(share=True,server_port=8888,debug=True)
