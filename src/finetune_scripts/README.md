# Inpainting fine-tuning for Stable Diffusion

This directory comprises of scripts for fine-tuning and evaluating Stable Diffusion (SD) for inpainting task. The script uses AdamW to optimize parameters.

## Requirements
Python version 3.12 - 3.13.2 is recommended. The required libraries are listed in [`./requirements`](./requirements.txt).\
To install all dependencies (a virtual environment is recommended), run:
```bash
pip install -r requirements.txt
```

## Quick Start
First, set up some environment variables:
```bash
export MODEl="stabilityai/stable-diffusion-2-inpainting"
export TRAIN_DIR="path/to/train-data/dir"
export TEST_DIR="path/to/test-data/dir"
export OUTPUT_DIR="path/to/output/dir"
```
or if using Powershell:
```powershell
$MODEl="stabilityai/stable-diffusion-2-inpainting"
$DATA_DIR="path/to/data/dir"
$TEST_DIR="path/to/test-data/dir"
$OUTPUT_DIR="path/to/output/dir"
```

Then run the script using accelerate (remove newlines when pasting the command):
```bash
accelerate launch train_diffusion.py
    --pretrained_model_name_or_path $MODEL
    --instance_data_dir $TRAIN_DIR
    --test_data_dir $TEST_DIR
    --output_dir $OUTPUT_DIR
    --resolution 512
    --train_batch_size 4
    --learning_rate 1e-6
    --lr_scheduler "constant"
    --max_train_steps 1000
    --seed 0
    --checkpointing_steps 500
```

To view the losses, run:
```bash
tensorboard --logdir $OUTPUT_DIR
```
and access the dashboard at `http://localhost:6006`.

## Speeding up training and optimize GPU memory usage
If the model/optimizer is too big to fit onto the GPU, here are some optimization strategies:
- Gradient accumulation: If you want to keep the initial batch size, you can use `--gradient_accumulation_steps` to simulate a bigger batch size with a smaller actual batch size. For example, to simulate a batch size 32 using actual batch size of 4, set `--gradient_acummulation_steps 8`.
- 8-bit optimizers: If AdamW is too big to fit onto the GPU, consider using 8-bit AdamW by setting `--use_8bit_adam`. To use this, you must install [bitsandbytes](https://huggingface.co/docs/diffusers/quantization/bitsandbytes).
- Mixed-precision training: Mixed precision can help reduce memory usage as well as speeding up training, see [Mixed-precision training](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#floating-data-types) for more information. This can be set with `--mixed_precision ['fp16' or 'bf16']`, `'bf16'` is only available for Ampere GPUs.
- Gradient checkpointing: As a last resort, gradient checkpointing further reduces memory usage with a large overhead of training time. This can be enabled with `--gradient_checkpointing`.

## Resume from checkpoint
You can resume from a checkpoint saved during training by setting `--resume_from_checkpoint [path/to/checkpoint]`. This is useful if training stopped abruptly.

You may also use this to continue training for extra steps. However, this will not work as expected for non-linear learning rate scheduler.