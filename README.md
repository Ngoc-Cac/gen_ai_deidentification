# Radiology Image De-identification using Generative AI 
This is a repository on the application of generative image inpainting for the de-identification of radiology images. The repository aims to provide of proof of concept that a tool integrated with segmentation models and inpainting models can suitably erase patients' sensitive information, allowing for the sharing of valuable medical materials without violating privacy-protecting regulations.

## Structure of repository
This repository involves two main activities:
### 1. The fine-tuning of diffusion models
The project involves a fine-tuning of Stable Diffusion for radiology image inpainting. The intent is to investigate the model's capability of capturing anatomical structures.

The materials related to this activity is provided in [`src/finetune_scripts`](src/finetune_scripts/README.md).
### 2. The integration of a multi-model pipeline
After obtaining a fine-tuned Stable Diffusion for radiology image inpainting, the project combines different models:
- Segment Anything (SaM): A segmenation model for sensitive information detection and mask generation.
- TesseractOCR: An OCR model to enhance the detection of sensitive textual information.
- LaMa and Stable Diffusion for sensitive information removal.

Further details on this activity is provided in [`src/web_ui`](src/web_ui/README.md).