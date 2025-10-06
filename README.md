# Radiology Image De-identification using Generative AI 
This is a repository on the application of generative image inpainting for the de-identification of radiology images. The repository aims to provide of proof of concept that a tool integrated with segmentation models and inpainting models can suitably erase patients' sensitive information, allowing for the sharing of valuable medical materials without violating privacy-protecting regulations.

## What is Image De-identification?
In the settings of medicine, medical images play a vital role in the support of medical research as they are one of the main diagnostic materials. However, medical images always carry with them sensitive information that can be used to trace patients' identities. These information can be direct identifiers such as patients' names, social security numbers or indirect identifiers such as their jewelries, accessories and other body chracteristics. As a result, regulations are made to ensure that patients and related entites are protected by restricting and enforcing standards on how medical images can be publicly shared, accessed and stored. Consequently, this restriction has made it more challenging for researchers to utilize and share these images.

Image de-identification is a process resulting from the aforementioned regulations. Simply put, the process attempts to completely remove all identifiers with risk of identification to an individual or related parties.

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