# Web UI demonstration

This is a Web UI developed with gradio to demonstrate our pipeline of medical image de-identification. The UI provides two functionalities:
- Segement anything: Images can be segmented with SaM to create mask for inpainting.
- Inpainting: After obtaining a mask, you can inpaint the image with a pretrained LaMa or our fine-tuned Stable Diffusion v2 for inpainting.

To host the UI on your local machine, follow the below steps.

## Prerequisites
Python is necessary. Python 3.12 is recommended, but early minor releases of Python 3.13 might also work.

You also need [Tesseract](https://github.com/tesseract-ocr/tesseract) (Tesseract 5 is recommended). Visit [Tesseract installation](https://tesseract-ocr.github.io/tessdoc/Installation.html) and navigate to your OS of choice to install the appropriate binary package. Once installed, check if Tesseract is in your system's PATH variable. If not, add it to PATH. You can also check this by executing the following command in your terminal:
```bash
tesseract --version
```

## Dependencies Installation and Quick Start
You should create a fresh virtual environment before installing the requirements.
1. Move to this directory and install PyTorch. Visit [PyTorch](https://pytorch.org/get-started/locally/), choose the suitable configurations for your system and run the generated command. For example, the command for installing PyTorch using pip on a system with CUDA 12.8 is:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
2. Install other dependencies with:
```bash
pip install -r requirements.txt
```
3. Navigate to the imgaug.py module (if you are using a virtual environment, this should be at `path-to-venv/Lib/imgaug/imgaug.py`). Replace lines 45-47 with the following lines:
```python
NP_FLOAT_TYPES = {np.float16, np.float32, np.float64}
NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64}
NP_UINT_TYPES = {np.uint8, np.uint16, np.uint32, np.uint64}
```
4. Visit [lama_and_sam_ckpt](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A) and install everything within the directory. Then unzip everything to [assets/models](../../assets/models/). Similarly, visit [sd_ckpt](https://drive.google.com/file/d/15LHfS5-3LbWefuTqW-OIVFlaXKp5OZ12/view?usp=sharing) to install the checkpoint and unzip to [assets/models](../../assets/models/). You should have the following directory tree after downloading everything:
```
assets/
├── ...
└── models/
    ├── lama
    ├── sd_inpaint
    ├── .gitignore
    ├── sam_vit_h_4b8939.pth
    └── sttn.pth
```
5. Move to [`app`](./app/) directory and run this command in the terminal:
```
python app.py
```
If the app ran successfully, you can finally access and interact with the Web UI at `http://localhost:7860`.