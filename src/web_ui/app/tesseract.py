import subprocess
import numpy as np, pandas as pd

from PIL import Image, ImageDraw
from tempfile import TemporaryDirectory


def run_makebox(
    img_fname: str,
    output_dir: str
):
    subprocess.call(['tesseract', img_fname, f'{output_dir}/bbox', 'tsv'])

    df = pd.read_csv(
        f'{output_dir}/bbox.tsv', sep='\t', header=0,
        usecols=['block_num', 'left', 'top', 'width', 'height', 'text'],
    ).dropna()
    non_text = df['text'].apply(lambda txt: txt.replace(' ', '')) == ''
    return df.drop(df[non_text].index)


def get_bboxes(
    img,
    detailed: bool = True
):
    with TemporaryDirectory() as dir:
        fname = f'{dir}/img.png'
        Image.fromarray(img).save(fname)

        data = run_makebox(fname, dir)
    
    data['right'] = data['left'] + data['width']
    data['bottom'] = data['top'] + data['height']
    if detailed:
        return_res = data[['left', 'top', 'right', 'bottom']]
    else:
        bbox_cols = [
            ('left', 'min'), ('top', 'min'),
            ('right', 'max'), ('bottom', 'max')
        ]
        return_res = data.groupby('block_num').agg(['min', 'max'])[bbox_cols]
    return return_res.to_numpy()

def get_masked_ocr(
    img,
    detailed_bbox: bool = True
):
    mask = Image.new('L', (img.shape[1], img.shape[0]))
    mask_drawer = ImageDraw.Draw(mask)

    for bbox in get_bboxes(img, detailed_bbox):
        mask_drawer.rectangle(bbox, fill=255)

    return np.array(mask)