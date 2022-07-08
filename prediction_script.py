import argparse
import os
import torch

from PIL import Image

from models import CNN
from models import ModelWrapper
from utils.transforms import eval_transform
from utils import label_to_class


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', required=True, type=str)
    parser.add_argument('--imgs-dir', required=True, type=str)
    parser.add_argument('--n_filters', default=10, type=int)
    parser.add_argument('-p', default=0.5, type=float)

    args = parser.parse_args()

    model_path = args.model_path
    imgs_dir = args.imgs_dir
    n_filters = args.n_filters
    p = args.p

    model = CNN(n_filters=n_filters, p=p)
    mw = ModelWrapper(model, None, None)
    mw.load(model_path)

    filenames = []
    imgs = []

    for filename in os.listdir(imgs_dir):
        filenames.append(filename)
        filepath = os.path.join(imgs_dir, filename)

        with Image.open(filepath) as img:
            img_transformed = eval_transform(img)
            imgs.append(img_transformed)

    imgs = torch.stack(imgs)
    preds = torch.argmax(mw.predict(imgs), dim=1)

    for filename, pred in zip(filenames, preds):
        predicted_class = label_to_class(pred)
        print(f"Filename: {filename} is an instance of a class: {predicted_class}")


if __name__ == '__main__':
    main()
