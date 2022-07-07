import torchvision.transforms as transforms


train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
    transforms.Grayscale(),
    transforms.Resize(size=(32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5,), std=(.5,))
])

eval_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(size=(32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5,), std=(.5,))
])
