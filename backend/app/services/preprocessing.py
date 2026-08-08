import torchvision.transforms as transforms

from backend.app.core.config import IMAGE_SIZE


_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)

GOAD_TRANSFORMS = [
    transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), _NORMALIZE]),
    transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), _NORMALIZE]),
    transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), _NORMALIZE]),
    transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomRotation((90, 90)), transforms.ToTensor(), _NORMALIZE]),
    transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomRotation((180, 180)), transforms.ToTensor(), _NORMALIZE]),
    transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomRotation((270, 270)), transforms.ToTensor(), _NORMALIZE]),
    transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), _NORMALIZE]),
    transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomRotation((90, 90)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), _NORMALIZE]),
]

GOAD_TRANSFORM_COUNT = len(GOAD_TRANSFORMS)
