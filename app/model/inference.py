import torch
from torchvision import transforms
from PIL import Image

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
if torch.cuda.is_available():
    model.to("cuda")
model.eval()
with open("model/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


def inference(input_image: Image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.no_grad():
        # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
        output = model(input_batch)
       
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    catid = torch.argmax(probabilities)

    return categories[catid]

