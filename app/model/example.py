import sys
from PIL import Image
from inference import inference

assert(len(sys.argv) == 2)
img_name = sys.argv[1]

input_image = Image.open(img_name)

clf = inference(input_image)

print(clf)

