import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('unbalanced_2.pth')
model.eval()

image_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]
)

def predict_image(image):
    image = Image.open(image)
    image_tensor = image_transform(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    input = Variable(image_tensor).to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    print(index)

predict_image('dataset/Swedish elkhound/img.png')
predict_image('dataset/Swedish elkhound/121642615_1302428693433846_4544118922928741841_n.jpg')
predict_image('dataset/Golden retriever/122829425_688546388486096_4749832367830120918_n.jpg')
predict_image('golden_interent.png')


#import os
#
#for subdir, dirs, files in os.walk('dataset'):
#    for file in files:
#        #print os.path.join(subdir, file)
#        filepath = subdir + os.sep + file
#
#        print(filepath)
#        predict_image(filepath)
#