import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Classes CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Transformação de pré-processamento
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Redimensionar para 32x32 (tamanho das imagens CIFAR-10)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Função para carregar uma imagem e transformá-la
def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Adicionar batch dimension
    return image

# Caminho do modelo salvo
PATH = '/home/azorbos/VsCodeProjects/Python/IA/cnn/treinamento/cifar_net.pth'

# Definição da Rede Neural Convolucional
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Camada convolucional 1
        self.pool = nn.MaxPool2d(2, 2)   # Camada de pooling
        self.conv2 = nn.Conv2d(6, 16, 5) # Camada convolucional 2
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # Primeira camada completamente conectada
        self.fc2 = nn.Linear(120, 84)    # Segunda camada completamente conectada
        self.fc3 = nn.Linear(84, 10)     # Camada de saída

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Aplicar convolução, ReLU e pooling
        x = self.pool(F.relu(self.conv2(x)))  # Aplicar convolução, ReLU e pooling
        x = x.view(-1, 16 * 5 * 5)            # Redimensionar para a camada totalmente conectada
        x = F.relu(self.fc1(x))               # Primeira camada totalmente conectada e ReLU
        x = F.relu(self.fc2(x))               # Segunda camada totalmente conectada e ReLU
        x = self.fc3(x)                       # Camada de saída
        return x

# Carregar modelo salvo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
net.load_state_dict(torch.load(PATH))
net.eval()  # Colocar o modelo em modo de avaliação

# Função para identificar uma imagem
def identify_image(image_path):
    image = load_image(image_path).to(device)
    outputs = net(image)
    _, predicted = outputs.max(1)
    return classes[predicted[0]]
# Função para processar todas as imagens em uma pasta e salvar os resultados

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            result = identify_image(image_path)
            print(f'A imagem {image_name} foi identificada como: {result}')

            # Salvar resultado em um arquivo de texto
            result_path = os.path.join(output_folder, f'{image_name}.txt')
            with open(result_path, 'w') as f:
                f.write(result)
            
            # Mostrar a imagem com a predição
            image = Image.open(image_path)
            plt.imshow(image)
            plt.title(f'Predicted: {result}')
            plt.savefig(os.path.join(output_folder, f'{image_name}.png'))  # Salvar imagem com predição
            plt.close()

# Caminho para a pasta de entrada e saída
input_folder = '/home/azorbos/VsCodeProjects/Python/IA/cnn/images'  # Altere para o caminho da sua pasta de entrada
output_folder = '/home/azorbos/VsCodeProjects/Python/IA/cnn/results'  # Altere para o caminho da sua pasta de saída

# Processar imagens
process_images(input_folder, output_folder)
