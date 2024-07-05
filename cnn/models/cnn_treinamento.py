import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Verifica se a GPU está disponível
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# Transformação dos dados
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Carregamento dos dados de treinamento e teste
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.inutils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

net = Net().to(device)  # Move o modelo para a GPU, se disponível

# Definição da função de custo e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Função para mostrar uma imagem
def imshow(img):
    img = img / 2 + 0.5  # Desnormaliza
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Função de avaliação
def evaluate(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move para a GPU
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Treinamento do modelo
for epoch in range(5):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move para a GPU

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = net(inputs)  # Forward
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward
        optimizer.step()  # Optimize

        running_loss += loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # Avaliação do modelo no conjunto de teste após cada época
    accuracy = evaluate(net, testloader)
    print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)

print('Finished Training')

# Avaliação final do modelo
accuracy = evaluate(net, testloader)
print('Final accuracy of the network on the 10000 test images: %d %%' % accuracy)

# Salvar o modelo treinado
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# Visualização de algumas imagens do conjunto de teste e suas previsões
dataiter = iter(testloader)
images, labels = next(dataiter)  # Usando a função next()

# Mostrar imagens
imshow(torchvision.utils.make_grid(images.cpu()))  # Mover para a CPU para visualização
# Imprimir rótulos verdadeiros
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Carregar modelo salvo e fazer previsões
net = Net().to(device)
net.load_state_dict(torch.load(PATH))

images, labels = images.to(device), labels.to(device)  # Move para a GPU para fazer previsões
outputs = net(images)
_, predicted = outputs.max(1)

# Imprimir previsões
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
