import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
# количество входных параметров
input_size = 784
# количество выходных
num_classes = 10
# количество итераций
num_epochs = 5
batch_size = 100
# шаг обучения
learning_rate = 0.001

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root ='./dataset',
							train = True,
							transform = transforms.ToTensor(),
							download = True)

test_dataset = dsets.MNIST(root ='./dataset',
						train = False,
						transform = transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
										batch_size = batch_size,
										shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
										batch_size = batch_size,
										shuffle = False)

# Описываем модель, используемую в нейронках
class LogisticRegression(nn.Module):

	def __init__(self, input_size, num_classes):
        # стандарт кода для расширяемости
		super(LogisticRegression, self).__init__()

		# то есть в этом месте, где инициализация мы описываем
		# вообще все слои нейронки

        # создаем один скрытый слой. То есть linear - некая функция
        # которой скармливаем x, а на выходе получаем y = Ax+b
		self.linear = nn.Linear(input_size, num_classes)

    # описание того, как будет идти прямой проход
	def forward(self, x):

        # просто подсчет значений при заданном x
		out = self.linear(x)
		return out

# объявляем модель
model = LogisticRegression(input_size, num_classes)

# объявляем функцию потерь, то есть "как будем считать ошибку?"
criterion = nn.CrossEntropyLoss()

# объявляем функцию оптимизатора, то есть "как будем обучать?"
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# обучение модели
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# Variable - возвращает Тензоры соответствующие
		images = Variable(images.view(-1, 28 * 28))
		labels = Variable(labels)

        # обнуляем все градиенты
		optimizer.zero_grad()
        # делаем прямой проход нейронки
		outputs = model(images)
        # считаем ошибку
		loss = criterion(outputs, labels)
        # обратное распространение ошибки
		loss.backward()
        # обновляем веса
		optimizer.step()

        # выводим информацию об процессе обучения
		if (i + 1) % 100 == 0:
			print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
				% (epoch + 1, num_epochs, i + 1,
					len(train_dataset) // batch_size, loss.data))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
	images = Variable(images.view(-1, 28 * 28))
	outputs = model(images)
	_, predicted = torch.max(outputs.data, 1)
	total += labels.size(0)
	correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: % d %%' % (
			100 * correct / total))
