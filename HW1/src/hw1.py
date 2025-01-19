import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import  SubsetRandomSampler
import time
import multiprocessing

## Fashion MNIST data loading
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,),)]) #mean and std have to be sequences (e.g., tuples),
                                                                      # therefore we should add a comma after the values

#Load the data: train and test sets
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=False, transform=transform)

#Preparing for validaion test
indices = list(range(len(trainset)))
np.random.shuffle(indices)
#to get 20% of the train set
split = int(np.floor(0.2 * len(trainset)))
train_sample = SubsetRandomSampler(indices[:split])
valid_sample = SubsetRandomSampler(indices[split:])

#Data Loader
trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=64)
validloader = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=64)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120,10)
        #defining the 20% dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x = self.dropout(F.relu(self.fc1(x)))
        #not using dropout on output layer
        x = F.tanh(self.fc2(x))

        return x

class ModelTrainer:
    def __init__(self, model, learning_rate=1, num_epochs=100):
        self.model = model
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_loss = []
        self.test_loss = []

    def train_model(self, loader, epoch):
        self.model.train()
        loss_list = []
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx*len(data), len(loader.dataset),
                        100. * batch_idx / len(loader), loss.item()))
        avg_loss = sum(loss_list)/len(loss_list)
        self.train_loss.append(avg_loss)


    def test_model(self, loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        loss_list = []
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                batch_loss = self.loss_func(output, target).item()  # Calculate the loss for this batch
                test_loss += batch_loss  # Add the batch loss to the total test loss
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                loss_list.append(batch_loss)  # Append the batch loss to the list

        avg_loss = sum(loss_list) / len(loss_list)
        self.test_loss.append(avg_loss)
        test_loss /= len(loader.dataset)
        accuracy = 100. * correct / len(loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(loader.dataset), accuracy))

    def run(self, trainloader, validloader):
        for epoch in range(self.num_epochs):
            self.train_model(trainloader, epoch)
            self.test_model(validloader)
        self.plot_loss()


    # TODO: Generate a plot of the training/testing loss vs. the number of epochs. (10pts)
    def plot_loss(self):
        ######################################
        ######### WRITE YOUR CODE HERE ########
        ######################################
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_loss) + 1)
        plt.plot(epochs, self.train_loss, 'b-', label='Training Loss')
        plt.plot(epochs, self.test_loss, 'r-', label='Validation Loss')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.show()



# ------------------------------------------------------------------------------------------------
#                Task 1: Visualize the model's correct and incorrect predictions on the testing dataset using a confusion Matrix.(10 pts)
# ------------------------------------------------------------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(model, testloader, class_names):
    model.eval()
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    with torch.no_grad():
        for data, target in testloader:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            for t, p in zip(target.cpu().numpy(), predicted.cpu().numpy()):
                cm[t, p] += 1
    
    accuracy = np.trace(cm) / np.sum(cm) * 100
    
    plt.figure(figsize=(12, 8))
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}%')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------------------------
#                Task 2: Experiment with different learning rates
# ------------------------------------------------------------------------------------------------

def train_and_evaluate_model(learning_rate, num_epochs=20):
    """
    Train and evaluate a model with given hyperparameters
    Returns training history for analysis
    """
    model = Classifier()
    trainer = ModelTrainer(model=model, learning_rate=learning_rate, num_epochs=num_epochs)
    trainer.run(trainloader, validloader)
    
    print(f"\nFinal evaluation for learning rate = {learning_rate}:")
    trainer.test_model(testloader)
    
    return trainer.train_loss, trainer.test_loss

def experiment_learning_rates():
    learning_rates = [0.01, 0.1, 0.5, 1.0, 2.0]
    num_epochs = 20 
    results = {}
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate = {lr}")
        train_loss, val_loss = train_and_evaluate_model(lr, num_epochs)
        results[lr] = {'train_loss': train_loss, 'val_loss': val_loss}
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    for lr in learning_rates:
        epochs = range(1, len(results[lr]['train_loss']) + 1)
        plt.plot(epochs, results[lr]['train_loss'], label=f'LR = {lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for lr in learning_rates:
        epochs = range(1, len(results[lr]['val_loss']) + 1)
        plt.plot(epochs, results[lr]['val_loss'], label=f'LR = {lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis of Learning Rate Effects:")
    for lr in learning_rates:
        final_train_loss = results[lr]['train_loss'][-1]
        final_val_loss = results[lr]['val_loss'][-1]
        print(f"\nLearning Rate = {lr}:")
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"Generalization Gap: {abs(final_val_loss - final_train_loss):.4f}")

# ------------------------------------------------------------------------------------------------
#                Task 3: Experiment with different activation functions
# ------------------------------------------------------------------------------------------------

def train_and_evaluate_model_with_activation(hidden_activation_name, output_activation_name, num_epochs=20, learning_rate=0.1):
    class ClassifierWithActivation(nn.Module):
        def __init__(self, hidden_activation, output_activation):
            super().__init__()
            self.fc1 = nn.Linear(784, 120)
            self.fc2 = nn.Linear(120, 10)
            self.dropout = nn.Dropout(0.2)
            self.hidden_activation = hidden_activation
            self.output_activation = output_activation

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = self.dropout(self.hidden_activation(self.fc1(x)))
            x = self.fc2(x)
            if self.output_activation:
                x = self.output_activation(x)
            return x

    def get_activation(name):
        if name == 'ReLU':
            return nn.ReLU()
        elif name == 'Tanh':
            return nn.Tanh()
        elif name == 'Sigmoid':
            return nn.Sigmoid()
        elif name == 'SoftMax':
            return nn.Softmax(dim=1)
        elif name == 'None':
            return None
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    hidden_activation = get_activation(hidden_activation_name)
    output_activation = get_activation(output_activation_name)
    
    model = ClassifierWithActivation(hidden_activation, output_activation)
    trainer = ModelTrainer(model=model, learning_rate=learning_rate, num_epochs=num_epochs)
    trainer.run(trainloader, validloader)
    
    print(f"\nFinal evaluation for hidden: {hidden_activation_name}, output: {output_activation_name}:")
    trainer.test_model(testloader)
    
    return trainer.train_loss, trainer.test_loss

def experiment_activation_functions():
    # Define combinations to test
    activation_combinations = [
        ('ReLU', 'None'),     # Original baseline
        ('ReLU', 'Tanh'),     # ReLU hidden, Tanh output
        ('Tanh', 'ReLU'),     # Tanh hidden, ReLU output
        ('Sigmoid', 'None'),   # Sigmoid hidden, no output activation
        ('ReLU', 'SoftMax'),  # ReLU hidden, SoftMax output
    ]
    
    num_epochs = 20
    results = {}
    
    for hidden_act, output_act in activation_combinations:
        combo_name = f"{hidden_act}:{output_act}"
        print(f"\nTraining with hidden activation = {hidden_act}, output activation = {output_act}")
        train_loss, val_loss = train_and_evaluate_model_with_activation(hidden_act, output_act, num_epochs)
        results[combo_name] = {'train_loss': train_loss, 'val_loss': val_loss}
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    for combo_name in results:
        epochs = range(1, len(results[combo_name]['train_loss']) + 1)
        plt.plot(epochs, results[combo_name]['train_loss'], label=f'Activation = {combo_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss for Different Activation Function Combinations')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for combo_name in results:
        epochs = range(1, len(results[combo_name]['val_loss']) + 1)
        plt.plot(epochs, results[combo_name]['val_loss'], label=f'Activation = {combo_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss for Different Activation Function Combinations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis of Activation Function Combinations:")
    for combo_name in results:
        final_train_loss = results[combo_name]['train_loss'][-1]
        final_val_loss = results[combo_name]['val_loss'][-1]
        print(f"\nActivation Combination = {combo_name}:")
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"Generalization Gap: {abs(final_val_loss - final_train_loss):.4f}")

def train_and_evaluate_model_with_neurons(hidden_neurons, num_epochs=20, learning_rate=0.1):
    class ClassifierWithNeurons(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.fc1 = nn.Linear(784, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 10)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    model = ClassifierWithNeurons(hidden_neurons)
    trainer = ModelTrainer(model=model, learning_rate=learning_rate, num_epochs=num_epochs)
    trainer.run(trainloader, validloader)
    
    print(f"\nFinal evaluation for hidden neurons = {hidden_neurons}:")
    trainer.test_model(testloader)
    
    return trainer.train_loss, trainer.test_loss

# ------------------------------------------------------------------------------------------------
#                Task 4: Experiment with different numbers of hidden neurons
# ------------------------------------------------------------------------------------------------

class ClassifierWithLayers(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(784, layer_sizes[0]))
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.layers.append(nn.Linear(layer_sizes[-1], 10))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for layer in self.layers[:-1]:
            x = self.dropout(F.relu(layer(x)))
        x = self.layers[-1](x)
        return x

def train_and_evaluate_model_with_architecture(hidden_sizes, num_epochs=20, learning_rate=0.1):
    model = ClassifierWithLayers(hidden_sizes)
    trainer = ModelTrainer(model=model, learning_rate=learning_rate, num_epochs=num_epochs)
    
    start_time = time.time()
    trainer.run(trainloader, validloader)
    end_time = time.time()
    
    print(f"\nFinal evaluation for architecture = {hidden_sizes}:")
    trainer.test_model(testloader)
    
    # Calculate accuracy on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testloader:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    
    return trainer.train_loss, trainer.test_loss, end_time - start_time, accuracy

def experiment_architectures():
    # Test different architectures
    architectures = [
        [10],           # Single layer, small
        [1000],         # Single layer, large
        [120, 84],      # Two layers, medium
        [500, 250],     # Two layers, large
        [120, 84, 60]   # Three layers, medium
    ]
    
    results = {}
    
    for arch in architectures:
        arch_name = "->".join(map(str, arch))
        print(f"\nTraining with architecture: {arch_name}")
        train_loss, val_loss, time_taken, accuracy = train_and_evaluate_model_with_architecture(arch)
        
        results[arch_name] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time': time_taken,
            'neurons': sum(arch),
            'accuracy': accuracy
        }
    
    # Create comparison table
    print("\nArchitecture Comparison:")
    headers = ["Architecture", "Total Neurons", "Training Time", "Final Loss", "Accuracy"]
    rows = []
    
    for arch_name, data in results.items():
        rows.append([
            arch_name,
            data['neurons'],
            f"{data['time']:.2f}s",
            f"{data['val_loss'][-1]:.4f}",
            f"{data['accuracy']:.2f}%"
        ])
    
    # Print the table
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    # Print headers
    header_str = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    
    # Print rows
    for row in rows:
        print(" | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row))))
    print("-" * len(header_str))

# ------------------------------------------------------------------------------------------------
#                Task 5: Experiment with different DataLoader settings
# ------------------------------------------------------------------------------------------------

def experiment_dataloader_performance():
    print("\nDataLoader Performance Comparison Experiment")
    print("=========================================")
    
    # Test with default settings
    print("\nTesting with default DataLoader settings...")
    trainloader_default = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=64)
    validloader_default = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=64)
    testloader_default = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    
    start_time = time.time()
    model = Classifier()
    trainer = ModelTrainer(model=model, learning_rate=0.5, num_epochs=20)
    trainer.run(trainloader_default, validloader_default)
    trainer.test_model(testloader_default)
    default_time = time.time() - start_time
    
    # Test with optimized settings
    print("\nTesting with optimized DataLoader settings...")
    num_cores = multiprocessing.cpu_count()
    num_workers = num_cores * 2
    print(f"Number of cores: {num_cores}")
    print(f"Number of workers: {num_workers}")
    
    trainloader_optimized = torch.utils.data.DataLoader(trainset, sampler=train_sample, 
                                                      batch_size=64, num_workers=num_workers, 
                                                      pin_memory=True)
    validloader_optimized = torch.utils.data.DataLoader(trainset, sampler=valid_sample, 
                                                       batch_size=64, num_workers=num_workers, 
                                                       pin_memory=True)
    testloader_optimized = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, 
                                                      num_workers=num_workers, pin_memory=True)
    
    start_time = time.time()
    model = Classifier()
    trainer = ModelTrainer(model=model, learning_rate=0.5, num_epochs=20)
    trainer.run(trainloader_optimized, validloader_optimized)
    trainer.test_model(testloader_optimized)
    optimized_time = time.time() - start_time
    
    # Print comparison
    print("\nPerformance Comparison:")
    print(f"Default settings: {default_time:.2f} seconds")
    print(f"Optimized settings: {optimized_time:.2f} seconds")
    print(f"Speed improvement: {(default_time - optimized_time) / default_time * 100:.1f}%")

def main():
    print("\nFashion MNIST Neural Network Experiments")
    print("=======================================")
    
    while True:
        print("\nAvailable tasks:")
        print("1. Confusion Matrix Visualization")
        print("2. Learning Rate Experiments")
        print("3. Activation Function Experiments")
        print("4. Network Architecture Experiments")
        print("5. DataLoader Performance Experiments")
        print("6. Exit")
        
        choice = input("\nSelect a task (1-6): ")
        
        if choice == '1':
            print("\nRunning Confusion Matrix Visualization...")
            model = Classifier()
            trainer = ModelTrainer(model=model, learning_rate=0.5, num_epochs=20)
            trainer.run(trainloader, validloader)
            plot_confusion_matrix(model, testloader, trainset.classes)
            
        elif choice == '2':
            print("\nRunning Learning Rate Experiments...")
            experiment_learning_rates()
            
        elif choice == '3':
            print("\nRunning Activation Function Experiments...")
            experiment_activation_functions()
            
        elif choice == '4':
            print("\nRunning Network Architecture Experiments...")
            experiment_architectures()
            
        elif choice == '5':
            print("\nRunning DataLoader Performance Experiments...")
            experiment_dataloader_performance()
            
        elif choice == '6':
            print("\nExiting program...")
            break
            
        else:
            print("\nInvalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()