from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from dataloader import get_data_loaders
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import wandb
import sys
import os

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''

    # Set model to train mode before each epoch
    model.train()

    # Empty list to store losses
    losses = []
    correct = 0

    # Track class-wise accuracy
    num_classes = 2
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    # Iterate over entire training samples (1 epoch)
    for batch_idx, (data, target) in enumerate(train_loader):

        # Push data/label to correct device
        data, target = data.to(device), target.to(device)

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()

        # Do forward pass for current set of data
        output = model(data)

        loss = criterion(output, target)

        # Computes gradient based on final loss
        loss.backward()

        # Store loss
        losses.append(loss.item())

        # Optimize model parameters based on learning rate and gradient
        optimizer.step()

        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()

        # Class-wise tracking
        for i in range(len(target)):
            label = target[i].item()
            class_correct[label] += (pred[i].item() == label)
            class_total[label] += 1

    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx+1) * batch_size)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(float(np.mean(losses)), correct, (batch_idx+1) * batch_size, 100. * train_acc))

    # Print class-wise accuracy
    print("Printing class wise accuracy for Training Set.")
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'  Class {i} Accuracy: {100. * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})')

    return train_loss, train_acc, class_correct, class_total


def validate(model, device, val_loader, criterion):
    '''
    Validates the model on the validation set.
    '''
    model.eval()
    losses = []
    correct = 0

    # Track class-wise accuracy
    num_classes = 2
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            losses.append(loss.item())

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Class-wise tracking
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += (pred[i].item() == label)
                class_total[label] += 1

    val_loss = float(np.mean(losses))
    val_accuracy = correct / len(val_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), 100. * val_accuracy))

    # Print class-wise accuracy
    print("Printing class wise accuracy for Validation Set.")
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'  Class {i} Accuracy: {100. * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})')

    return val_loss, val_accuracy, class_correct, class_total


# def test(model, device, test_loader, optimizer, criterion, epoch, batch_size):
def test(model, device, test_loader, criterion):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''

    # Set model to eval mode to notify all layers.
    model.eval()

    losses = []
    correct = 0

    # Track class-wise accuracy
    num_classes = 2
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # data, target = sample
            data, target = data.to(device), target.to(device)

            # Predict for data by doing forward pass
            output = model(data)

            loss = criterion(output, target)

            # Append loss to overall test loss
            losses.append(loss.item())

            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

            # Class-wise tracking
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += (pred[i].item() == label)
                class_total[label] += 1

    test_loss = float(np.mean(losses))
    test_accuracy = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * test_accuracy))
    
    # Print class-wise accuracy
    print("Printing class wise accuracy for Test Set.")
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'  Class {i} Accuracy: {100. * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})')


    return test_loss, test_accuracy, class_correct, class_total


def run_main(FLAGS):
    sys.stdout = Logger(FLAGS.log_dir)

    wandb.login(key="a240530f017b49dac5c562009a9979e8f813634b")
    run = wandb.init(
        # Set the project where this run will be logged
        project=f"MedImgComp_PA1_{FLAGS.mode}",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": FLAGS.learning_rate,
            "epochs": FLAGS.num_epochs,
            "mode": FLAGS.mode
        })
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set proper device based on cuda availability
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    print(
        f"Mode: {FLAGS.mode}, Learning Rate: {FLAGS.learning_rate}, Epochs: {FLAGS.num_epochs}")
    
    if FLAGS.mode == 'scratch':
        from models import ResNet18_Scratch
        model = ResNet18_Scratch(FLAGS.dropout).to(device) 
    elif FLAGS.mode == 'pretrained':
        from models import ResNet18
        model = ResNet18(FLAGS.dropout).to(device)
    else:
        raise ("Invalid mode selected")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=FLAGS.learning_rate)

    data_dir = '/home/ashmal/Courses/MedImgComputing/Assignment_1/data/2/chest_xray/chest_xray'
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, FLAGS.batch_size)

    best_accuracy = 0.0
    best_epoch = 0.0
    loss_train, loss_val, loss_test = [], [], []
    accuracy_train, accuracy_val, accuracy_test = [], [], []

    # Run training for n_epochs specified in config
    for epoch in range(1, FLAGS.num_epochs + 1):
        print(f"================= Running Epoch : {epoch} =================")
        train_loss, train_accuracy, train_class_correct, train_class_total = train(model, device, train_loader,
                                           optimizer, criterion, epoch, FLAGS.batch_size)
        
        val_loss, val_acc, val_class_correct, val_class_total = validate(model, device, val_loader, criterion)  # Validation after each epoch
        
        test_loss, test_accuracy, test_class_correct, test_class_total = test(model, device, test_loader, criterion)

        num_classes = 2
        class_acc = {f"train_class_{i}_accuracy": (train_class_correct[i] / train_class_total[i]) * 100 if train_class_total[i] > 0 else 0 for i in range(num_classes)}
        class_acc.update({f"val_class_{i}_accuracy": (val_class_correct[i] / val_class_total[i]) * 100 if val_class_total[i] > 0 else 0 for i in range(num_classes)})
        class_acc.update({f"test_class_{i}_accuracy": (test_class_correct[i] / test_class_total[i]) * 100 if test_class_total[i] > 0 else 0 for i in range(num_classes)})

        wandb.log({
            "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss,
            "train_accuracy": train_accuracy, "val_accuracy": val_acc, "test_accuracy": test_accuracy, **class_acc
        })


        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch

        loss_train.append(train_loss)
        accuracy_train.append(train_accuracy)
        loss_test.append(test_loss)
        accuracy_test.append(test_accuracy)
        loss_val.append(val_loss)
        accuracy_val.append(val_acc)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss,
                  "train_accuracy": train_accuracy, "val_accuracy": val_acc ,"test_accuracy": test_accuracy})

    loss_train = np.array(loss_train)
    accuracy_train = np.array(accuracy_train)
    loss_test = np.array(loss_test)
    accuracy_test = np.array(accuracy_test)
    loss_val = np.array(loss_val)
    accuracy_val = np.array(accuracy_val)


    fig = plt.figure(1)
    epochs = range(1, FLAGS.num_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_test, 'b', label='Test loss')
    plt.plot(epochs, loss_val, 'r', label='Validation loss')
    plt.title('Model mode: '+str(FLAGS.mode)+' Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    fig.savefig('plots/loss_plot mode: '+str(FLAGS.mode)+'.png')

    fig = plt.figure(2)
    epochs = range(1, FLAGS.num_epochs + 1)
    plt.plot(epochs, accuracy_train*100, 'g', label='Training accuracy')
    plt.plot(epochs, accuracy_test*100, 'b', label='Test accuracy')
    plt.plot(epochs, accuracy_val*100, 'r', label='Validation accuracy')
    plt.title('Model mode:'+str(FLAGS.mode) +
              ' Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy in %')
    plt.legend()
    plt.show()
    fig.savefig('plots/accuracy_plot mode: '+str(FLAGS.mode)+'.png')

    print("accuracy is {:2.2f}".format(best_accuracy))
    print("convergence epoch is {}".format(best_epoch))

    
    # Print final results
    print("\nFinal Results:")
    print("-"*50)
    print(f"Best Test Accuracy: {best_epoch:.2f}%")
    print(f"Final Train Loss: {loss_train[-1]:.4f}")
    print(f"Final Test Loss: {loss_test[-1]:.4f}")
    
    print("\n" + "="*50)
    print(f"Finished Mode {FLAGS.mode} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    print("Training and evaluation finished")


    save_folder = "model_weights"
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, FLAGS.model_save_path)
    torch.save(model.state_dict(), save_path)
    print(f"Model weights have been saved at: {save_path}")


if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('ResNet used to detect pneumonia in chest x-rays.')
    parser.add_argument('--mode',
                        type=str, default='pretrained',
                        help="Select between 'scratch' and 'pretrained'.")
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=40,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--dropout', 
                        type = float, 
                        default = 0.3, 
                        help= 'Dropout rate.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument("--model_save_path", 
                        type=str,
                        default='/home/ashmal/Courses/MedImgComputing/Assignment_1/A1/model_weights',
                        help="Enter the path for saving weights")

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    run_main(FLAGS)
