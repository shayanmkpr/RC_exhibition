import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from DHO import ReservoirComputing
from HEB import Heb_func

def train_RC_voice(device_1, device_2, sr, n_nodes, input_size, 
                   output_size, alpha, gamma, omega, lr_Wih, lr_Who,
                   h, c_rate, lr_h, sigma, p, ps, num_epochs, clip_value,
                     data_path, model_save_folder):
    
    device = device_1
    device_RC = device_2

    #load the data
    data = torch.load(data_path)
    
    #save the model
    name = f'model_{c_rate}_{n_nodes}_{lr_h * 1000}_{sr/1000}.pt'
    model_save_path = os.path.join(model_save_folder, name)


    model = ReservoirComputing(n_nodes, input_size, output_size,
                           alpha, gamma, omega,
                           device=device_RC)
    
    criterion = nn.CrossEntropyLoss().to(device_RC)
    optimizer = optim.Adam([
    {'params': model.Wih.parameters(), 'lr': lr_Wih},
    {'params': model.Who.parameters(), 'lr': lr_Who}
    ])
    
    loss_hist = []
    clip_value = 1
    losses = 0
    hist = []
    acc = []
    f1 = []

    delta_norms = []
    mel_train = data['mel_tensors']
    mel_test = data['mel_test']
    labels_train = data['labels_tensor']
    labels_test = data['labels_test']

    for epoch in range(num_epochs):

        model.train()
        for i, (audio, label) in enumerate(zip(mel_train, labels_train)):

            audio = audio.to(device_RC)
            label = label.to(device_RC)

            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            x_t = torch.zeros(n_nodes).to(device_RC)
            y_t = torch.zeros(n_nodes).to(device_RC)

            x_h = []
            y_h = []

            c = int((audio.t()[0].size(0)) * c_rate)

            for t in range(audio.t()[0].size(0)):
                wave_value = audio[t]
                output, x_t, y_t = model(x_t, y_t, h, wave_value)

                #HEBBIAN

                if t>=c:
                    x_h.append(x_t[0])
                    y_h.append(y_t[0])
                    X_h = torch.stack(x_h)
                    Y_h = torch.stack(y_h)
                    x_heb = X_h.t()
                    y_heb = Y_h.t()

                    delta_w = Heb_func(sigma, lr_h, p, ps, x_heb,x_heb, n_nodes)

                    model.Whh.weight.data += delta_w
                    
                    delta_norm = torch.linalg.norm(delta_w.detach())
                    delta_norms.append(delta_norm.item() / torch.norm(model.Whh.weight.data))
            loss = criterion(output, label.view(1))
            losses += loss.item()
            loss.backward()
            optimizer.step()

            checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
    }
        torch.save(checkpoint, model_save_path)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {losses:.4f}')
        hist.append(losses)
        losses = 0 


        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for audio, label in zip(mel_test, labels_test):
                x_t = torch.zeros(n_nodes).to(device_RC)
                y_t = torch.zeros(n_nodes).to(device_RC)

                for t in range(audio.size(0)):
                    wave_value = audio[t]
                    output, x_t, y_t = model(x_t, y_t, h, wave_value)

                _, predicted = torch.max(output.data, 1)
                all_preds.append(predicted.item())
                all_labels.append(label.item())

        accuracy = accuracy_score(all_labels, all_preds)
        fscore = f1_score(all_labels, all_preds, average='weighted')

        acc.append(accuracy)
        f1.append(fscore)

        print(f"Accuracy on test set after epoch {epoch+1}: {accuracy:.4f}")
        print(f"F1 Score on test set after epoch {epoch+1}: {fscore:.4f}")





def train_RC_sMNIST(device_1, device_2, n_nodes, input_size, 
                    output_size, alpha, gamma, omega, lr_Wih, lr_Who,
                    h, c, lr_h, sigma, p, ps, n_epochs, clip_value,
                    data_path, model_save_folder):
    
    device = device_1
    device_RC = device_2

    #load the data
    transform = transforms.Compose([
    transforms.Resize((14, 14)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1, 1))
    ])

    full_train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    train_dataset, _ = random_split(full_train_dataset, [1000, len(full_train_dataset) - 1000])
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=40, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=True, pin_memory=True)
        
    #save the model
    name = f'model_{c}_{n_nodes}_{lr_h * 1000}_sMNIST.pt'
    model_save_path = os.path.join(model_save_folder, name)


    model = ReservoirComputing(n_nodes, input_size, output_size,
                           alpha, gamma, omega,
                           device=device_RC)
    
    criterion = nn.CrossEntropyLoss().to(device_RC)
    optimizer = optim.Adam([
    {'params': model.Wih.parameters(), 'lr': lr_Wih},
    {'params': model.Who.parameters(), 'lr': lr_Who}
    ])

    loss_hist = []
    stop_all = 0
    hist = []
    batch_losses = []
    accuracies = []
    delta_norms = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device_RC), labels.to(device_RC)
            
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            batch_loss = 0.0
            for i in range(images.size(0)):
                image = images[i]
                label = labels[i]

                x_t = torch.zeros(n_nodes).to(device_RC)
                y_t = torch.zeros(n_nodes).to(device_RC)

                x_h = []
                y_h = []


                for t in range(image.size(0)):
                    feeding_input = image[t, 0]
                    output, x_t, y_t = model(x_t, y_t, h, feeding_input)

                    # Hebbian:
                    if t >= c:
                        x_h.append(x_t[0])
                        y_h.append(y_t[0])
                        X_h = torch.stack(x_h)
                        Y_h = torch.stack(y_h)
                        x_heb = X_h.t()
                        y_heb = Y_h.t()

                        delta_w = Heb_func(sigma, lr_h, p, ps, x_heb,x_heb, n_nodes)

                        model.Whh.weight.data += delta_w
                        
                        delta_norm = torch.linalg.norm(delta_w.detach())
                        delta_norms.append(delta_norm.item() / torch.norm(model.Whh.weight.data))
                        
                outputs = output.view(1, -1)

                loss = criterion(outputs, label.view(1))
                batch_loss += loss.item()
                loss.backward()
                optimizer.step()


            # print(f"Batch Loss={batch_loss}")
            batch_losses.append(batch_loss)
            running_loss += batch_loss / images.size(0)
        
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        hist.append(running_loss/len(train_loader))

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device_RC), labels.to(device_RC)

                for i in range(images.size(0)):  
                    image = images[i]
                    label = labels[i]

                    x_t = torch.zeros(n_nodes).to(device_RC)
                    y_t = torch.zeros(n_nodes).to(device_RC)

                    for t in range(image.size(0)):  
                        feeding_input = image[t, 0]
                        output, x_t, y_t = model(x_t, y_t, h, feeding_input)

                    outputs = output.view(1, -1)
                    _, predicted = torch.max(outputs.data, 1)
                    total += 1
                    correct += (predicted == label).sum().item()

        accuracy = 100 * correct / total
        print(f'Batch [{batch_idx+1}/{len(train_loader)}], Accuracy: {accuracy:.2f}%')
        accuracies.append(accuracy)
        model.train()
        hist.append(running_loss/len(train_loader))