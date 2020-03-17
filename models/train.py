import time
import logging
import torch
from os import system
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualizer import visualize

def train_model(dataloaders, model, criterion, hook, acc_fn, optimizer, scheduler, num_epochs=10, name="model"):
    since = time.time()

    # send to the gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_acc = 0.0
    best_model_wts = model.state_dict()

    results = []

    for epoch in range(num_epochs):
        logging.info('Beginning Epoch {}/{}'.format(epoch+1, num_epochs))
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        # make new batches to shuffle the data
        batched_data = {
            'train': dataloaders['train'].makeBatches(dataloaders['train'].batch_size),
            'valid': dataloaders['valid'].makeBatches(dataloaders['valid'].batch_size)
        }

        for phase in dataloaders:
            logging.debug("Entering {} phase...".format(phase))

            running_loss = 0.0
            running_acc = 0.0
            image_count = 0

            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            
            num_batches = 0
            for data in batched_data[phase]:
                num_batches += 1
                print("{} batch {} of {}".format(phase, num_batches, len(batched_data[phase])))

                # zero the parameter gradients
                optimizer.zero_grad()

                images, labels, fileNames = data
            
                outputs = model(torch.stack(images).to(device))
                labels = torch.IntTensor(labels)

                loss = criterion(outputs, labels)
                logging.debug("{} batch {} loss: {}".format(phase, num_batches, loss))

                # backward + optimize only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # track epoch total loss
                    running_loss += loss.data.item()

                    acc, img_pairs = acc_fn.get_acc(outputs, labels)
                    # iterate through each image and its most similar image in batch
                    for pair_id, [idx1, idx2, correct] in enumerate(img_pairs):
                        # extract features from last convolutional layer
                        features = [hook.features[idx1], hook.features[idx2]]
                        # save fileNames for opening image
                        files = [fileNames[idx1], fileNames[idx2]]
                        # convert int correct value to word
                        correct = 'correct' if correct else 'incorrect'
                        # construct visualization file name based on pair id and labels
                        folder = 'results/{}/maps/epoch{}/{}/batch{}/'.format(name,epoch+1,phase,num_batches)
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        out_file = folder + '{}-{}_and_{}-{}.png'.format(correct,labels[idx1],labels[idx2],pair_id)
                        # create activation map with original image shape
                        visualize(features, files, out_file, images[0].shape[1:3])

                    logging.debug("{} batch {} top 1 acc: {}".format(phase, num_batches, acc))
                    running_acc += acc

            avg_loss = running_loss / num_batches
            avg_acc = running_acc / num_batches
            logging.info("{} loss: {}".format(phase, avg_loss))
            logging.info("{} acc: {}".format(phase, avg_acc))
            best_acc = avg_acc if phase is "valid" and avg_acc > best_acc else best_acc

            results.append([epoch+1, phase, avg_loss, avg_acc])

        scheduler.step()

    # output metrics to CSV
    results = pd.DataFrame(results, columns=['epoch', 'phase', 'loss', 'accuracy'])
    results.to_csv('results/{}/results.csv'.format(name), index = False)
    
    # seperate into training and validation dataframes
    train = []
    valid = []
    for index, row in results.iterrows():
        if row['phase'] == 'train':
            train.append([row['epoch'], row['loss'], row['accuracy']])
        elif row['phase'] == 'valid':
            valid.append([row['epoch'], row['loss'], row['accuracy']])
    train = pd.DataFrame(train, columns=['epoch', 'loss', 'accuracy'])
    valid = pd.DataFrame(valid, columns=['epoch', 'loss', 'accuracy'])

    # plot the training vs validation metrics
    train_loss, = plt.plot(train['epoch'], train['loss'], label = "Training Loss")
    valid_loss, = plt.plot(valid['epoch'], valid['loss'], label = "Validation Loss")
    plt.legend(handles=[train_loss, valid_loss])
    plt.title(name)
    plt.savefig('results/{}/loss.png'.format(name))
    plt.clf()
    train_loss, = plt.plot(train['epoch'], train['accuracy'], label = "Training Accuracy")
    valid_loss, = plt.plot(valid['epoch'], valid['accuracy'], label = "Validation Accuracy")
    plt.legend(handles=[train_loss, valid_loss])
    plt.title(name)
    plt.savefig('results/{}/acc.png'.format(name))

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed / 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))
    print('Training complete! Check the log file {}.log and csv file {}.csv for results'.format(name,name))
    return model