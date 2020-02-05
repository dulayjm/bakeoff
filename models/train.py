import time
import logging
import torch
from os import system
import pandas as pd

def train_model(dataloaders, model, criterion, acc_fn, optimizer, scheduler, num_epochs=10, name="model"):
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

        if (epoch % 1 == 0):
            batched_data = {
                'train': dataloaders['train'].makeBatches(dataloaders['train'].batch_size),
                'valid': dataloaders['valid'].makeBatches(dataloaders['valid'].batch_size)
            }

        running_loss = 0.0
        running_acc = 0.0
        image_count = 0

        for phase in dataloaders:
            logging.info("Entering {} phase...".format(phase))

            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
 
            running_batch = 0
            
            batch_num = 1
            for data in batched_data[phase]:
                print("{} batch {} of {}".format(phase, batch_num, len(batched_data[phase])))

                # zero the parameter gradients
                optimizer.zero_grad()

                images, labels = data
                outputs = model(torch.stack(images).to(device))
                labels = torch.IntTensor(labels)

                # backward + optimize only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        loss, _, _, _ = criterion(outputs, labels)
                        logging.debug("{} batch {} loss: {}".format(phase, batch_num, loss))
                        loss.backward()
                        optimizer.step()

                        # track epoch total loss
                        running_loss += loss.data.item() * len(images)
                        image_count += len(images)
                    elif phase == 'valid':
                        acc = acc_fn.get_acc(outputs, labels)
                        logging.debug("{} batch {} top 1 acc: {}".format(phase, batch_num, acc))
                        running_acc += acc * len(images)
                batch_num += 1

        scheduler.step()

        epoch_loss = running_loss / image_count
        epoch_acc = running_acc / len(dataloaders["valid"].dataset)
        best_acc = epoch_acc if epoch_acc > best_acc else best_acc

        logging.info("Train loss: {}".format(epoch_loss))
        logging.info("Valid acc: {}".format(epoch_acc))

        results.append(['{}/{}'.format(epoch+1, num_epochs), epoch_loss, epoch_acc])

    # output metrics to CSV
    df = pd.DataFrame(results, columns=['epoch', 'train loss', 'valid accuracy'])
    df.to_csv('{}.csv'.format(name))

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed / 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))
    print('Training complete! Check the log file {}.log and csv file {}.csv for results'.format(name,name))

    return model