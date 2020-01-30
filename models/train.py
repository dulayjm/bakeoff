import time
import logging
import torch
from os import system
import pandas as pd

def train_model(dataloaders, model, loss_fn, acc_fn, optimizer, scheduler, num_epochs=10, acc_top=5):
    since = time.time()

    # send to the gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model_wts = model.state_dict()

    results = []

    for epoch in range(num_epochs):
        logging.info('Beginning Epoch {}/{}'.format(epoch+1, num_epochs))
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        running_loss = 0.0
        running_acc = 0.0

        for phase in dataloaders:
            logging.info("Entering {} phase...".format(phase))

            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
 
            running_batch = 0
            
            batch_num = 1
            batched_data = dataloaders[phase].makeBatches(dataloaders[phase].batch_size)
            for data in batched_data:
                print("{} batch {} of {}".format(phase, batch_num, len(batched_data)))

                # zero the parameter gradients
                optimizer.zero_grad()
                
                images, labels = data
                outputs = model(torch.stack(images).to(device))
                labels = torch.IntTensor(labels)
                # backward + optimize only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        loss, _, _, _ = loss_fn(outputs, labels)
                        logging.debug("{} batch {} loss: {}".format(phase, batch_num, loss))

                        loss.backward()
                        optimizer.step()

                        # track epoch total loss
                        running_loss += loss.data.item() * min(dataloaders[phase].batch_size, len(images))
                    elif phase == 'valid':
                        acc = acc_fn.get_acc(outputs, labels)
                        logging.debug("{} batch {} top 1 acc: {}".format(phase, batch_num, acc))
                        running_acc += acc * min(dataloaders[phase].batch_size, len(images))
                batch_num += 1

            scheduler.step()

        epoch_loss = running_loss / len(dataloaders["train"].dataset)
        epoch_acc = running_acc / len(dataloaders["valid"].dataset)

        logging.info("Train loss: {}".format(epoch_loss))
        logging.info("Valid acc: {}".format(epoch_acc))

        results.append(['{}/{}'.format(epoch+1, num_epochs), epoch_loss, epoch_acc])

    # output metrics to CSV
    df = pd.DataFrame(results, columns=['epoch', 'train loss', 'valid accuracy'])
    df.to_csv('{}.csv'.format(model.name))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed / 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model