import time
import logging
import torch
from os import system
import pandas as pd

def train_model(dataloaders, model, loss_fn, acc_fn, optimizer, scheduler, num_epochs=10):
    since = time.time()

    # send to the gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    

    best_model_wts = model.state_dict()
    best_acc = 0.0

    results = []

    for epoch in range(num_epochs):
        logging.info('Beginning Epoch {}/{}'.format(epoch+1, num_epochs))

        for phase in dataloaders:
            logging.info("Entering {} phase...".format(phase))
            dataset_size = len(dataloaders[phase].dataset)

            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_batch = 0
            running_images = 0
            running_corrects = 0

            i = 1
            for data in dataloaders[phase].batched_data:
                logging.info("Batch {} of {}".format(i, len(dataloaders[phase].batched_data)))
                i += 1

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss = loss_fn.getLoss(data, model, device)
                    logging.debug("Batch loss: {}".format(loss))
                    loss.backward()

                    # statistics
                    running_loss += loss.data.item()
                elif phase == 'valid':
                    j = 0
                    corrects = 0
                    for test in data:
                        test_data = data[:j] + data[i+j:]
                        corrects += acc_fn.correct(test, test_data, model, device)
                        j += 1

                    running_corrects += corrects

            optimizer.step()
            running_batch += 1

            scheduler.step()

        epoch_loss = running_loss / running_batch
        epoch_acc = running_corrects / dataset_size

        logging.info("Train loss: {}".format(epoch_loss))
        logging.info("Valid acc: {}".format(epoch_acc))

        results.append(['{}/{}'.format(epoch+1, num_epochs), epoch_loss, epoch_acc])

    # output metrics to CSV
    df = pd.DataFrame(results, columns=['epoch', 'train loss', 'valid accuracy'])
    df.to_csv('results.csv')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed / 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model