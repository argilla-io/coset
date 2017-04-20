from torchtext import data
import spacy
import visdom
import numpy as np
import torch
from torch import nn

from sklearn.metrics import f1_score

from models import BaseClassifier

vis = visdom.Visdom()
nlp = spacy.load('es')

def tokenize_with_filter(filter):
    def tokenize(text):
        return [t.lower_ for t in nlp(text) if filter(t)]
    return tokenize

def log(time, epoch, iterations, batch_idx, train_iter, loss, train_acc, dev_loss=None, dev_acc=None, lot=None):
    header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy \n'
    dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
    log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
    print(header)
    if(dev_loss):
        print(dev_log_template.format(time,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.data[0], dev_loss, train_acc, dev_acc))

        vis.line(
                X=torch.ones((1, 2)).cpu() * iterations,
                Y=torch.Tensor([loss.data[0], dev_loss]).unsqueeze(0).cpu(),
                win=lot[0],
                update='append'
            )
        vis.line(
                X=torch.ones((1, 2)).cpu() * iterations,
                Y=torch.Tensor([train_acc, dev_acc]).unsqueeze(0).cpu(),
                win=lot[1],
                update='append'
            )
    else:

        print(log_template.format(time,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.data[0], ' '*8, train_acc, ' '*12))
    print()


def train(model, batches, num_epochs=2, lot=None, optimizer=None):
    import time
    train_iter, dev_iter = batches
    # First we need to define our loss/objective function
    # Cross Entropy Loss already applies softmax
    criterion = nn.CrossEntropyLoss()
    # And the optimizer (Gradient-descent methods)

    from torch.autograd import Variable
    # Now the code for training our network
    iterations = 0
    start = time.time()
    for epoch in range(num_epochs):
        train_iter.init_epoch()
        n_correct, n_total, f1, total_loss = 0, 0, 0, 0
        for batch_idx, batch in enumerate(train_iter):
            optimizer.zero_grad()
            output = model(batch.text)
            iterations += 1
            n_correct += (torch.max(output, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            f1 += f1_score(torch.max(output, 1)[1].view(batch.label.size()).data.numpy(), batch.label.data.numpy(), average='macro')
            #print(f1)
            loss = criterion(output, batch.label)
            total_loss += loss
            loss.backward()
            optimizer.step()

        train_acc = f1/len(train_iter)

        model.eval(); dev_iter.init_epoch()
        n_dev_correct, n_dev_total, f1, dev_loss = 0, 0, 0, 0
        for dev_batch_idx, dev_batch in enumerate(dev_iter):
            answer = model(dev_batch.text)
            n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
            f1 += f1_score(torch.max(answer, 1)[1].view(dev_batch.label.size()).data.numpy(), dev_batch.label.data.numpy(), average='macro')
            n_dev_total += dev_batch.batch_size
            dev_loss += criterion(answer, dev_batch.label)

        #dev_acc = 100. * n_dev_correct / n_dev_total
        dev_acc = f1/len(dev_iter)
        log(time.time()-start,
                        epoch,
                        iterations,
                        batch_idx,
                        train_iter,
                        loss,
                        train_acc,
                        dev_loss.data[0]/len(dev_iter),
                        dev_acc,
                        lot=lot)


def trainer(batch_size, lr, min_freq=None, vocab_size=None, model=None, optimizer=None):
    print('Starting training with batch size {}'.format(batch_size))
    train_iter, dev_iter = data.BucketIterator.splits((trainset, devset),
                                                  batch_size=batch_size,
                                                  sort_key=lambda x: len(x.text),
                                                  device=-1,
                                                  shuffle=True,
                                                  repeat=False)
    lot = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='LOSS - Batch.{}.LR.{}.Vocab_size.{}.Min_freq.{}'.format(batch_size,lr, vocab_size, min_freq),
                legend=['Train Loss', 'Dev Loss']
            )
        )
    lot_acc = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='F1-Macro',
                title='F1-Macro - Batch.{}.LR.{}.Vocab_size.{}.Min_freq.{}'.format(batch_size,lr, vocab_size, min_freq),
                legend=['Train F1', 'Dev F1']
            )
        )
    # Let's call the training process
    train(model, (train_iter, dev_iter),num_epochs=200, lot=[lot, lot_acc], optimizer=optimizer)

hidden_size = 100
# for batch_size in range(8, 64, 8):
#     for lr in np.arange(0.0001, 0.0005, 0.0002):
#         trainer(batch_size, lr)

DATASET_PATH = 'data/'

# Primero definimos los campos del dataset de entrenamiento
f = (lambda t: not t.is_stop and t.is_alpha and len(t.orth_)>=2) # Examples: t.is_alpha, full documentation at: and t.is_alpha

twitter_id = data.Field()
TEXT = data.Field(tokenize=tokenize_with_filter(f))
LABEL = data.Field(sequential=False)

trainset = data.TabularDataset(path= DATASET_PATH + 'coset-train.csv',
                            format='csv',
                            fields= [('id', None), ('text', TEXT),('label',LABEL)],
                            skip_header=True)

devset = data.TabularDataset(path= DATASET_PATH + 'coset-dev.csv',
                            format='csv',
                            fields= [('id', None), ('text', TEXT),('label',LABEL)],
                            skip_header=True)

# Let's build the vocabs
LABEL.build_vocab(trainset, devset)
# Input dimensions are defined by the len of the input vocab

# Output dimensions are two:
output_dim = len(LABEL.vocab)



for min_frequency in range(1,10):
    TEXT.build_vocab(trainset, devset, min_freq=min_frequency)
    vocab_size = len(TEXT.vocab)

    for batch_size in range(16, 32, 8):
        for lr in np.arange(0.0002, 0.001, 0.0002):
            model = BaseClassifier(vocab_size, hidden_size, output_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.005)
            trainer(batch_size, lr, min_freq=min_frequency, vocab_size=vocab_size, model=model, optimizer=optimizer)
