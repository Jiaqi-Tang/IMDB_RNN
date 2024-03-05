import torch
import torch.nn.functional as F
import torchtext
import time
import random
from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer, ngrams_iterator

# import pandas as pd

# torch.backends.cudnn.deterministic = True

# General Setting
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 20000
LEARNING_RATE = 0.005
BATCH_SIZE = 100
NUM_EPOCHS = 15
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 2


### Defining the feature processing
def get_data(split = "train"):
    url_dp = IterableWrapper([f"{split}.csv"])
    data_dp = FileOpener(url_dp, mode="b")
    return data_dp.parse_csv().map(fn=lambda t: (t[0], t[1] ))


train_dataset = torchtext.datasets.IMDB(root='./data', split='train')
test_dataset = torchtext.datasets.IMDB(root='./data', split='test')


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# count = 0
# for a, b in train_loader:
#     print(a)
#     print(type(b))
#     count = count + 1
#
# print(count)
# print(type(train_loader))
# exit()

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# train_iter = torchtext.datasets.IMDB(split='train')
#
# for (idx, batch) in train_iter:
#     print(idx)
#     print(batch)

tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter, ngrams):
    for _, text in data_iter:
        yield ngrams_iterator(tokenizer(text), ngrams)


ngrams = 3
train_iter = torchtext.datasets.IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter, ngrams), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


def text_pipeline(x):
    return vocab(list(ngrams_iterator(tokenizer(x), ngrams)))


MAX_LEN = 200


def collate_batch(batch):
    for label, text in batch:

        # print(type(text))
        text_list = []

        count = 0
        for t in text:
            processed_text = torch.tensor(text_pipeline(t), dtype=torch.int64)
            # print(type(processed_text))
            # processed_text = torch.tensor(processed_text)
            text_list.append(processed_text)
            count = count + 1

    padded_tensor_list = [torch.nn.functional.pad(tensor[:MAX_LEN], (0, MAX_LEN - len(tensor))) for tensor in text_list]

#-----------------------------------------------------------

class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        # self.rnn = torch.nn.RNN(embedding_dim,
        #                        hidden_dim,
        #                        nonlinearity='relu')
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, input_text):
        # text dim: [sentence length, batch size]

        # print("Forward")
        # print(input_text.size())

        embedded = self.embedding(input_text)
        # embedded dim: [sentence length, batch size, embedding dim]

        # print(embedded.size())
        output, (hidden, cell) = self.rnn(embedded)
        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        # print(output.size())
        # print(hidden.size())
        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]

        # print(hidden.size())

        output = self.fc(hidden)
        # print(output.size())
        return output


# torch.manual_seed(RANDOM_SEED)
model = RNN(input_dim=len(vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES # could use 1 for binary classification
)

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Train ----------------------------------------------------------

def compute_accuracy(model, data_loader, device):

    with torch.no_grad():

        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            targets_list = []

            print(features)
            # print(targets)

            for t in targets:
                processed_targets = torch.tensor(text_pipeline(t), dtype=torch.int64)
                targets_list.append(processed_targets)

            padded_list = [torch.nn.functional.pad(tensor[:MAX_LEN], (0, MAX_LEN - len(tensor[:MAX_LEN]))) for
                                  tensor in targets_list]

            targets = torch.stack(padded_list)

            features = features.to(device)
            targets = targets.to(device)

            predicted_labels = model(targets.t())


            # print(predicted_labels)

            correct_pred += (predicted_labels.argmax(1) == features).sum().item()
            num_examples += targets.size(0)

            # print("Current acc " + str(correct_pred/num_examples))
    return correct_pred/num_examples * 100


start_time = time.time()


def text_pipeline(x):
    return vocab(list(ngrams_iterator(tokenizer(x), ngrams)))


for epoch in range(NUM_EPOCHS):
    model.train()
    print("Train")

    for i, (label, text) in enumerate(train_loader):
        print(label)

        text_list = []

        for t in text:
            processed_text = torch.tensor(text_pipeline(t), dtype=torch.int64)
            # print(type(processed_text))
            # processed_text = torch.tensor(processed_text)
            text_list.append(processed_text)

        padded_tensor_list = [torch.nn.functional.pad(tensor[:MAX_LEN], (0, MAX_LEN - len(tensor[:MAX_LEN]))) for tensor in text_list]
        # print(type(text_list))
        # print(len(text_list))
        # print(len(label))
        # print(len(padded_tensor_list))

        text = torch.stack(padded_tensor_list)
        # print(type(text))
        # print(len(text))

        text = text.to(DEVICE)
        labels = label.to(DEVICE)

        # text = text.split()
        # text = batch.

        ### FORWARD AND BACK PROP
        # print(len(text))
        logits = model(text.t())
        # print(logits)
        # print(len(logits))
        # print(type(logits))

        loss = torch.nn.functional.cross_entropy(logits, label)
        optimizer.zero_grad()


        loss.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        print("Batch " + str(i) + " of Epoch " + str(epoch))

        # if not batch_idx % 50:
        #     print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
        #           f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
        #           f'Loss: {loss:.4f}')

    with torch.set_grad_enabled(False):
        print("Test")
        print(f'training accuracy: ')
              # f'\nvalid accuracy: '
              # f'{compute_accuracy(model, valid_loader, DEVICE):.2f}%')
    print("Accuracy " + str(compute_accuracy(model, test_loader, DEVICE)))
    print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')
    # exit()

    time.sleep(60)


print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')





