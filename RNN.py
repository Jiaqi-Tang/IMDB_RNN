import torch
import torchtext
import time
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer, ngrams_iterator


def yield_tokens(data_iter, input_ngrams):
    for _, text in data_iter:
        yield ngrams_iterator(tokenizer(text), input_ngrams)


def text_pipeline(x):
    return vocab(list(ngrams_iterator(tokenizer(x), ngrams)))


# RNN model
class RNN(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, input_text):
        # text dim: [sentence length, batch size]

        embedded = self.embedding(input_text)
        # embedded dim: [sentence length, batch size, embedding dim]

        output, (hidden, cell) = self.rnn(embedded)
        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]

        output = self.fc(hidden)
        return output


# Helper function
# Tokenizes text, pads and truncates text to equal length, then formats in tensor
def format_text(text):
    text_list = []
    for t in text:
        processed_text = torch.tensor(text_pipeline(t), dtype=torch.int64)
        text_list.append(processed_text)

    padded_text_list = [torch.nn.functional.pad(tensor[:MAX_LEN], (0, MAX_LEN - len(tensor[:MAX_LEN])))
                        for tensor in text_list]
    output = torch.stack(padded_text_list)
    return output.t()


# Helper function
# Formats label to meet the output classes
def format_label(label):
    output = [i - 1 for i in label]
    output = torch.tensor(output)
    return output


# Testing the model
# Runs through the test dataset and computes model accuracy
def compute_accuracy(model, data_loader, device):

    with torch.no_grad():

        correct_pred, num_examples = 0, 0

        for i, (label, text) in enumerate(data_loader):

            text = format_text(text)
            label = format_label(label)

            label = label.to(device)
            text = text.to(device)

            predicted_labels = model(text)

            correct_pred += (predicted_labels.argmax(1) == label).sum().item()
            num_examples += text.size(0)

    return correct_pred/num_examples * 100


# Training
def train(model, data_loader, test_dataloader, device):

    for epoch in range(NUM_EPOCHS):

        model.train()

        for i, (label, text) in enumerate(data_loader):

            text = format_text(text)
            label = format_label(label)

            text = text.to(device)
            label = label.to(device)

            # FORWARD AND BACK PROP
            logits = model(text)
            loss = torch.nn.functional.cross_entropy(logits, label)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            print("Batch " + str(i + 1) + " of Epoch " + str(epoch + 1))

            if i == 4:
                break

        model.eval()

        # Testing
        print(f'Accuracy : {compute_accuracy(model, test_dataloader, DEVICE):.2f}')
        print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')

        # time.sleep(60)


# General Setting
VOCABULARY_SIZE = 20000
LEARNING_RATE = 0.005
BATCH_SIZE = 100
NUM_EPOCHS = 100

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = "model.pth"

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 2
MAX_LEN = 200

# Datasets and Loaders
print("Loading data")
train_dataset = torchtext.datasets.IMDB(root='./data', split='train')
test_dataset = torchtext.datasets.IMDB(root='./data', split='test')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Tokenization
print("Building Vocab")
tokenizer = get_tokenizer("basic_english")
ngrams = 3
vocab = build_vocab_from_iterator(yield_tokens(train_dataset, ngrams), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Model and Optimizer
print("Building Model")
RNN_model = RNN(input_dim=len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, output_dim=NUM_CLASSES)
RNN_model = RNN_model.to(DEVICE)
optimizer = torch.optim.Adam(RNN_model.parameters(), lr=0.005)

# Train
print("Start of Training")
start_time = time.time()
# train(RNN_model, train_loader, test_loader, DEVICE)
print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')

# Save model
print("Saving Model")
torch.save(RNN_model.state_dict(), PATH)
