#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets

from transformers import GPT2Tokenizer, GPT2LMHeadModel

torch.manual_seed(0)
np.random.seed(0)

lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
sys.path.insert(1, lab_root)


class ClassificationHead(torch.nn.Module):
    """ Classification Head for the transformer """

    def __init__(self, class_size=5, embed_size=2048):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


class Discriminator(torch.nn.Module):
    def __init__(self, class_size=5, embed_size=1024,
                 pretrained_model="gpt2-medium"):
        super(Discriminator, self).__init__()
        self.embed_size = embed_size
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.classifier_head = ClassificationHead(
            class_size=class_size,
            embed_size=embed_size
        )

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        pass
        self.classifier_head.train()

    def forward(self, x):
        mask_src = 1 - x.eq(0).unsqueeze(1).type(
            torch.FloatTensor).cuda().detach()
        mask_src = mask_src.repeat(1, self.embed_size, 1)

        hidden, _ = self.encoder.transformer(x)

        hidden = hidden.permute(0, 2, 1)
        _, _, batch_length = hidden.shape
        hidden = hidden * mask_src  # / torch.sum(mask_src, dim=-1).unsqueeze(2).repeat(1, 1, batch_length)
        hidden = hidden.permute(0, 2, 1)

        avg_hidden = torch.sum(hidden, dim=1) / (
                torch.sum(mask_src, dim=-1).detach() + 1e-10)
        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)

        return probs


class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data['X'] = self.X[index]
        data['y'] = self.y[index]
        return data


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_seqs = torch.zeros(len(sequences),
                                  max(lengths)).long().cuda()  # padding index 0
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["X"]), reverse=True)  # sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # input
    x_batch, _ = merge(item_info['X'])
    y_batch = item_info['y']

    return x_batch, torch.tensor(y_batch, device='cuda', dtype=torch.long)


def train_epoch(data_loader, discriminator, device='cuda', args=None, epoch=1):
    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
    discriminator.train_custom()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = discriminator(data)
        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.item()))


def test_epoch(data_loader, discriminator, device='cuda', args=None):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = discriminator(data)
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1,
                                 keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description='Train a discriminator on top of GPT-2 representations')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='Number of training epochs')
    parser.add_argument('--save_model', action='store_true',
                        help='whether to save the model')
    parser.add_argument('--dataset', type=str, default='SST',
                        choices=('SST', 'clickbait', 'toxic'))
    args = parser.parse_args()

    batch_size = args.batch_size
    device = 'cuda'
    # load sst
    if args.dataset == 'SST':
        idx2class = ["positive", "negative", "very positive", "very negative",
                     "neutral"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(class_size=len(idx2class)).to(device)

        text = torchtext_data.Field()
        label = torchtext_data.Field(sequential=False)
        train_data, val_data, test_data = datasets.SST.splits(
            text,
            label,
            fine_grained=True,
            train_subtrees=True,
            # filter_pred=lambda ex: ex.label != 'neutral'
        )
        x = []
        y = []

        for i in range(len(train_data)):
            seq = TreebankWordDetokenizer().detokenize(
                vars(train_data[i])["text"])
            seq = discriminator.tokenizer.encode(seq)
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            x.append(seq)
            y.append(class2idx[vars(train_data[i])["label"]])

        dataset = Dataset(x, y)

        test_x = []
        test_y = []
        for i in range(len(test_data)):
            seq = TreebankWordDetokenizer().detokenize(
                vars(test_data[i])["text"])
            seq = discriminator.tokenizer.encode(seq)
            seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
            test_x.append(seq)
            test_y.append(class2idx[vars(test_data[i])["label"]])
        test_dataset = Dataset(test_x, test_y)

    elif args.dataset == 'clickbait':
        idx2class = ["non_clickbait", "clickbait"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(class_size=len(idx2class)).to(device)

        # data = pickle.load(open("/home/gilocal/lab/exp/language/datasets/clickbait/clickbait.p", "r"))
        with open("datasets/clickbait/clickbait_train_prefix.txt") as f:
            data = []
            for class2idx in f:
                try:
                    data.append(eval(class2idx))
                except:
                    continue
        x = []
        y = []
        for d in data:
            try:
                # seq = tokenizer.encode("Apple's iOS 9 'App thinning' feature will give your phone's storage a boost")
                try:
                    seq = discriminator.tokenizer.encode(d["text"])
                except:
                    continue
                seq = torch.tensor([50256] + seq, device=device,
                                   dtype=torch.long)
                x.append(seq)
                y.append(d['label'])
            except:
                pass

        dataset = Dataset(x, y)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                              [train_size,
                                                               test_size])

    elif args.dataset == 'toxic':
        idx2class = ["non_toxic", "toxic"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(class_size=len(idx2class)).to(device)

        # data = pickle.load(open("/home/gilocal/lab/exp/language/datasets/clickbait/clickbait.p", "r"))
        with open("datasets/toxic/toxic_train.txt") as f:
            data = []
            for d in f:
                data.append(eval(d))

        x = []
        y = []
        for d in data:
            try:
                # seq = tokenizer.encode("Apple's iOS 9 'App thinning' feature will give your phone's storage a boost")
                seq = discriminator.tokenizer.encode(d["text"])

                device = 'cuda'
                if (len(seq) < 100):
                    seq = torch.tensor([50256] + seq, device=device,
                                       dtype=torch.long)
                else:
                    continue
                x.append(seq)
                y.append(int(np.sum(d['label']) > 0))
            except:
                pass

        dataset = Dataset(x, y)
        print(dataset)
        print(len(dataset))
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                              [train_size,
                                                               test_size])

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    sentence = "This is incredible! I love it, this is the best chicken I have ever had."

    for epoch in range(args.epochs):
        train_epoch(discriminator=discriminator, data_loader=data_loader,
                    args=args, device=device, epoch=epoch)
        test_epoch(data_loader=test_loader, discriminator=discriminator,
                   args=args)

        predict(sentence, discriminator, idx2class, device)

        if (args.save_model):
            #torch.save(discriminator.state_dict(),
            #           "{}_discriminator_{}.pt".format(
            #               args.dataset, epoch))
            torch.save(discriminator.get_classifier().state_dict(),
                       "{}_classifier_head.pt".format(
                           args.dataset))


def predict(input_sentence, model, classes, device):
    input_sentence_tok = model.tokenizer.encode(input_sentence)
    input_sentence_tok_t = torch.tensor([input_sentence_tok], device=device,
                                        dtype=torch.long)
    logprobs = model(
        input_sentence_tok_t).data.cpu().numpy().flatten().tolist()
    print('Input sentence:', input_sentence)
    print('Predictions:', ", ".join(
        "{}: {:.4f}".format(c, math.exp(logprob)) for c, logprob in
        zip(classes, logprobs)
    ), "\n")


if __name__ == '__main__':
    main()
