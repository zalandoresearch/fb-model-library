import argparse
import classify
import json
import matplotlib.pyplot
import numpy
import os
import pandas
import PIL
import re
import torch
import torch.utils.data
import torchabc
import torchvision


class Data(torchabc.data.MultiModalData):
    def __init__(self, dir_='data/coco'):
        super().__init__(dir_=dir_, sep='\t')


class Iterator(torchabc.data.MultiModalIterator):
    def __init__(self, data, fold='train', ssd=False):

        super().__init__(
            data,
            fold=fold,
            types_={'annotation': 'word', 'file_': 'image'},
            stop=True,
            start=True,
            ssd=ssd,
        )

        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.trans_ = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    def reverse_batch(self, item):
        y, l, x = self[item]

        print(self.df.file_.iloc[item])

        str_ = ' '.join(list(map(
            self.lookup.__getitem__,
            y.tolist()[:l.item() + 1],
        )))

        print(str_)


class Annotator(torch.nn.Module):
    def __init__(self, n_vocab=1249):
        torch.nn.Module.__init__(self)

        self.resnet = classify.resnet()
        self.rnn = torch.nn.GRU(64, 512, 1)
        self.embed = torch.nn.Embedding(n_vocab, 64)
        self.project = torch.nn.Linear(512, n_vocab)
        self.init = torch.nn.Linear(2048, 512)

    def forward(self, x, y, hidden=None):
        if hidden is None:
            hidden = self.init(self.resnet.features(x).detach())
            hidden = hidden[None, :, :]

        hidden, _ = self.rnn(self.embed(y.transpose(1, 0)), hidden)

        return self.project(hidden), hidden

    def feed(self, y, l, x):

        return self(x, y)

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.project.weight)
        torch.nn.init.xavier_normal_(self.embed.weight)
        torch.nn.init.xavier_normal_(self.init.weight)


    def annotate(self, x, start, finish, sample=True):
        last = torch.LongTensor([start])
        y = [start]
        hidden = None
        while True:
            out, hidden = self(x[None, :, :, :], last[None, :], hidden)

            if sample:
                p = out.squeeze().exp() / out.squeeze().exp().sum()
                choice = numpy.random.choice(len(p), p=p.detach().numpy())
                last = torch.LongTensor([choice])
                y.append(choice)
            else:
                choice = out.topk(1)[1].squeeze().item()
                y.append(choice)
                last = torch.LongTensor([choice])

            if y[-1] == finish:
                break
        return y


class Trainer(torchabc.train.Trainer):
    def __init__(self,
                 save_,
                 dataset='coco',
                 n_epochs=2,
                 lr=0.01,
                 batch_size=2,
                 num_workers=0,
                 ssd=False,
                 subsample=None):

        data = Data('data/' + dataset)
        model = Annotator(len(data.dictionary))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        argz = {
            'weights_interval': 25,
            'dump_interval': 100,
            'callback_interval': 1,
            'n_epochs': n_epochs,
            'annealing': 4,
            'batch_size': batch_size,
            'subsample': subsample,
            'num_workers': num_workers,
            'ssd': ssd,
        }

        super().__init__(
            save_,
            model,
            data,
            Iterator,
            optimizer,
            **argz,
        )

    def get_loss(self, y, l, x):

        yhat = self.model(x, y)[0].transpose(1, 0)
        loss = 0

        for i in range(y.shape[0]):
            loss += torch.nn.functional.cross_entropy(
                yhat[i, :l[i, 0] - 1, :],
                y[i, 1:l[i, 0]],
            ) / y.shape[0]

        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default='train_annotation',
        help='script call',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/annotator',
        help='model checkpoint',
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=200,
        help='number of epochs trained'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate of SGD'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=25,
        help='batch-size for training',
    )
    parser.add_argument(
        '--catalog',
        type=str,
        default='coco',
        help='type of model attributes to train on',
    )
    parser.add_argument(
        '--ssd',
        action='store_true',
        help='use fusion drive',
    )

    args = parser.parse_args()

    if args.mode == 'unit':

        cf = torchabc.config.Config(
            Data,
            Iterator,
            Annotator,
            Trainer,
        )

        params = torchabc.config.Params(
            {'dir_': 'data/coco'},
            {'n_vocab': 11274},
            {'dataset': 'coco', 'ssd': True},
        )

        torchabc.unit.Tester(
            cf,
            params=params,
            subsample=200,
            batch_size=2,
        ).test()

    elif args.mode == 'train':

        Trainer(
            args.checkpoint,
            args.catalog,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            ssd=args.ssd,
        ).train()

    elif args.mode == 'plot':

        torchabc.plot.Plotter(
            Trainer,
            args.checkpoint,
            regex='(?!resnet)^.*'
        ).plot()

        os.system('open {}/*.png'.format(args.checkpoint))

    elif args.mode == 'annotate':

        iterator = Iterator(Data('data/' + args.catalog), 'valid')
        model = Annotator(len(iterator.dictionary))
        model.load_state_dict(torch.load(
            args.checkpoint + '/model.pt',
            map_location=lambda storage, loc: storage,
        ))

        for i in range(10):

            choice = numpy.random.choice(len(iterator))

            x = iterator[choice][2]

            print(iterator.df.iloc[choice].file_)

            y = model.annotate(
                x, iterator.dictionary['<s>'], iterator.dictionary['</s>'],
                sample=False,
            )

            print(' '.join(list(map(iterator.lookup.__getitem__, y))))
            print('\n\n')
