import argparse
import json
import math
import matplotlib.pyplot
import numpy
import os
import pandas
import re
import skimage.io
import skimage.transform
import torch
import torch.utils.data
import torch.utils.model_zoo
import tqdm


blurb = 'Fashion Brain: Library of trained deep learning models (D2.5)'



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = \
            torch.nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(planes * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AvgPool2d(7, stride=1)
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(torch.utils.model_zoo.load_url(
        'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    ))
    return model


class Classify(torch.nn.Module):
    def __init__(self, n_classes, finetune=False):
        torch.nn.Module.__init__(self)

        self.cnn = resnet()
        self.linear = torch.nn.Linear(2048, n_classes)
        self.finetune = finetune

    def forward(self, x):
        if self.finetune:
            return self.linear(self.cnn.features(x))
        else:
            return self.linear(self.cnn.features(x).detach())


class Data:
    def __init__(self, model_type):
        self.items = pandas.read_csv('data/{}.csv'.format(model_type))

        with open('data/{}.json'.format(model_type)) as f:
            self.dictionary = json.load(f)

        self.lookup = dict(zip(
            self.dictionary.values(), self.dictionary.keys()
        ))

        ix = numpy.random.permutation(len(self.items))

        self.items['fold'] = ['train' for _ in range(len(self.items))]
        self.items['fold'].iloc[ix[-2000:-1000]] = 'valid'
        self.items['fold'].iloc[ix[-1000:]] = 'test'


class Iterator:
    def __init__(self, data, fold, model_type, local=False):

        if fold == 'train':
            self.items = stratify(
                data.items[data.items.fold == fold],
                model_type,
                1000,
            )
        else:
            self.items = stratify(
                data.items[data.items.fold == fold],
                model_type,
                25,
            )

        self.dictionary = data.dictionary
        self.lookup = data.lookup
        self.model_type = model_type
        self.local = local

    def __getitem__(self, item):
        if self.local:
            x = 'data/img/' + \
                str(self.items['ix'].iloc[item]).zfill(6) + '.jpg'
        else:
            x = '/data/fb-model-library/data/large/img/' + \
                str(self.items['ix'].iloc[item]).zfill(6) + '.jpg'

        x = torch.FloatTensor(skimage.io.imread(x).transpose(2, 0, 1))

        y = torch.LongTensor([
            self.dictionary[self.items.iloc[item][self.model_type]]
        ])

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        return x, y

    def __len__(self):
        return len(self.items)


class Trainer:
    def __init__(self,
                 save_='checkpoints/classify',
                 model_type='color',
                 n_epochs=200,
                 lr=0.01,
                 batch_size=50,
                 local=False):

        self.save_ = save_
        self.model_type = model_type

        data = Data(model_type)
        train_it = Iterator(
            data, fold='train', local=local, model_type=model_type,
        )
        valid_it = Iterator(
            data, fold='valid', local=local, model_type=model_type,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_it,
            batch_size=batch_size,
            shuffle=True,
        )

        self.valid_loader = torch.utils.data.DataLoader(
            valid_it,
            batch_size=batch_size,
        )

        self.model = Classify(len(train_it.dictionary))

        self.optimizer = \
            torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=lr)

        self.n_epochs = n_epochs
        self.it = 0

    def train(self):

        if torch.cuda.is_available():
            self.model.cuda()

        vl = []
        va = []
        for epoch in range(self.n_epochs):

            print('EPOCH {} '.format(epoch) + '*' * 20)
            loss, acc = self.do_epoch()
            vl.append(loss)
            va.append(acc)

            format_str = \
                'VALIDATION iteration: {}; validation-loss: {}; ' + \
                'validation-acc: {};'

            print(format_str.format(self.it, vl[-1], va[-1]))

            if va[-1] == max(va):
                print('saving...')
                torch.save(self.model, self.save_ + '/model.pt')
            else:
                print('Annealing learning rate...')
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /= 4

    def do_epoch(self):

        for x, y in self.train_loader:

            loss, acc = self.get_loss(x, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('TRAIN iteration: {}; loss: {}; accuracy: {};'.format(
                self.it, loss, acc,
            ))

            self.it += 1

        validation_loss = []
        validation_acc = []

        for x, y in self.valid_loader:

            loss, acc = self.get_loss(x, y)
            validation_loss.append(loss)
            validation_acc.append(acc)

        return sum(validation_loss) / len(validation_loss), \
            sum(validation_acc) / len(validation_acc)

    def get_loss(self, x, y):
        yhat = self.model(x)
        loss = torch.nn.functional.cross_entropy(yhat, y.squeeze())
        acc = []

        for i in range(yhat.shape[0]):
            acc.append(y[i, 0] in yhat[i, :].topk(1)[1])

        return loss, numpy.mean(acc)


def plot(checkpoint):

    with open(checkpoint + '/log') as f:
        lines = f.read().split('\n')

    v_err = []
    t_err = []
    v_acc = []
    t_acc = []
    v_it = []
    t_it = []
    for line in lines:

        matcher = 'TRAIN iteration: (\d+); ' + \
            'loss: (\d+\.\d+); accuracy: (\d+\.\d+);'
        match = re.search(matcher, line)

        if match:
            t_it.append(int(match.groups()[0]))
            t_err.append(float(match.groups()[1]))
            t_acc.append(float(match.groups()[2]))

        matcher = 'VALIDATION iteration: (\d+); ' + \
            'validation-loss: (\d+\.\d+); validation-acc: (\d+\.\d+);'
        match = re.search(matcher, line)

        if match:
            v_it.append(int(match.groups()[0]))
            v_err.append(float(match.groups()[1]))
            v_acc.append(float(match.groups()[2]))

    matplotlib.pyplot.figure()

    matplotlib.pyplot.subplot(121)
    matplotlib.pyplot.plot(t_it, t_err)
    matplotlib.pyplot.plot(v_it, v_err, '-rx', linewidth=2, markersize=10)
    matplotlib.pyplot.title('Learning curves: Classification')
    matplotlib.pyplot.ylabel('Cross-Entropy Loss')
    matplotlib.pyplot.xlabel('# Iteration')
    matplotlib.pyplot.grid(linestyle='--')
    matplotlib.pyplot.legend(['training', 'validation'])

    matplotlib.pyplot.subplot(122)
    matplotlib.pyplot.plot(t_it, t_acc)
    matplotlib.pyplot.plot(v_it, v_acc, '-rx', linewidth=2, markersize=10)
    matplotlib.pyplot.title('Learning curves: Classification')
    matplotlib.pyplot.ylabel('Accuracy')
    matplotlib.pyplot.xlabel('# Iteration')
    matplotlib.pyplot.grid(linestyle='--')
    matplotlib.pyplot.legend(['training', 'validation'])

    matplotlib.pyplot.show()


def stratify(df, col, n):
    vals = df[col].unique()

    out = []

    for i, val in enumerate(vals):
        local = df[df[col] == val]
        ix = numpy.random.choice(len(local), size=n, replace=True)

        temp = local.iloc[ix]
        out.append(temp)

    return pandas.concat(out, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=blurb)
    parser.add_argument(
        '--mode',
        type=str,
        default='train_annotation',
        help='script call',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/classifier',
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
        default=10,
        help='batch-size for training',
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='color',
        help='type of model attributes to train on',
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='run locally',
    )

    args = parser.parse_args()

    if args.mode == 'train':

        trainer = Trainer(
            args.checkpoint,
            args.model_type,
            args.n_epochs,
            args.lr,
            args.batch_size,
            args.local,
        ).train()

    elif args.mode == 'plot':

        plot(args.checkpoint)

    elif args.mode == 'test':

        model = torch.load(
            args.checkpoint + '/model.pt',
            map_location=lambda storage, loc: storage,
        )

        data = Data(args.model_type)

        model_type = str(args.checkpoint).split('/')[-1]

        iterator = Iterator(
            data,
            fold='test',
            model_type=model_type,
            local=args.local
        )

        choice = numpy.random.choice(len(iterator))
        image = iterator[choice][0]

        file_ = iterator.items.file.iloc[choice]
        label = iterator.items[model_type].iloc[choice]

        print(label)

        prediction = model(image[None, :, :, :]).squeeze().detach().numpy()

        best = iterator.lookup[numpy.argmax(prediction)]

        print(best)

        os.system('open ' + file_)
