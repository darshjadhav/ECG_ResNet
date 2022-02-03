import time
import pandas as pd
import numpy as np
import torch
from torch import nn
import pytorch_model_summary as pms
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from data_loader import *


# Using a function to apply 1d convolution where kernel_size=7
def conv3x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

# Using a function to apply 1d convolution where kernel_size=1
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# Residual Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    # Feed data through the network
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Bottleneck layer to reduce overfitting
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    # Feed data through the network
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ResNet model (defining architecture)
class ResNet(nn.Module):
    # in_channel=number of leads, out_channel=number of classes
    def __init__(self, block, layers, in_channel=12, out_channel=27, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64 # Batch size
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=5000, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    # Feed data through the network
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

# ResNet18 model
def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

# ResNet34 model
def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

# Training ResNet
def train(data, labels, validationleads, validationlabels, epochnum, threshval, opt, resnettype):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    # Batch size
    batchval=64
    batchind = np.array(range(0, data.shape[0], batchval))
    #Initialise Model
    if resnettype == 18:
        model = resnet18().double()
    elif resnettype == 34:
        model = resnet34().double()
    model.cuda()
    # Array to store trained models
    models = []
    # Optimiser
    if opt == 0:
        optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-5)
    elif opt == 1:
        optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initialise loss function
    criterion = nn.BCEWithLogitsLoss()
    # Model Save path
    model_save_path = "torch_models/"
    # Ndarray to store predicted labels
    pred_labels = np.zeros(shape=labels.shape)
    # Used to store models based on f-measure values
    prev_f_measure_val = 0.0
    # Arrays to store history of results
    f_measure_hist = np.ones(epochnum)
    training_loss_hist = []
    validation_loss_hist = []
    training_acc_hist = []
    validation_acc_hist = np.ones(epochnum)
    loss_batches = 0.0
    # Training phase
    for epoch in range(0, epochnum):
        model.train()
        print("TRAINING EPOCH {}".format(epoch))
        batchcounter = 0
        loss_epoch = 0.0
        loss = 0.0
        pred_labels = np.zeros(shape=labels.shape)
        epoch_training_loss = []
        # For each epoch, train on the training set
        with torch.set_grad_enabled(True):
            for i in batchind[:-1]:
                # training data - Numpy to Tensor
                x = torch.from_numpy(data[i:i+batchval,:,:] / 1000)
                x = x.cuda()
                
                # training labels - Numpy to Tensor
                trainlabels = torch.from_numpy(labels[i:i+batchval,:]).double()
                trainlabels = trainlabels.to(device=dev)
                
                # Reset gradients
                optimiser.zero_grad()
                
                # Make prediction
                y = model(x).to(dev)
                # Apply output layer function to predictions
                y_prob = torch.sigmoid(y)
                
                # Calculate loss
                loss = criterion(y, trainlabels)
                temploss = loss.item() * x.size(0)
                loss_epoch += temploss
                
                # Backwards step
                loss.backward()
                optimiser.step()
                
                # Calculate loss
                loss_batches += temploss
                batchcounter += x.size(0)
                current_loss = loss_batches / batchcounter

                # Store predictions and labels
                if i == 0:
                    all_train_labels = trainlabels
                    all_pred_prob = y_prob
                else:
                    all_train_labels = torch.cat((all_train_labels, trainlabels), 0)
                    all_pred_prob = torch.cat((all_pred_prob, y_prob), 0)

                # Covert predictions from tensor to numpy
                pred_labels[i:i+batchval,:] = y_prob.cpu().detach().numpy()

                # Apply threshold to predictions to recieve binary output
                pred_labels[pred_labels < threshval] = 0
                pred_labels[pred_labels > threshval] = 1

                # Calculate Training Accuracy
                training_acc = accuracy_score(labels[i:i+batchval,:], pred_labels[i:i+batchval,:])

                # Store Loss
                epoch_training_loss.append(current_loss)

                # Output current training information
                print("{}".format(time.strftime("%H:%M:%S", time.localtime())), ", Epoch: {} [{}/{}],".format(epoch,i+batchval,data.shape[0]), 
                "Loss: {}".format(current_loss), "Accuracy: {}".format(training_acc))
            
            training_acc_hist.append(training_acc)
            training_loss_hist.append(epoch_training_loss)
        
        # Generate confusion matrix
        conf_mat_train = multilabel_confusion_matrix(labels[:pred_labels.shape[0],:], pred_labels)
        class_report = classification_report(labels[:pred_labels.shape[0],:], pred_labels)
    


        # Evaluation mode (using validation set)
        model.eval()
        # Batch size
        val_batchval=64
        batchindval = np.array(range(0, validationleads.shape[0], val_batchval))
        valid_loss = 0.0 # Loss variable
        batchcounterval = 0 # batch counter variable
        val_pred_labels = np.zeros(shape=validationlabels.shape)
        epoch_validation_loss = []
        # Initialise loss function
        criterion = nn.BCEWithLogitsLoss()
        with torch.set_grad_enabled(False):
            for j in batchindval[:-1]:
                # validation leads - numpy to tensor
                x_val = torch.from_numpy(validationleads[j:j+val_batchval,:,:] / 1000)
                x_val = x_val.cuda()
                # validation labels - numpy to tensor
                val_labels = torch.from_numpy(validationlabels[j:j+val_batchval,:]).double().cuda()
                val_labels = val_labels.to(device=dev)

                # Make prediction
                val_y = model(x_val).to(device=dev)
                # Apply output layer function
                val_y_prob = torch.sigmoid(val_y)
                
                # Calculate loss
                val_loss = criterion(val_y, val_labels)
                valid_loss = val_loss.item() * x_val.size(0)
                batchcounterval += x_val.size(0)
                current_val_loss = valid_loss / batchcounterval

                # Store raw predictions
                if j == 0:
                    all_val_labels = val_labels
                    all_val_prob = val_y_prob
                else:
                    all_train_labels = torch.cat((all_val_labels, val_labels), 0)
                    all_pred_prob = torch.cat((all_val_prob, val_y_prob), 0)

                # Output predicted values from tensor to numpy
                val_pred_labels[j:j+val_batchval,:] = y_prob.cpu().detach().numpy()

                # Ouput current validation loss
                epoch_validation_loss.append(current_val_loss)
                print("Epoch: {}".format(epoch), "Validation Loss: {}".format(current_val_loss))
        
        # Calculating and Storing f-measure and validation accuracy
        validation_loss_hist.append(epoch_validation_loss)
        models.append(model)
        
        tempfmeasure = []
        tempaccmeasure = []


        val_pred_labels[val_pred_labels > threshval] = 1
        val_pred_labels[val_pred_labels < threshval] = 0
        
        val_pred_labels = val_pred_labels.astype(int)
        f_measure_val = f1_score(validationlabels, val_pred_labels, average='samples')
        validation_acc = accuracy_score(validationlabels, val_pred_labels)
        tempfmeasure.append(f_measure_val)
        tempaccmeasure.append(validation_acc)

        validation_acc_hist[epoch] = validation_acc

        f_measure_hist[epoch] = f_measure_val

        print("F-MEASURE ({}): {}".format(threshval, f_measure_val), "ACCURACY: {}".format(validation_acc))

        # Save model to path
        if f_measure_val > prev_f_measure_val:
            newmodelpath = model_save_path + "model_e" + str(epoch) + "_f" + str(f_measure_val) + ".pth"
            torch.save(model.state_dict(), newmodelpath)
            prev_f_measure_val = f_measure_val

    # print(challenge_metric_hist)    
    print("F-measure History: ", f_measure_hist)
    print("Accuracy History: ", validation_acc_hist)

    return training_loss_hist, validation_loss_hist, f_measure_hist, validation_acc_hist, conf_mat_train, class_report, models

# Testing on Test Set
def test_eval(model, testingleads, testinglabels, threshval):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    with torch.no_grad():
        model.cuda()
        # Batch size
        batchval=64
        batchind = np.array(range(0, testingleads.shape[0], batchval))
        test_pred_labels = np.zeros(shape=testinglabels.shape)

        for i in batchind[:-1]:
            # testing leads - numpy to tensor
            x = torch.from_numpy(testingleads[i:i+batchval,:,:] / 1000)
            x = x.cuda()

            # testing labels - numpy to tensor
            test_labels = torch.from_numpy(testinglabels[i:i+batchval,:]).double().cuda()

            # Make prediction
            y = model(x).to(device=dev)
            # Apply output layer function
            test_y_prob = torch.sigmoid(y)

            # Store predictions
            if i == 0:
                all_test_labels = test_labels
                all_test_pred_prob = test_y_prob
            else:
                all_test_labels = torch.cat((all_test_labels, test_labels), 0)
                all_test_pred_prob = torch.cat((all_test_pred_prob, test_y_prob), 0)

            # Get predicted labels in numpy format
            test_pred_labels[i:i+batchval,:] = test_y_prob.cpu().detach().numpy()

            test_pred_labels = test_pred_labels.astype(int)

            test_pred_labels[test_pred_labels > threshval] = 1
            test_pred_labels[test_pred_labels < threshval] = 0

            # Calculate Accuracy
            f_measure_temp_val = accuracy_score(testinglabels[i:i+batchval,:], test_pred_labels[i:i+batchval,:])
                
            # Print testing accuracy per batch
            print("Accuracy: {}".format(f_measure_temp_val))
        finalpred = all_test_pred_prob.cpu().detach().numpy()
        finalpred[finalpred > threshval] = 1
        finalpred[finalpred < threshval] = 0

        # Calculate weighted f-measure
        FMEASUREVAL = f1_score(testinglabels[:all_test_pred_prob.shape[0],:], finalpred, average='samples')
        # Ouput confusion matrix and classification report
        conf_mat_test = multilabel_confusion_matrix(testinglabels[:finalpred.shape[0],:], finalpred)
        test_class_report = classification_report(testinglabels[:finalpred.shape[0],:], finalpred)
        print(FMEASUREVAL)

        return conf_mat_test, test_class_report



def main():
    trainingleads, testingleads, traininglabels, testinglabels, validationleads, validationlabels = dataloader("../Data/fullecgdata.pkl")
    training_loss_hist, validation_loss_hist, f_measure_hist, validation_acc_hist, conf_mat_train, class_report, models = train(trainingleads, traininglabels, validationleads, validationlabels, epochnum=1, threshval=0.35, opt=0, resnettype=34)
    testingmodel = models[0]
    conf_mat_test, test_class_report = test_eval(testingmodel, testingleads, testinglabels, threshval=0.3)


if __name__ == "__main__":
    main()

