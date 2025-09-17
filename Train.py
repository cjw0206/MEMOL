import torch.optim as optim

class Train_model(object):
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        # self.optimizer = optim.SGD(
        #     self.model.parameters(),
        #     lr=lr,
        #     weight_decay=weight_decay
        # )

    def train(self, dataset):
        loss, _, _, _,_,_,_= self.model(dataset)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.to('cpu').data.numpy()


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, data):
        (loss, correct_labels, predicted_labels,
         predicted_scores,rate1,rate2, rate3) = self.model(data, train=False)
        return loss, correct_labels, predicted_labels, predicted_scores,rate1,rate2, rate3

    '''
    def save_AUCs(self, AUCs, file_name):
        with open(file_name, "a+") as f:
            line = "epoch: " + str(AUCs[0]) + \
                   "    Size: " + str(AUCs[1]) + \
                   "    Batch: " + str(AUCs[2]) + \
                   "    loss: " + str(AUCs[3]) + \
                   "    AUC: " + str(AUCs[4]) + \
                   "    AUPRC: " + str(AUCs[5]) + \
                   "    Precision: " + str(AUCs[6]) + \
                   "    Recall: " + str(AUCs[7]) + \
                   "    F1: " + str(AUCs[8]) + \
                   "    rate1: " + str(AUCs[9]) + \
                   "    rate2: " + str(AUCs[10]) + \
                   "\n"
            f.write(line)
    '''