import torch
from torch.utils.data.dataloader import DataLoader
# from data import PrepASV15Dataset, PrepASV19Dataset
import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from ASVRawDataset import ASVRawDataset
import os



def asv_cal_accuracies(test_loader, net, device):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        softmax_acc = 0
        num_files = 0
        probs = torch.empty(0, 3).to(device)


        # test_loader = DataLoader(test_loader, batch_size=64, shuffle=False, num_workers=4)

        for test_batch in test_loader:
            # load batch and infer
            test_sample, _, _, test_label = test_batch
            num_files += len(test_label)
            test_sample = test_sample.to(device)
            test_label = test_label.to(device)
            infer = net(test_sample)

            # obtain output probabilities
            t1 = F.softmax(infer, dim=1)
            t2 = test_label.unsqueeze(-1)
            row = torch.cat((t1, t2), dim=1)
            probs = torch.cat((probs, row), dim=0)

            # calculate example level accuracy
            infer = infer.argmax(dim=1)
            batch_acc = infer.eq(test_label).sum().item()
            softmax_acc += batch_acc

        softmax_acc = softmax_acc / num_files

    return softmax_acc, probs.to('cpu')

def asv_cal_socres(test_loader, net, device, comment, epoch):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        softmax_acc = 0
        num_files = 0
        probs = torch.empty(0, 3).to(device)
        fname_list = []
        key_list = []
        att_id_list = []
        key_list = []
        score_list = []

        # test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

        for step, test_batch in enumerate(test_loader):
            # load batch and infer
            test_sample, file_name, attack_id, test_label = test_batch

            test_sample = test_sample.to(device)
            num_files += len(test_label)
            test_label = test_label.to(device)
            infer = net(test_sample)

            t1 = F.softmax(infer, dim=1)
            t2 = test_label.unsqueeze(-1)
            row = torch.cat((t1, t2), dim=1)
            probs = torch.cat((probs, row), dim=0)

            # calculate example level accuracy
            infer = infer.argmax(dim=1)
            batch_acc = infer.eq(test_label).sum().item()
            softmax_acc += batch_acc

            fname_list.extend(list(file_name))
            key_list.extend(
            ['bonafide' if key == 1 else 'spoof' for key in test_label.tolist()])
            att_id_list.extend(list(attack_id))
            score_list.extend(t1[:,1].tolist())
        save_folder = os.path.join(comment, 'scores')
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_path = os.path.join(save_folder, 'epoch_{}.txt'.format(epoch))
        with open(save_path, 'a') as fh:
            for f, s, k, cm in zip(fname_list, att_id_list, key_list, score_list):
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))

        softmax_acc = softmax_acc / num_files
    return softmax_acc, probs.to('cpu')


def cal_roc_eer(probs, show_plot=True):
    """
    probs: tensor, number of samples * 3, containing softmax probabilities
    row wise: [genuine prob, fake prob, label]
    TP: True Fake
    FP: False Fake
    """
    all_labels = probs[:, 2]
    zero_index = torch.nonzero((all_labels == 0)).squeeze(-1)
    one_index = torch.nonzero(all_labels).squeeze(-1)
    zero_probs = probs[zero_index, 0]
    one_probs = probs[one_index, 0]

    threshold_index = torch.linspace(-0.1, 1.01, 10000)
    tpr = torch.zeros(len(threshold_index),)
    fpr = torch.zeros(len(threshold_index),)
    cnt = 0
    for i in threshold_index:
        tpr[cnt] = one_probs.le(i).sum().item()/len(one_probs)
        fpr[cnt] = zero_probs.le(i).sum().item()/len(zero_probs)
        cnt += 1

    sum_rate = tpr + fpr
    distance_to_one = torch.abs(sum_rate - 1)
    eer_index = distance_to_one.argmin(dim=0).item()
    out_eer = 0.5*(fpr[eer_index] + 1 - tpr[eer_index]).numpy()

    if show_plot:
        print('EER: {:.4f}%.'.format(out_eer * 100))
        plt.figure(1)
        plt.plot(torch.linspace(-0.2, 1.2, 1000), torch.histc(zero_probs, bins=1000, min=-0.2, max=1.2) / len(zero_probs))
        plt.plot(torch.linspace(-0.2, 1.2, 1000), torch.histc(one_probs, bins=1000, min=-0.2, max=1.2) / len(one_probs))
        plt.xlabel("Probability of 'Genuine'")
        plt.ylabel('Per Class Ratio')
        plt.legend(['Real', 'Fake'])
        plt.grid()

        plt.figure(3)
        plt.scatter(fpr, tpr)
        plt.xlabel('False Positive (Fake) Rate')
        plt.ylabel('True Positive (Fake) Rate')
        plt.grid()
        plt.show()

    return out_eer


if __name__ == '__main__':

    test_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    protocol_file_path = 'F:/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
    data_path = 'F:/ASVSpoof2019/LA/data/dev_6/'

   # 'F:/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
   # 'F:/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
   # 'F:/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

    # protocol_file_path = 'F:/ASVspoof2015/CM_protocol/cm_develop.ndx.txt'
    # # cm_train.trn
    # # cm_develop.ndx
    # # cm_evaluation.ndx
    # data_path = 'F:/ASVspoof2015/data/dev_6/'

    Net = models.SSDNet1D()
    num_total_learnable_params = sum(i.numel() for i in Net.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

    check_point = torch.load('./trained_models/***.pth')
    Net.load_state_dict(check_point['model_state_dict'])

    accuracy, probabilities = asv_cal_accuracies(protocol_file_path, data_path, Net, test_device, data_type='time_frame', dataset=19)
    print(accuracy * 100)

    eer = cal_roc_eer(probabilities)

    print('End of Program.')
