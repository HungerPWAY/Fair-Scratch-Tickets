import torch
import pdb
import pandas as pd


TARGET_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
    "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses",
    "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes",
    "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling",
    "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young"
]


def getDevice(gpu_id=None):

    #CPU
    device = 'cpu'

    #GPU
    if torch.cuda.is_available():

        device = 'cuda'

        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        cuda_str = 'Using CUDA '
        for i in range(0, ng):

            if i == gpu_id:
                device += ':' + str(i)

            if i == 1:
                cuda_str = ' ' * len(cuda_str)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                (cuda_str, i, x[i].name, x[i].total_memory / c))
            
    print('Device:', device)
    return device


def calculateAccuracy(outputs, targets, threshold=0.5):
    """
        Calculates the average accuracy.
          outputs: Tensor of shape (batch_size, num_attributes)
          targets: Tensor of shape (batch_size, num_attributes)
          threshold: float
    """

    preds = torch.sigmoid(outputs) > threshold

    # Per-attribute average accuracy
    # Tensor of shape (num_attributes)
    accuracy = torch.true_divide((preds == targets).sum(0) * 1.0, targets.size(0))
    
    # Overall average accuracy
    # Float tensor
    average_accuracy = accuracy.sum() / targets.size(1)
    return average_accuracy, accuracy

def calculateConfusionMatrix(preds, targets):
    """
    Adapted from https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
    """
    confusion_vector = torch.true_divide(preds, targets)
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    true_positives = torch.sum(confusion_vector == 1, 0)
    false_positives = torch.sum(confusion_vector == float('inf'), 0)
    true_negatives = torch.sum(torch.isnan(confusion_vector), 0)
    false_negatives = torch.sum(confusion_vector == 0, 0)
    return true_positives, false_positives, true_negatives, false_negatives

def calculateGenderConfusionMatrices(outputs, targets, genders, threshold=0.5):
    preds = torch.sigmoid(outputs) > threshold
    cm_m = calculateConfusionMatrix(preds[genders], targets[genders])
    cm_f = calculateConfusionMatrix(preds[genders == 0], targets[genders == 0])
    return cm_m, cm_f

def calculateProbCorrect(cm):
    t_p, f_p, t_n, f_n = cm
    prob_correct_1 = torch.true_divide(t_p, t_p + f_n)
    prob_correct_0 = torch.true_divide(t_n, t_n + f_p)
    for prob_correct in [prob_correct_1, prob_correct_0]:
        prob_correct[torch.isnan(prob_correct)] = 0
    return prob_correct_1, prob_correct_0

def calculateEqualityGap(cm_m, cm_f):
    prob_correct_1_m, prob_correct_0_m = calculateProbCorrect(cm_m)
    prob_correct_1_f, prob_correct_0_f = calculateProbCorrect(cm_f)
    equality_gap_1 = (prob_correct_1_m - prob_correct_1_f).abs()
    equality_gap_0 = (prob_correct_0_m - prob_correct_0_f).abs()
    average_equality_gap_1 = equality_gap_1.mean()
    average_equality_gap_0 = equality_gap_0.mean()
    return average_equality_gap_0, average_equality_gap_1, equality_gap_0, equality_gap_1

def calculateProbTrue(cm):
    t_p, f_p, t_n, f_n = cm
    prob_true = torch.true_divide(t_p + f_p, t_p + f_p + t_n + f_n)
    prob_true[torch.isnan(prob_true)] = 0
    return prob_true

def calculateParityGap(cm_m, cm_f):
    prob_true_m = calculateProbTrue(cm_m)
    prob_true_f = calculateProbTrue(cm_f)
    parity_gap = (prob_true_m - prob_true_f).abs()
    average_parity_gap = parity_gap.mean()
    return average_parity_gap, parity_gap

def save_attr_metrics(accuracy, equality_gap_0, equality_gap_1, parity_gap, filename):
    df = pd.DataFrame(data=[accuracy.view(-1).cpu().numpy(), equality_gap_0.view(-1).cpu().numpy(),
                            equality_gap_1.view(-1).cpu().numpy(), parity_gap.view(-1).cpu().numpy()],
                      index=['accuracy', 'equality_gap_0', 'equality_gap_1', 'parity_gap'], columns=TARGET_ATTRIBUTES)
    df.to_csv(filename + '.csv')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, shape=None, device='cpu'):
        self.shape = shape
        self.device = device
        self.reset()

    def reset(self):

        self.val = torch.zeros(self.shape, dtype=torch.float, device=self.device) if self.shape else 0
        self.avg = torch.zeros(self.shape, dtype=torch.float, device=self.device) if self.shape else 0
        self.sum = torch.zeros(self.shape, dtype=torch.float, device=self.device) if self.shape else 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.shape:
            self.avg = torch.true_divide(self.sum, self.count)
        else:
            self.avg = self.sum / self.count
