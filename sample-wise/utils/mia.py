from sklearn.linear_model import LogisticRegression
import torch

import numpy as np
from torch.nn import functional as F


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    for i in range(len(modified_probs)):
        modified_probs[i, labels[i]] = reverse_prob[i, labels[i]]
    # modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    for i in range(len(modified_probs)):
        modified_log_probs[i, labels[i]] = modified_log_probs[i, labels[i]]
    # modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model):
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=256, shuffle=False
    )
    prob = []
    targets = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
            targets.append(target)
    return torch.cat(prob), torch.cat(targets)


def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob, retain_label = collect_prob(retain_loader, model)
    forget_prob, forget_label = collect_prob(forget_loader, model)
    test_prob, test_label = collect_prob(test_loader, model)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


def get_membership_attack_data_m(retain_loader, forget_loader, test_loader, model):
    retain_prob, retain_label = collect_prob(retain_loader, model)
    forget_prob, forget_label = collect_prob(forget_loader, model)
    test_prob, test_label = collect_prob(test_loader, model)

    X_r = (
        torch.cat([m_entropy(retain_prob, retain_label), m_entropy(test_prob, test_label)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = m_entropy(forget_prob, forget_label).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r

import time
def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):

    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )


    # clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)

    results = clf.predict(X_f)
    return results.mean() * 100


def get_membership_attack_prob_m(retain_loader, forget_loader, test_loader, model):

    X_f, Y_f, X_r, Y_r = get_membership_attack_data_m(
        retain_loader, forget_loader, test_loader, model
    )


    # clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)

    results = clf.predict(X_f)
    return results.mean() * 100