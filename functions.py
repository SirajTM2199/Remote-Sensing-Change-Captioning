import json
import torch
import time

from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor

import matplotlib.pyplot as plt


def load_json(path):
    with open(path) as f:
        file = json.load(f)
    f.close()
    return file


def save_json(file, path):
    with open(path, "w") as f:
        json.dump(file, f)
    f.close()
    print("Saved Successfully")


def rem_print(word):
    t_word = word
    for _ in range(250 - len(t_word)):
        word = word + " "
    print(word, end="\r")


# DATA
word_vocab = load_json("assets/vocab_mci.json")


# Validation
def model_validation(encoder, encoder_trans, decoder, dataloader, device):
    test_start_time = time.time()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    print("Validation.....\n")
    with torch.no_grad():
        # Batches
        for idx, batch_data in enumerate(dataloader):
            imgA, imgB, seg_label, token_all, token_all_len, token, token_len, name = (
                batch_data
            )

            # Getting Data and moving to GPU if possible
            imgA = imgA.cuda(device)
            imgB = imgB.cuda(device)
            # imgSM = imgSM.cuda(device)
            token = token.cuda(device)
            token_len = token_len.cuda(device)

            # Texts = get_text_inputs(name,split='test')

            # imgSM = ChangeCLIP.forward(xA=imgA,xB=imgB,Texts=Texts)

            # imgA, imgB, seg_label = Image_store_test[idx]

            feat1, feat2 = encoder(
                imgA.to(device), imgB.to(device), (seg_label > 0).int().to(device)
            )

            # eat1,feat2,feat3 = encoder(imgA.to(device),imgB.to(device),seg_label.to(device))
            feat = encoder_trans(feat1, feat2)

            seq = decoder.sample(feat, k=1)

            except_tokens = {
                word_vocab["<START>"],
                word_vocab["<END>"],
                word_vocab["<NULL>"],
            }
            img_token = token_all.tolist()
            img_tokens = list(
                map(lambda c: [w for w in c if w not in except_tokens], img_token[0])
            )  # remove <start> and pads
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in except_tokens]
            hypotheses.append(pred_seq)

            pred_caption = ""
            ref_caption = ""
            for i in pred_seq:
                pred_caption += (list(word_vocab.keys())[i]) + " "
            ref_caption = ""
            for i in img_tokens[0]:
                ref_caption += (list(word_vocab.keys())[i]) + " "
            ref_captions = ""
            for i in img_tokens:
                for j in i:
                    ref_captions += (list(word_vocab.keys())[j]) + " "
                ref_captions += ".    "

    test_time = time.time() - test_start_time
    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])]

    hypo = [
        [" ".join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]
    ]
    ref = [
        [" ".join(reft) for reft in reftmp]
        for reftmp in [
            [[str(x) for x in reft] for reft in reftmp] for reftmp in references
        ]
    ]
    score = []
    method = []

    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(
            method_i
        )
        # print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    get_eval_score  # score_dict = get_eval_score(references, hypotheses)
    Bleu_1 = score_dict["Bleu_1"]
    Bleu_2 = score_dict["Bleu_2"]
    Bleu_3 = score_dict["Bleu_3"]
    Bleu_4 = score_dict["Bleu_4"]
    # Meteor = score_dict['METEOR']

    # print(f"{VERSION}_{epoch} Results")
    # print(f'Testing:\n Time: {test_time}s\n BLEU-1: {Bleu_1*100}  %\n BLEU-2: {Bleu_2*100}  %\n BLEU-3: {Bleu_3*100}  %\n BLEU-4: {Bleu_4*100}  %\n Rouge: {Rouge*100}  %\n Cider: {Cider}\t')

    return {
        "Bleu_1": Bleu_1,
        "Bleu_2": Bleu_2,
        "Bleu_3": Bleu_3,
        "Bleu_4": Bleu_4,
        "test_time": test_time,
    }


def plot_metrics_over_epochs(metrics_dict, display=False, save_path=None):
    """
    Plots metric values over epochs.

    Args:
        metrics_dict (dict): Dictionary with epochs as keys and each value is a dict with keys:
            'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'Rouge', 'Cider', 'test_time'
    """
    # Sort epochs
    epochs = sorted(metrics_dict.keys())
    metrics = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "test_time"]
    values = {m: [metrics_dict[e][m] for e in epochs] for m in metrics}

    plt.figure(figsize=(12, 7))
    for m in metrics:
        if m != "test_time":
            plt.plot(epochs, np.array(values[m]) * 100, marker="o", label=m)
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.title("Evaluation Metrics over Epochs")
    plt.legend()
    plt.grid(True)
    if display:
        plt.show()
    if save_path:
        plt.savefig(save_path)
