import json
import torch
import time

from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image



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

def manual_preprocess(images, size=224):
    """
    Manual preprocessing function that replicates:
    Resize(bicubic) -> CenterCrop -> ToTensor -> Normalize
    (without RGB conversion)
    
    Args:
        images: Single image (PIL Image, tensor, numpy array) or batch of images
            - For batch: list of images or 4D tensor (B, C, H, W)
        size: Target size for resize and crop (default: 224)
    
    Returns:
        torch.Tensor: Preprocessed image tensor(s)
                    - Single image: (C, H, W)
                    - Batch: (B, C, H, W)
    """
    
    def process_single_image(img):
        # Convert to PIL Image if it's a tensor or numpy array
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:  # Single image tensor (C, H, W)
                img = TF.to_pil_image(img)
            else:
                raise ValueError(f"Unexpected tensor dimensions: {img.dim()}")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        # 1. Resize with bicubic interpolation
        w, h = img.size
        if w < h:
            new_w = size
            new_h = int(size * h / w)
        else:
            new_h = size
            new_w = int(size * w / h)
        
        img = img.resize((new_w, new_h), Image.BICUBIC)
        
        # 2. Center Crop
        img = TF.center_crop(img, (size, size))
        
        # 3. Convert to Tensor
        tensor = TF.to_tensor(img)
        
        # 4. Normalize
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor
    
    # Handle different input types
    if isinstance(images, (list, tuple)):
        # Batch of images as list/tuple
        processed_batch = []
        for img in images:
            processed_batch.append(process_single_image(img))
        return torch.stack(processed_batch)
    
    elif isinstance(images, torch.Tensor) and images.dim() == 4:
        # Batch tensor (B, C, H, W)
        batch_size = images.shape[0]
        processed_batch = []
        for i in range(batch_size):
            single_img = images[i]
            processed_batch.append(process_single_image(single_img))
        return torch.stack(processed_batch)
    
    elif isinstance(images, np.ndarray) and images.ndim == 4:
        # Batch numpy array (B, H, W, C) or (B, C, H, W)
        batch_size = images.shape[0]
        processed_batch = []
        for i in range(batch_size):
            single_img = images[i]
            processed_batch.append(process_single_image(single_img))
        return torch.stack(processed_batch)
    
    else:
        # Single image
        return process_single_image(images)