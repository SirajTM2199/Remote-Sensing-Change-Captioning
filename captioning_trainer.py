# Imports
import os
import sys
import json
import random
import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt


sys.path.append("/home/guest/Documents/Siraj TM/RSCaMa")
from model.model_decoder import DecoderTransformer
from model.model_encoder_attMamba import EnhancedEncoder
from model.model_encoder_attMamba import CrossAttentiveEncoder
from utils_tool.utils import *
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence
from clip import clip
from PIL import Image
from functions import *

#Input args
#python3 train_cap VERSION EPOCH VAL_FREQ DEVICE 

VERSION,EPOCHS,VAL_FREQ,device = sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]

# Functions
print(f"VERSION:{VERSION} EPOCHS:{EPOCHS} FREQ: {VAL_FREQ} DEVICE: {device} ")



# Data
from dataloader import LEVIRMCIDataset_Modified

network='CLIP-ViT-B/32'
clip_model_type = network.replace("CLIP-", "")
clip_model, preprocess = clip.load(clip_model_type,device=device)

Dataset_Path = 'data/LEVIR-MCI-dataset/images'
token_path = 'Change-Agent/Multi_change/data/LEVIR_MCI/tokens/'

train_loader = data.DataLoader(
    LEVIRMCIDataset_Modified(
        data_folder=Dataset_Path,
        list_path="Change-Agent/Multi_change/data/LEVIR_MCI/",
        preprocess=preprocess,
        split="train",
        token_folder=token_path,
        vocab_file="vocab",
        max_length=42,
        allow_unk=1,
    ),
    batch_size=8,
    shuffle=True, 
    num_workers=36,
    pin_memory=True,
)
val_loader = data.DataLoader(
                LEVIRMCIDataset_Modified(data_folder=Dataset_Path, list_path='Change-Agent/Multi_change/data/LEVIR_MCI/',preprocess=preprocess, split='val', token_folder=token_path, vocab_file='vocab', max_length=42, allow_unk=1),
                batch_size=1, shuffle=True, num_workers=36, pin_memory=True)


word_vocab = load_json("assets/vocab_mci.json")


# Models
layers, atten_layers, decoder_layers, heads = VERSION.split(".")

decoder = DecoderTransformer(
    decoder_type="transformer_decoder",
    embed_dim=768,
    vocab_size=len(word_vocab),
    max_lengths=42,
    word_vocab=word_vocab,
    n_head=8,
    n_layers=int(decoder_layers),
    dropout=0.1,
    device=device,
).to(device)
encoder_trans = CrossAttentiveEncoder(
    n_layers=int(layers),
    feature_size=[7, 7, 768],
    heads=int(heads),
    dropout=0.1,
    atten_layers=int(atten_layers),
    device=device,
).to(device)
encoder = EnhancedEncoder("CLIP-ViT-B/32").to(device)
encoder.fine_tune(True)

# Optimizers and schedulers
num_epochs = EPOCHS

encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=1e-4)
encoder_trans_optimizer = torch.optim.Adam(
    params=filter(lambda p: p.requires_grad, encoder_trans.parameters()), lr=1e-4
)
decoder_optimizer = torch.optim.Adam(
    params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=1e-4
)

encoder_trans.cuda(device)
decoder.cuda(device)

encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    encoder_optimizer, step_size=5, gamma=1.0
)
encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    encoder_trans_optimizer, step_size=5, gamma=1.0
)
decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    decoder_optimizer, step_size=5, gamma=1.0
)
hist = np.zeros((num_epochs * 2 * len(train_loader), 5))

l_resizeA = torch.nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True)
l_resizeB = torch.nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True)
index_i = 0

criterion_cap = torch.nn.CrossEntropyLoss().cuda(device)



# Training Loop
print_freq = 5000
EPOCHS = num_epochs
index_i = 0
#network = "CLIP-ViT-B/32"
#clip_model_type = network.replace("CLIP-", "")
#clip_model, preprocess = clip.load(clip_model_type, device=device)

encoder.train()
encoder_trans.train()
decoder.train()

decoder_optimizer.zero_grad()
encoder_trans_optimizer.zero_grad()

# Image_store = {id:get_image(batch_data[-1],preprocess=preprocess) for id,batch_data in tqdm(enumerate(train_loader))}

benchmark = {}
MAX_SCORE = 0
Prev_Epoch = False
for epoch in range(0, EPOCHS):
    loss_set = []
    acc_set = []
    for id, batch_data in enumerate(train_loader):
        imgA, imgB, seg_label, token_all, token_all_len, token, token_len, name = (
            batch_data
        )

        # Texts = get_text_inputs(name,split='train')
        start_time = time.time()
        accum_steps = 64 // 64

        # Getting Data and moving to GPU if possible

        # imgA = imgA.cuda(device)
        # imgB = imgB.cuda(device)
        # imgSM = imgSM.cuda(device)
        token = token.cuda(device)
        token_len = token_len.cuda(device)

        # Feat1 and Feat2
        """with torch.no_grad():
            
            imgSM = ChangeCLIP.forward(xA=imgA,xB=imgB,Texts=Texts)"""

        # del imgA
        # del imgB

        # imgA, imgB, seg_label = Image_store[id]

        feat1, feat2 = encoder(
            imgA.to(device), imgB.to(device), (seg_label > 0).int().to(device)
        )

        featcap = encoder_trans(feat1, feat2)

        scores, caps_sorted, decode_lengths, sort_ind = decoder(
            featcap, token, token_len
        )
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        """seg_targets = imgSM[:,:,:,:1].permute(0,3,1,2)
        seg_targets = seg_targets.float()  # Convert to float
        
        if seg_targets.max() > 1.0:
            seg_targets = seg_targets / 255.0  # Normalize if needed
        """
        # Ensure targets are long integers for captioning

        targets = targets.long()
        loss = criterion_cap(scores, targets.to(torch.int64))

        loss = loss / accum_steps
        loss.backward()

        if (id + 1) % accum_steps == 0 or (id + 1) == len(train_loader):
            decoder_optimizer.step()
            encoder_trans_optimizer.step()

            # Adjust learning rate
            decoder_lr_scheduler.step()
            encoder_trans_lr_scheduler.step()

            decoder_optimizer.zero_grad()
            encoder_trans_optimizer.zero_grad()

        hist[index_i, 0] = time.time() - start_time  # batch_time
        hist[index_i, 1] = loss.item()  # train_loss

        accuracy = accuracy_v0(scores, targets, 5)

        hist[index_i, 2] = accuracy  # top5

        index_i += 1

        if index_i % 5 == 0:
            rem_print(
                f"Training Epoch: {epoch} | Index:{index_i} | Loss: {loss} | Top-5 Accuracy: {accuracy} "
            )
        loss_set.append(loss.item())
        acc_set.append(accuracy)

    # print(f'Training Epoch: {epoch} | Index:{index_i} | Mean Loss: {np.mean(loss_set)}\n')

    if (epoch % VAL_FREQ == 0 and epoch) or Prev_Epoch:
        print(
            f"Training Epoch: {epoch} | Index:{index_i} | Mean Loss: {np.mean(loss_set)} | Mean Accuracy : {np.mean(acc_set)}\n"
        )
        benchmark[epoch] = model_validation(encoder, encoder_trans, decoder, val_loader)
        print("\n")
        save_json(
            benchmark, f"benchmarks/benchmark.{VERSION}.json"
        )

        if benchmark[epoch]["Bleu_1"] * 100 > MAX_SCORE:
            MAX_SCORE = benchmark[epoch]["Bleu_1"] * 100
            print(f"\nNew Best Score: {MAX_SCORE[0]} at epoch {epoch}\n")

            torch.save(
                encoder.state_dict(),
                f"data/Pre-Trained Models/Finetuning/encoder_{VERSION}_best.pt",
            )
            torch.save(
                encoder_trans.state_dict(),
                f"data/Pre-Trained Models/Finetuning/encoder_trans_{VERSION}_best.pt",
            )
            torch.save(
                decoder.state_dict(),
                f"data/Pre-Trained Models/Finetuning/decoder_{VERSION}_best.pt",
            )

            Prev_Epoch = True
        else:
            print(
                f"No Improvement at epoch {epoch}, Previous Best Bleu_1: {MAX_SCORE[0]}"
            )
            Prev_Epoch = False
    print("\n")
    save_json(
        hist.tolist(),
        f"benchmarks/training_history.{VERSION}.json",
    )
    plot_loss_and_accuracy(hist,save_path=f'benchmarks/{VERSION}.png',epoch=epoch,data_loader_len=len(train_loader))

# torch.save(encoder.state_dict(),f'data/Pre-Trained Models/Finetuning/encoder_{VERSION}_{epoch}.pt')
# torch.save(encoder_trans.state_dict(),f'data/Pre-Trained Models/Finetuning/encoder_trans_{VERSION}_{epoch}.pt')
# torch.save(decoder.state_dict(),f'data/Pre-Trained Models/Finetuning/decoder_{VERSION}_{epoch}.pt')

save_json(benchmark, f"benchmarks/benchmark.{VERSION}.json")
save_json(
    hist.tolist(), f"benchmarks/training_history.{VERSION}.json"
)
plot_metrics_over_epochs(benchmark,save_path=f'benchmarks/metric_over_epoch_{VERSION}.png')


# To Do
# 1) Make three .py files
#    -> Trains just captioning with evaluation
#    -> Trains just segmenting with evaluation
#    -> Test captioning
#    -> Test segmenting
#    -> Evaluate both at the same time
#    -> Dataloader script
