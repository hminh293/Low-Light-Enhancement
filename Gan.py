from utils.dataset import get_loader
from models.gan import Generator, Discriminator
from turtle import forward
from torch._dynamo.convert_frame import input_codes
from utils.dataset import get_loader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim
from tqdm import tqdm
import os
import joblib
import gc

if __name__ == "__main__":
    root_dir = './dataset'
    if not os.path.exists(root_dir):
        print("Dataset not found")
        exit()
    mode = 'train'
    batch_size = 16
    num_workers = 4
    train_loader = get_loader(root_dir, mode, batch_size, num_workers)
    valid_loader = get_loader(root_dir, 'val', batch_size, num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    
    
    model_gen = Generator()
    model_dis = Discriminator()
    
    model_gen.to(device)
    model_dis.to(device)
    
    optimizer_Gen = torch.optim.Adam(params = model_gen.parameters(), lr = 0.001)
    optimizer_Dis = torch.optim.Adam(params = model_dis.parameters(), lr = 0.001)
    
    criteria_GAN = nn.BCEWithLogitsLoss()
    criteria = nn.L1Loss()
    
    if not os.path.exists('./checkpoints/GANs'):
        os.makedirs('./checkpoints/GANs')
        
    gc.collect()
    torch.cuda.empty_cache()
    start_epoch = 0
    try: 
        last_Gen_checkpoint = sorted(os.listdir('./checkpoints/GANs/Gen'))[-1]
        last_Dis_checkpoint = sorted(os.listdir('./checkpoints/GANs/Dis'))[-1]
        model_gen.load_state_dict(torch.load(os.path.join('./checkpoints/GANs/Gen', last_Gen_checkpoint)))
        model_dis.load_state_dict(torch.load(os.path.join('./checkpoints/GANs/Dis', last_Dis_checkpoint)))
        start_epoch = int(last_Gen_checkpoint.split('_')[-1].split('.')[0])
        print(f"Loaded checkpoint: {last_Gen_checkpoint}")
        print(f"Resuming training from epoch {start_epoch}")
    except:
        print("No checkpoint found, training from scratch.")
        
    for epoch in range(start_epoch, epochs):
        gen_loss = 0.0
        dis_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+start_epoch+1}/{epochs} [Train]")

        for batch_idx, (low_img, light_img) in enumerate(train_bar):
            batch_idx += 1
            low_img = low_img.to(device)
            light_img = light_img.to(device)
            
            model_gen.eval()
            model_dis.train()
            
            optimizer_Dis.zero_grad()
            # train với ảnh thật trước
            
            D_light = model_dis(light_img)
            true_target = D_light.new_full(D_light.size(), 0.9)
            loss_light = criteria_GAN(D_light, true_target)
            
            with torch.no_grad():
                fake_light_img = model_gen(low_img)
            D_low = model_dis(fake_light_img.detach())
            fake_target = D_low.new_full(D_low.size(), 0.1)
            loss_low = criteria_GAN(D_low, fake_target)
            
            loss_D = (loss_light + 0.5 * loss_low) * 0.5
            loss_D.backward()
            optimizer_Dis.step()
            
            model_gen.train()
            model_dis.eval()
            
            optimizer_Gen.zero_grad()
            
            gen_fake = model_gen(low_img)
            loss_G_l1 = criteria(gen_fake, light_img)
            
            dis_eval = model_dis(gen_fake)
            true_target_G = dis_eval.new_full(dis_eval.size(), 0.9)
            loss_G_adv = criteria_GAN(dis_eval, true_target_G)
            
            loss_G = loss_G_adv + 100 * loss_G_l1
            loss_G.backward()
            optimizer_Gen.step()
            gen_loss += loss_G.item()
            dis_loss += loss_D.item()
            avg_g_loss = gen_loss / (batch_idx + 1)
            avg_d_loss = dis_loss / (batch_idx + 1)
            train_bar.set_postfix(G_loss=f'{avg_g_loss:.4f}', D_loss=f'{avg_d_loss:.4f}')
        torch.save(model_dis.state_dict(), f'./checkpoints/GANs/Dis/generator_epoch_{epoch+start_epoch+1}.pth')
        torch.save(model_gen.state_dict(), f'./checkpoints/GANs/Gen/generator_epoch_{epoch+start_epoch+1}.pth')

        # with torch.no_grad():
        #     val_bar = tqdm(valid_loader, desc=f"Epoch {epoch+start_epoch+1}/{epochs} [Valid]")
        #     for low_img, light_img in val_bar:
        #         low_img = low_img.to(device)
        #         light_img = light_img.to(device)
        #         pred = model(low_img)
        #         loss = criteria(pred, light_img)
        #         valid_loss += loss.item()
        #         val_bar.set_postfix(loss=loss.item())
        # avg_valid_loss = valid_loss / len(valid_loader)
        

        