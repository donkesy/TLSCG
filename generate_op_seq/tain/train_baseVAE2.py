import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from generate_op_seq.model.baseVAE import baseVAE2, Discriminator
import pandas as pd 
import pickle
from tools import getdataset, get_Bigram, embedding2opcode


# 自定义数据加载器
class MyDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        input = self.data[index][:500]  # 只选择前500个词作为输入
        target = self.data[index][:500]  # 只选择前500个词作为目标
        return {'input': input, 'target': target}

    def __len__(self):
        return len(self.data)


def train(name):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_dir = './result/basevae/'
    # 超参数设置 Hyper-parameters
    embedding_size = 128
    vocab_size = 128
    hidden_size = 256
    latent_size = 128
    num_epochs = 500
    batch_size = 128
    learning_rate = 1e-5
    kl_weight = 0.1
    pad_idx = 0
    max_len = 500  # 句子长度

    # 读取数据
    X_train = getdataset(name)
    # 加载数据
    train_data = MyDataset(X_train)
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 实例化模型
    model = baseVAE2(embedding_size, vocab_size, hidden_size, latent_size, max_len, device)  # .to(device)
    # model_state_dict = torch.load(f'baseVAE2_normal_all_model_with_sem_64.pt')
    # model.load_state_dict(model_state_dict)
    model = model.to(device)

    discriminator = Discriminator(vocab_size, max_len, hidden_size)
    discriminator = discriminator.to(device)
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # 交叉熵损失函数，忽略0
    criterion2 = nn.MSELoss()
    bce_loss = nn.BCELoss()

    print("=======train begin========")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, batch in enumerate(data_loader):
            input = batch['input'].to(device)
            target = batch['target'].to(device)

            real_x = input.to(device)
            # print(real_x.shape)
            # print(type(real_x))
            # exit(0)
            # ======== 训练 Discriminator ========
            with torch.no_grad():
                _, mu, logvar = model(real_x)
                z = model.reparameterize(mu, logvar)
                fake_logits = model.decode_z(z)
                fake_sample = torch.softmax(fake_logits, dim=2)
                fake_sample = torch.argmax(fake_sample, dim=2)
            # print(f"real_x's shape: {real_x.shape}")
            # print(fake_sample, fake_sample.shape)
            real_score = discriminator(real_x)
            fake_score = discriminator(fake_sample.detach())
            d_loss = bce_loss(real_score, torch.ones_like(real_score)) + \
                     bce_loss(fake_score, torch.zeros_like(fake_score))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ======== 训练 VAE（Encoder + Decoder） ========
            optimizer.zero_grad()
            output, mu, logvar = model(input)
            vae_logits = output
            # print(output.shape)
            # print(target.shape)

            # Reshape for loss calculation
            recon_x = output.permute(0, 2, 1)
            output = output.view(-1, output.size(2))
            target = target.view(-1)
            # print(f"output's shape: {output.shape}")

            Boutput = F.softmax(output, dim=-1)
            Boutput = torch.sum(Boutput, dim=1)
            # Boutput = model.generate_text2(model.reparameterize(mu, logvar))
            # print(Boutput.long().tolist())
            # print(target.tolist())
            # print("target's shape: ", end='')
            # print(target.shape)
            # print("output's shape: ", end='')
            # print(Boutput.shape)
            # calculate the loss
            recon_loss = criterion(output, target)  # 重构损失
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL散度

            # GAN loss:
            # print(output.shape)
            gen_sample = torch.softmax(vae_logits, dim=2)
            gen_sample = torch.argmax(gen_sample, dim=2)
            gen_score = discriminator(gen_sample)
            adv_loss = bce_loss(gen_score, torch.ones_like(gen_score))

            # g_output = model.generate_text(z)
            Btarget = get_Bigram([target.tolist()])
            Boutput = get_Bigram([Boutput.long().tolist()])
            Bigram_loss = criterion2(torch.tensor(Boutput), torch.tensor(Btarget))  # 语义损失

            # 反向传播和优化
            loss = recon_loss + kl_weight * kl_loss + 0.5 * adv_loss + 0.3 * Bigram_loss
            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器对 x 的值进行更新

            epoch_loss += loss.item()  # 当前loss值
            # 利用训练的模型进行测试
        print(f'Epoch {epoch + 1}: Loss = {epoch_loss / len(data_loader)}')
    torch.save(model.state_dict(), 'baseVAE2_' + name + '_model_with_sem_gan_final.pt')
    return model


if __name__ == '__main__':
    dir = [
        # "reentrancy", "timestamp", "delegatecall",
        #    "SBunchecked_low_level_calls"
        "normal_all"
                                                      # " mix_vulnerabilities"
           ]
    for name in dir:
        train(name)
    # train("mix_vulnerabilities")
