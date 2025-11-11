import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# 超参数设置 Hyper-parameters
embedding_size = 100
h_dim = 400
z_dim = 128
num_epochs = 10
batch_size = 64
learning_rate = 1e-3
max_len = 500  # 句子长度


class Discriminator(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(max_len, vocab_size)
        self.input_dim = vocab_size * max_len
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        # print(f"x's shape: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten
        return self.net(x)


# VAE model
class baseVAE(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, latent_size, max_len):
        super(baseVAE, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_len = max_len

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.fc1 = nn.Linear(self.embedding_size * self.max_len, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_size)  # 均值 向量
        self.fc_logvar = nn.Linear(hidden_size, latent_size)  # 标准差 向量
        self.fc4 = nn.Linear(latent_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, self.embedding_size * self.max_len)
        self.softmax = nn.Softmax(dim=2)
    # 编码过程
    def encoder(self, x):
        x = self.embedding(x)
        x = x.view(-1, self.embedding_size * self.max_len)
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    # 随机生成隐含向量(重参数)
    def reparameterize(self, mean, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mean + eps * std
    # 解码过程
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        x = F.relu(self.fc5(h))
        x = x.view(-1, self.max_len, self.embedding_size)
        return F.softmax(x)
    # 整个前向传播过程：编码->解码
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        # print("z.shape:", z.shape)
        x_reconst = self.decoder(z)
        return x_reconst, mu, log_var


class baseVAE2(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, latent_size, max_len, device):
        super(baseVAE2, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_len = max_len
        self.device = device

        self.encoder_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.encoder_fc = nn.Linear(self.embedding_size * self.max_len, self.hidden_size)
        self.fc_mu = nn.Linear(self.hidden_size, self.latent_size)  # 均值 向量
        self.fc_logvar = nn.Linear(self.hidden_size, self.latent_size)  # 标准差 向量

        # self.decoder_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.decoder_fc = nn.Linear(self.latent_size, self.hidden_size)
        self.decoder_out = nn.Linear(self.hidden_size, self.vocab_size * self.max_len)
        # LSTM
        # self.decoder_lstm = nn.LSTM(self.latent_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fc_out = nn.Linear(self.hidden_size, self.vocab_size * self.max_len)

    # 随机生成隐含向量(重参数)
    def reparameterize(self, mean, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        # print((mean + eps * std).shape)
        return mean + eps * std
    
    # 解码过程
    def decoder(self, z):
        h = F.relu(self.decoder_fc(z))
        x = F.relu(self.decoder_out(h))
        x = x.view(-1, self.max_len, self.vocab_size)
        return F.softmax(x, dim=2)

    def decode_z(self, z):
        hidden = F.relu(self.decoder_fc(z))
        output = F.relu(self.decoder_out(hidden))
        output = output.view(-1, self.max_len, self.vocab_size)
        return output

    # 整个前向传播过程：编码->解码
    def forward(self, x):
        # Encode
        embed = self.encoder_embedding(x)  # embed.shape = [73, 500, 128]
        # print(embed.shape)

        embed = embed.view(-1, self.embedding_size * self.max_len)
        # print(embed.shape)
        hidden = self.encoder_fc(embed)  # [batch_size, hidden_size]

        # VAE
        mu = self.fc_mu(hidden)  # [batch_size, latent_size]
        logvar = self.fc_logvar(hidden)  # [batch_size, latent_size]
        z = self.reparameterize(mu, logvar)  # [batch_size, latent_size]

        #LSTM
        # output, _ = self.decoder_lstm(z)
        # output = self.fc_out(output)

        # Decode
        output = self.decoder_fc(z)  # [batch_size, hidden_size]
        output = self.decoder_out(output)  # [batch_size, max_len * vocab_size]
        output = output.view(-1, self.max_len, self.vocab_size)
        # print(output.shape)
        # output = F.softmax(output, dim=-2)
        # print(output.shape)
        return output, mu, logvar



    # 生成器
    def generate(self, z, length):
        # 整句一起生成
        # Decode
        outputs = []
        for _ in range(length):
            output = self.decoder_fc(embed)
            output = self.fc_out(output)
            output = output.view(-1, 1, self.vocab_size)
            outputs.append(output)

            # Randomly sample the next word from the output distribution 从输出分布中随机抽取下一个单词
            probs = F.softmax(output, dim=-1)
            next_word = torch.multinomial(probs[:, -1], num_samples=1)

            x = next_word
            embed = self.decoder_embedding(x)
            embed = torch.cat([embed, z.unsqueeze(1)], -1)
            embed = embed.view(-1, self.embedding_size + self.latent_size)

        return torch.cat(outputs, 1)

    def generate_text2(model, z):
        # 逐个单词生成
        device = model.device
        # max_len = model.max_len
        max_len = 500
        vocab_size = model.vocab_size
        model.eval()
        z = z.to(device)
        # 解码过程
        with torch.no_grad():
            hidden = F.relu(model.decoder_fc(z))
            output = model.decoder_out(hidden)
            output = output.view(-1, max_len, vocab_size)
            softmax_output = F.softmax(output, dim=2)
        # 生成文本
        generated_text = []
        current_token = 1
        for i in range(500):
            # 从每个时间步的概率分布中采样一个单词
            probs = softmax_output[:, i, :]
            generated_text.append(current_token)
            current_token = torch.multinomial(probs, num_samples=1).item()

        return generated_text

    def generate_text(model, z):
        # 逐个单词生成
        device = model.device
        # max_len = model.max_len
        max_len = 500
        vocab_size = model.vocab_size
        model.eval()
        z = z.to(device)
        # 解码过程
        with torch.no_grad():
            hidden = F.relu(model.decoder_fc(z))
            output = model.decoder_out(hidden)
            output = output.view(-1, max_len, vocab_size) ##要和输入的形状一样！！！
            softmax_output = F.softmax(output, dim=2)
        # 生成文本
        generated_text = []
        current_token = 1
        for i in range(random.randint(200,500)):
            # 从每个时间步的概率分布中采样一个单词
            probs = softmax_output[:, i, :]
            generated_text.append(current_token)
            current_token = torch.multinomial(probs, num_samples=1).item()
            
        return generated_text
