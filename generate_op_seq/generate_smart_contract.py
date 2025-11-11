import numpy as np
import torch
import pandas as pd
from generated_smart_contracts.model.baseVAE import baseVAE2
from keras_preprocessing import sequence
from tqdm import tqdm

from generated_smart_contracts.model.sentence_VAE import Seq2SeqVAE


def generate_text(name, num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_size = 128
    vocab_size = 64 #128
    hidden_size = 256
    latent_size = 128
    max_len = 500
    num_layers = 1
    # 实例化模型
    model = baseVAE2(embedding_size, vocab_size, hidden_size, latent_size, max_len, device)
    pre_model = "./train/baseVAE2_" + name + "_model_with_sem_64_gan.pt"
    print(pre_model)
    # model = Seq2SeqVAE(vocab_size, embedding_size, hidden_size, latent_size, max_len, num_layers, device)
    # pre_model = "E:\\Python_test\\research\\VAE_ShiXiong\\SCG\\Seq2SeqVAE_model.pt"
    # 加载模型参数
    model.load_state_dict(torch.load(pre_model, map_location=torch.device('cuda')))
    model.to(device)

    generated_texts = []
    z_list = []
    for i in tqdm(range(num), ncols=100):
        # 生成一个分布 z
        z = torch.randn(1, model.latent_size)  # 生成一个随机分布
        z_list.append(z.squeeze().numpy())
        # 生成文本
        generated_text = model.generate_text(z)
        opcode_seq_mat = sequence.pad_sequences([generated_text], maxlen=max_len)
        generated_texts.append(opcode_seq_mat)
        # print(np.array(generated_texts).shape)
        # 打印生成的文本
        # print("Generated Text: ", opcode_seq_mat)
        # print(np.array(generated_text).shape)
    z_array = np.array(z_list)
    z_df = pd.DataFrame(z_array)
    z_df.to_csv('original_latent_vectors.csv', index=False)

    # 将 generated_texts 转换为二维数组
    generated_texts = np.squeeze(np.array(generated_texts))
    print(generated_texts.shape)
    df = pd.DataFrame(generated_texts)
    df.to_csv("../dataset/embedding/generated_contract/generated_"+name+"_with_sem_64_gan.csv")

if __name__ == '__main__':
    dir = [
        "reentrancy", "timestamp", "delegatecall", "SBunchecked_low_level_calls",
        # "mix_vulnerabilities"
    ]
    for name in dir:
        if name =="mix_vulnerabilities":
            num = 5000
        else:
            num = 5000
        generate_text(name, num)