from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utility.preprocess_data import PreprocessData


class SGNS(nn.Module):
    def __init__(self, vocal_size, emb_dim):
        super().__init__()
        self.vocal_size = vocal_size
        self.emb_dim = emb_dim
        self.u_embedding = nn.Embedding(vocal_size, emb_dim, sparse=True)  # optim.SparseAdam needed
        self.v_embedding = nn.Embedding(vocal_size, emb_dim, sparse=True)
        self.init_emb()

    def init_emb(self):
        init_range = 0.5 / self.emb_dim
        self.u_embedding.weight.data.uniform_(-init_range, init_range)
        self.v_embedding.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embedding(pos_u)
        emb_v = self.v_embedding(pos_v)
        neg_emb_v = self.v_embedding(neg_v)
        pos_score = F.logsigmoid(torch.sum(torch.mul(emb_u, emb_v), dim=1))
        neg_score = F.logsigmoid(-torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze())
        loss = - torch.sum(pos_score) - torch.sum(neg_score)
        return loss

    def save_embedding(self, id2program, output_file_path, use_cuda):
        if use_cuda:
            embedding = self.u_embedding.weight.cpu().data.numpy()
        else:
            embedding = self.u_embedding.weight.data.numpy()

        with open(output_file_path, 'w') as f:
            for id, program in id2program.items():
                e = embedding[id]
                e = ' '.join(map(lambda x: str(x), e))
                f.write(f'{program} {e}\n')
        print(f'Embeddings have been saved as {output_file_path}!')


class Program2Vec:
    def __init__(self, input_file_path, output_file_path, emb_dim=100, min_count=5, batch_size=50, window_size=2,
                 neg_number=5, iteration=2, lr=0.001, gpu=-1):
        self.data = PreprocessData(input_file_path, min_count, batch_size, window_size, neg_number)
        self.model = SGNS(len(self.data.program2id), emb_dim)
        self.optim = optim.SparseAdam(self.model.parameters(), lr=lr)
        self.output_file_path = output_file_path
        self.batch_size = batch_size
        self.iteration = iteration
        self.use_cuda = torch.cuda.is_available()
        self.gpu = gpu

    def train(self):
        pair_count = self.data.calculate_pair_count()
        batch_count = self.iteration * pair_count / self.batch_size
        device = f'cuda: {self.gpu}' if self.use_cuda else 'cpu'
        if self.use_cuda:
            print('CUDA available')
            self.model.to(device)
        progress_bar = tqdm(range(int(batch_count)))

        for i in progress_bar:
            pos_pairs = self.data.get_positive_pairs_batch()
            neg_v = self.data.get_neg_sample_batch(len(pos_pairs))
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]

            pos_u = torch.LongTensor(pos_u).to(device)
            pos_v = torch.LongTensor(pos_v).to(device)
            neg_v = torch.LongTensor(neg_v).to(device)

            loss = self.model.forward(pos_u, pos_v, neg_v)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if i % 1000 == 0:
                print(f'{i}th batch /total {batch_count} batch, Loss: {loss.item():0.6}')
            # progress_bar.set_description(f'Loss: {loss.item():0.6}')

        self.model.save_embedding(self.data.id2program, self.output_file_path, self.use_cuda)

