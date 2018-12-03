import numpy as np
from collections import deque, defaultdict
# from torch.utils.data import Dataset

class PreprocessData:
    def __init__(self, input_file_path, min_count, batch_size, window_size, neg_number):
        self.input_file_path = input_file_path
        self.input_file = open(input_file_path)
        self.min_count = min_count
        self.batch_size = batch_size
        self.window_size = window_size
        self.neg_number = neg_number
        self.sequence_count = 0
        self.program_count = 0
        self.program2id = {}
        self.id2program = {}
        self.program_frequency = {}
        self.sample_table = []
        self.program_pairs_catch = deque()
        self.vocal_size = None

        self._bulid_vocal()
        self._init_sampling_table()


    def _bulid_vocal(self):
        with open(self.input_file_path) as f:
            raw_program_frequency = defaultdict(int)
            for line in f:
                self.sequence_count += 1
                for program in line.split():
                    raw_program_frequency[program] += 1
                    self.program_count += 1
            print(f'Total raw sequence number is {self.sequence_count}.')
            print(f'Total raw program number is {self.program_count}')

        program_id = 0
        for program, count in raw_program_frequency.items():
            if count < self.min_count:
                self.program_count -= count
                continue
            self.program2id[program] = program_id
            self.id2program[program_id] = program
            self.program_frequency[program] = count
            program_id += 1

        self.vocal_size = len(self.program2id)
        print(f'Vocabulary size after filtering　frequency smaller than {self.min_count} is {self.vocal_size}')

    def _init_sampling_table(self):
        sample_table_size = 1e8
        pow_frequency = np.array(list(self.program_frequency.values())) ** 0.75
        pow_sum = np.sum(pow_frequency)
        ratio = pow_frequency / pow_sum
        count = np.round(ratio * sample_table_size)
        for program_id, c in enumerate(count):
            self.sample_table += [program_id] * int(c)
        self.sample_table = np.array(self.sample_table)

    def get_positive_pairs_batch(self):
        while len(self.program_pairs_catch) < self.batch_size:
            programs = self.input_file.readline()
            if programs is None or programs == '':
                self.input_file = open(self.input_file_path)
                programs = self.input_file.readline()
            program_ids = []
            for program in programs.split():
                try:
                    program_ids.append(self.program2id[program])
                except:
                    continue
            for i, u in enumerate(program_ids):
                for v in program_ids[max(i - self.window_size, 0): i + self.window_size ]:
                    if u == v:
                        continue
                    self.program_pairs_catch.append((u, v))

        batch_pairs = [self.program_pairs_catch.popleft() for _ in range(self.batch_size)]
        return batch_pairs

    def get_neg_sample_batch(self, positive_pairs_batch_length):
        neg = np.random.choice(self.sample_table, size=(positive_pairs_batch_length, self.neg_number))
        return neg

    def calculate_pair_count(self):
        """Assume all length of programs are greater than 2*window_size"""
        return self.program_count * (2 * self.window_size - 1) - self.sequence_count * (1 + self.window_size) * self.window_size




# TODO: dataset/dataloaderに書き換える
# class PrepareDataset(Dataset):
#     def __init__(input_file_path, min_count, batch_size, window_size, neg_number):
#         self.dataset = PreprocessData(input_file_path, min_count, batch_size, window_size, neg_number)
#
#     def __len__(self):
#         return self.dataset.sequence_count
#
#     def __getitem__(self, i):
#         pos_pair = self.dataset.get_positive_pairs_batch(),
#         neg_pair = self.get_neg_sample_batch(len(pos_pair))
#         return pos_pair, neg_pair
