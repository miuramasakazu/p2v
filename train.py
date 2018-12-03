import argparse
from utility.model import Program2Vec


def main():
    parser = argparse.ArgumentParser(description='Program2id')
    parser.add_argument('--input_file_path', '-i', default='zhihu.txt',
                        help='input file consists of programs splitted by space')
    parser.add_argument('--output_file_path', '-o', default='res.txt',
                        help='output file consists of word embeddings')
    parser.add_argument('--emb_dim', '-ed', type=int, default=100,
                        help='dimensions of program vector')
    parser.add_argument('--min_count', '-mc', type=int, default=5,
                        help='minimal number of programs to be filtered')
    parser.add_argument('--batch_size', '-b', type=int, default=50,
                        help='number of programs in each mini-batch')
    parser.add_argument('--window_size', '-ws', type=int, default=2,
                        help='context size for each center program')
    parser.add_argument('--neg_number', '-nn', type=int, default=2,
                        help='negative samples number for each center program')
    parser.add_argument('--iteration', '-it', type=int, default=2,
                        help='iteration number for training')
    parser.add_argument('--learning_rate', '-lr', type=int, default=0.01,
                        help='learning rate for optimizer(sparse ADAM)')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # Set up model
    model = Program2Vec(args.input_file_path,
                        args.output_file_path,
                        args.emb_dim,
                        args.min_count,
                        args.batch_size,
                        args.window_size,
                        args.neg_number,
                        args.iteration,
                        args.learning_rate,
                        args.gpu)

    # Train model
    model.train()


if __name__ == '__main__':
    main()