import argparse
import os
import string

import matplotlib.pyplot as plt
import numpy as np
import tinynn as tn
import pandas as pd
from sklearn.manifold import TSNE


def prepare_dataset(args):
    url = "https://raw.githubusercontent.com/borgwang/toys/master/word2vec/data/shakespeare.csv"
    save_path = os.path.join(args.data_dir, "shakespeare.csv")
    checksum = "c947523a426577fa80294c20086f1e58"
    tn.downloader.download_url(url, save_path, checksum)

    corpus = pd.read_csv(save_path)
    raw_lines = corpus[~corpus.ActSceneLine.isna()].PlayerLine.values
    raw_lines = np.random.choice(raw_lines,
                                 size=int(len(raw_lines) * args.sample_rate),
                                 replace=False)
    print(f"corpus lines: {len(raw_lines)}")
    table = str.maketrans("", "", string.punctuation)
    vocab = set()
    lines = []
    for line in raw_lines:
        words = []
        for w in line.split():
            word = w.translate(table).lower()
            words.append(word)
            vocab.add(word)
        lines.append(words)

    vocab = list(vocab)
    vocab_size = len(vocab)
    word2idx = dict(zip(vocab, range(vocab_size)))
    idx2word = dict(zip(range(vocab_size), vocab))
    print(f"vocab size: {vocab_size}")

    idx_pairs = []
    for line in lines:
        indices = [word2idx[word] for word in line]
        for center_pos in range(len(indices)):
            for offset in range(-args.window_size, args.window_size + 1):
                context_pos = center_pos + offset
                if (context_pos < 0 or context_pos >= len(indices) or 
                        context_pos == center_pos):
                    continue
                idx_pairs.append((indices[center_pos], indices[context_pos]))
    idx_pairs = np.asarray(idx_pairs)

    train_x = np.zeros((len(idx_pairs), vocab_size), dtype=np.int8)
    train_x[np.arange(len(idx_pairs)), idx_pairs[:, 0]] = 1
    train_y = np.zeros((len(idx_pairs), vocab_size), dtype=np.int8)
    train_y[np.arange(len(idx_pairs)), idx_pairs[:, 1]] = 1
    print(f"train data size: {len(train_x)}")
    return train_x, train_y, idx2word


def main(args):
    if args.seed >= 0:
        tn.seeder.random_seed(args.seed)

    train_x, train_y, idx2word = prepare_dataset(args)
    vocab_size = len(idx2word)

    net = tn.net.Net([
        tn.layer.Dense(16),
        tn.layer.Dense(vocab_size)
    ])
    loss = tn.loss.SoftmaxCrossEntropy()
    optimizer = tn.optimizer.Adam(lr=args.lr)
    model = tn.model.Model(net=net, loss=loss, optimizer=optimizer)

    if args.model_path is not None:
        model.load(args.model_path)
        visualize(model, idx2word, args)
    else:
        iterator = tn.data_iterator.BatchIterator(batch_size=args.batch_size)
        for epoch in range(args.num_ep):
            loss_list = []
            for batch in iterator(train_x, train_y):
                pred = model.forward(batch.inputs)
                loss, grads = model.backward(pred, batch.targets)
                model.apply_grads(grads)
                loss_list.append(loss)
            print(f"Epoch: {epoch + 1} loss: {np.mean(loss_list)}")

        # save model
        if not os.path.isdir(args.model_dir):
            os.makedirs(args.model_dir)
        model_name = f"skip-gram-epoch{args.num_ep}.pkl"
        model_path = os.path.join(args.model_dir, model_name)
        model.save(model_path)
        print(f"Model saved in {model_path}")
        visualize(model, idx2word, args)


def visualize(model, idx2word, args):
    test_x = np.diag(np.ones(len(idx2word)))
    embed_layer = model.net.layers[0]
    embedding = test_x @ embed_layer.params["w"] + embed_layer.params["b"]
    labels = list(idx2word.values())

    # reduce dimension with T-SNE
    embedding_reduced = TSNE().fit_transform(embedding)

    # visualization
    plt.figure(figsize=(12, 6))
    words = ["he", "she", "man", "woman", "is", "are", "you", "i"]
    for word in words:
        i = labels.index(word)
        x, y = embedding_reduced[i]
        plt.scatter(x, y)
        plt.annotate(labels[i], (x, y))
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, "tsne-sample.jpg")
    plt.savefig(output_path)
    plt.close()
    print(f"visualization result: {output_path}")


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(curr_dir, "outputs"))
    parser.add_argument("--model_dir", type=str,
                        default=os.path.join(curr_dir, "models"))
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_ep", default=5, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=31, type=int)
    parser.add_argument("--window_size", default=2, type=int)
    parser.add_argument("--sample_rate", default=0.1, type=float)
    main(parser.parse_args())
