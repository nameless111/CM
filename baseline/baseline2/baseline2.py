# -*- coding: utf-8 -*-
# @Time    : 4/29/19 5:36 PM
# @Author  : Melo Wang
# @Email   : melo_wyw@foxmail.com
# @File    : baseline2.py
import os
import pickle

from data_prepare.prepareData import prepareVocs


def generate_corpus(n_cluster, n_exp):
    save_dir = os.path.join("..", "..", "data", "save", "cluster" + str(n_cluster), "exp" + str(n_exp))
    option = 'story_fp_1tm'
    # vocab_file = os.path.join(save_dir, 'vocab.pickle')
    data_file = os.path.join(save_dir, option + '.pickle')
    # voc = prepareVocs(vocab_file)
    train_pairs, test_pairs = [], []
    if os.path.isfile(data_file):
        train_pairs, test_pairs = pickle.load(open(data_file, "rb"))
    for train_pair in train_pairs:
        with open(os.path.join(save_dir, "train_corpus_story-fps.story"), 'a') as f:
            f.write(' '.join(list(train_pair[0])) + '\n')
        with open(os.path.join(save_dir, "train_corpus_story-fps.fps"), 'a') as f:
            for i, fp in enumerate(train_pair[1]):
                if fp == '':
                    continue
                else:
                    f.write(' '.join(list(fp)))
                    if ''.join(train_pair[1][i+1:]) == '':
                        pass
                    else:
                        f.write(' SEP ')
            f.write('\n')

    for test_pair in test_pairs:
        with open(os.path.join(save_dir, "test_corpus_story-fps.story"), 'a') as f:
            f.write(' '.join(list(test_pair[0])) + '\n')


def read_predicted_fps(save_dir):
    fps_list = []
    for line in open(os.path.join(save_dir, 'test_corpus_story-fps.fps'), 'r'):
        fps = line.strip().replace(' ', '').split('SEP')
        fps_list.append(fps)
    return fps_list


def run():
    # n_cluster = [3]
    # n_exp = 2
    # for i in n_cluster:
    #     for j in range(n_exp):
    #         generate_corpus(i, j)
    read_predicted_fps(2, 0)


if __name__ == '__main__':
    run()
