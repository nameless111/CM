# -*- coding: utf-8 -*-
# @File    : prepareData.py
import itertools
import random
import os
import pickle
import torch

from utils.db_dao.story_dao import read_stories, read_stories_without_fps, read_fps, read_stories_from_xls, \
    read_data_from_xls, read_new_stories_from_xls
from utils.matchStoryFp import get_matchest_fp, refine_storyList, refine_storyList_withmorefps
from utils.voc import Voc, EOS_token, PAD_token, SEP_token
from utils.verb_cluster import VerbCluster
from model.one_to_many_utils import place_rule, getEmbedding, pca, cluster, cluster_rule, verb_cluster, \
    verb_cluster_rule, man_verb_cluster_rule


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
# def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    # return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
# def filterPairs(pairs):
#     return [pair for pair in pairs if filterPair(pair)]


def prepareVocs(vocab_file):
    print("Start preparing vocabulary ...")
    if os.path.isfile(vocab_file):
        voc = pickle.load(open(vocab_file, "rb"))
    else:
        voc = Voc("Story-FPs")
        stories = read_stories_without_fps()
        fps = read_fps()
        story_texts = [story.summary + story.description + story.acceptance for story in stories]
        # print(fps[0])
        # print(story_texts[0])
        print("Read {!s} stories and {!s} function points".format(len(stories), len(fps)))
        print("Counting words...")
        for story_text in story_texts:
            voc.addSentence(story_text)
        for fp in fps:
            voc.addSentence(fp)
        pickle.dump(voc, open(vocab_file, "wb"))
    print("Counted words:", voc.num_words)
    return voc


# Using the functions defined above, return a populated voc object and pairs list
def prepareData(vocab_file, data_file, option):
    voc = prepareVocs(vocab_file)
    print("Start preparing data ...")
    if os.path.isfile(data_file):
        if option != 'story_fp':
            pairs = pickle.load(open(data_file, "rb"))
            return voc, pairs
        else:
            train_pairs, test_pairs = pickle.load(open(data_file, "rb"))
            return voc, train_pairs, test_pairs
    else:
        if option == 'story_summary':
            storyList = read_stories_without_fps()
            pairs = [[story.summary, story.summary] for story in storyList]
            pickle.dump(pairs, open(data_file, "wb"))
            return voc, pairs
        elif option == 'story_description':
            storyList = read_stories_without_fps()
            pairs = [[story.description, story.description] for story in storyList]
            pickle.dump(pairs, open(data_file, "wb"))
            return voc, pairs
        elif option == 'story_acceptance':
            storyList = read_stories_without_fps()
            pairs = [[story.acceptance, story.acceptance] for story in storyList]
            pickle.dump(pairs, open(data_file, "wb"))
            return voc, pairs
        elif option == 'fp':
            fps = read_fps()
            pairs = [[fp, fp] for fp in fps]
            pickle.dump(pairs, open(data_file, "wb"))
            return voc, pairs
        else:
            storyList = read_data_from_xls(os.path.join("..", "data", "story-fp_new.xls"))
            # storyList = refine_storyList(read_stories())
            # stories = read_stories_from_xls(os.path.join("..", "data", "new_story-fp_0.7.xls"))

            # for story in stories:
            #     is_exist = False
            #     for s in storyList:
            #         if story.story_id == s.story_id:
            #             is_exist = True
            #             break
            #     if not is_exist:
            #         storyList.append(story)

            train_storyList, test_storyList = train_test_split(storyList, 0.1, shuffle=True)
            train_pairs, test_pairs = [], []
            # for story in train_storyList:
            #     for fp in itertools.permutations(story.fps, 3):
            #         train_pairs.append([story.summary, '', '', list(fp)])

            # for story in test_storyList:
            #     test_pairs.append([story.summary, '', '', story.fps])
            for story in train_storyList:
                # train_pairs.append([story.summary, story.description, story.acceptance, [getMatchestFP(story)]])
                train_pairs.append([story.summary, '', '', [get_matchest_fp(story)]])

            for story in test_storyList:
                # test_pairs.append([story.summary, story.description, story.acceptance, [getMatchestFP(story)]])
                test_pairs.append([story.summary, '', '', [get_matchest_fp(story)]])

            # random.shuffle(train_pairs)
            # pairs = [[story.summary, story.description, story.acceptance, story.fps[:3]] for story in storyList]
            pickle.dump([train_pairs, test_pairs], open(data_file, "wb"))
            return voc, train_pairs, test_pairs


def prepareData_1tm(vocab_file, data_file):
    voc = prepareVocs(vocab_file)
    embedding = getEmbedding(voc)
    # cluster_model, _ = verb_cluster(voc, embedding)
    cluster_model, _ = cluster(voc, embedding)
    # verb_cluster = VerbCluster()
    # storyList = read_stories()
    # print(len(storyList))
    # stories = read_stories_from_xls(os.path.join("..", "data", "new_story-fp_0.7.xls"))
    # for story in stories:
    #     is_exist = False
    #     for s in storyList:
    #         if story.story_id == s.story_id:
    #             is_exist = True
    #             break
    #     if not is_exist:
    #         storyList.append(story)
    # print(len(storyList))
    # storyList = refine_storyList_withmorefps(storyList)
    # print(len(storyList))
    # storyList = cluster_rule(storyList, cluster_model, voc, embedding)
    # print(len(storyList))
    # pca(_, cluster_model.labels_)
    print("Start preparing data ...")
    if os.path.isfile(data_file):
        train_pairs, test_pairs = pickle.load(open(data_file, "rb"))
        # l = len(train_pairs) + len(test_pairs)
        # offset = int(l * 0.1)
        # train_pairs = train_pairs[offset:]
        return voc, train_pairs, test_pairs
    else:
        storyList = read_stories()
        print(len(storyList))
        stories = read_new_stories_from_xls(os.path.join("..", "data", "story-fp_new_data.xls"))
        for story in stories:
            is_exist = False
            for s in storyList:
                if story.story_id == s.story_id:
                    is_exist = True
                    break
            if not is_exist:
                storyList.append(story)
        print(len(storyList))
        storyList = refine_storyList(storyList)
        print(len(storyList))
        # storyList = man_verb_cluster_rule(storyList, verb_cluster)
        storyList = cluster_rule(storyList, cluster_model, voc, embedding)
        print(len(storyList))
        train_storyList, test_storyList = train_test_split(storyList, 0.1, shuffle=True)

        #pairs = []
        #for story in storyList:
        #    pairs.append([story.summary, story.fps])
        #    if len(story.description) > 10 and len(story.description) < 200:
        #        pairs.append([story.description, story.fps])
        #    if len(story.acceptance) > 10 and len(story.acceptance) < 200:
        #        pairs.append([story.acceptance, story.fps])
        #print(len(pairs))
        train_pairs, test_pairs = [], []
        for story in train_storyList:
            train_pairs.append([story.summary, story.fps])
        for story in test_storyList:
            test_pairs.append([story.summary, story.fps])
        #train_pairs, test_pairs = train_test_split(pairs, 0.1, shuffle=True)
        pickle.dump([train_pairs, test_pairs], open(data_file, "wb"))
        return voc, train_pairs, test_pairs


def indexesFromFPs(voc, fps):
    res = []
    for fp in fps:
        res += [voc.word2index[word] for word in list(fp)]
        res += [SEP_token]
    res[-1] = EOS_token
    return res


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in list(sentence)] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.Tensor([len(indexes) for indexes in indexes_batch])

    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc, option):
    if option != 'story_fp':
        indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    else:
        indexes_batch = [indexesFromFPs(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar_1tm(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch, option):
    pair_batch.sort(key=lambda x: len(list(x[0])), reverse=True)
    if option != 'story_fp':
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = inputVar(input_batch, voc)
        output, mask, max_target_len = outputVar(output_batch, voc, option)
        return inp, lengths, output, mask, max_target_len
    else:
        input1_batch, input2_batch, input3_batch, output_batch = [], [], [], []
        for pair in pair_batch:
            input1_batch.append(pair[0])
            input2_batch.append(pair[1])
            input3_batch.append(pair[2])
            output_batch.append(pair[3])
        inp1, lengths1 = inputVar(input1_batch, voc)
        inp2, lengths2 = inputVar(input2_batch, voc)
        inp3, lengths3 = inputVar(input3_batch, voc)
        output, mask, max_target_len = outputVar(output_batch, voc, option)
        return (inp1, inp2, inp3), (lengths1, lengths2, lengths3), output, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData_1tm(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(list(x[0])), reverse=True)

    input_batch, output_batch1, output_batch2, output_batch3, output_batch4 = [], [], [], [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch1.append(pair[1][0])
        output_batch2.append(pair[1][1])
        output_batch3.append(pair[1][2])
        #output_batch4.append(pair[1][3])

    inp, lengths = inputVar(input_batch, voc)
    output1, mask1, max_target_len1 = outputVar_1tm(output_batch1, voc)
    output2, mask2, max_target_len2 = outputVar_1tm(output_batch2, voc)
    output3, mask3, max_target_len3 = outputVar_1tm(output_batch3, voc)
    #output4, mask4, max_target_len4 = outputVar_1tm(output_batch4, voc)

    return inp, lengths, (output1, output2, output3), (mask1, mask2, mask3), (max_target_len1, max_target_len2, max_target_len3)
    #return inp, lengths, (output1, output2), (mask1, mask2), (max_target_len1, max_target_len2)


def train_test_split(data_pairs, test_ratio, shuffle=False):
    n_total = len(data_pairs)
    offset = int(n_total * test_ratio)
    if n_total == 0 or offset < 1:
        return [], data_pairs
    if shuffle:
        random.shuffle(data_pairs)
    test = data_pairs[:offset]
    train = data_pairs[offset:]
    return train, test


if __name__ == '__main__':
    # Example for validation
    small_batch_size = 4
    save_dir = os.path.join("..", "data", "save")
    option = 'story_fp_1tm'
    voc, train_pairs, _ = prepareData_1tm(os.path.join(save_dir, 'vocab.pickle'), os.path.join(save_dir, option + '.pickle'))
    batches = batch2TrainData_1tm(voc, [random.choice(train_pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches
    print(train_pairs[0])
    # print(voc.word2index['作'])
    # print(len(train_pairs))
    # print(len(test_pairs))
    # for test_pair in train_pairs[:36]:
    #     print(test_pair)
    # print("input_variable:", input_variable)
    # print("lengths:", lengths)
    # print("target_variable:", target_variable)
    # print("mask:", mask)
    # print("max_target_len:", max_target_len)
