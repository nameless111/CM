# -*- coding: utf-8 -*-
# @Time    : 4/12/19 5:03 PM
# @Author  : Melo Wang
# @Email   : melo_wyw@foxmail.com
# @File    : one_to_many_utils.py
import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from data_prepare.story import Story
from model import rnn
from utils.db_dao.story_dao import read_fps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import jieba.posseg as pseg

HIDDEN_SIZE = 100
N_LAYERS = 2
RNN_TYPE = 'GRU'
EMBEDDING_SIZE = 100
DROPOUT = 0.5
TIE_WEIGHT = False

save_dir = os.path.join("..", "data", "save")
option = 'story_summary_lm'
model_name = option + '_model'
checkpoint_iter = 5120000
loadFilename = os.path.join(save_dir, model_name,
                            '{}_{}'.format(N_LAYERS, HIDDEN_SIZE),
                            '{}_checkpoint.tar'.format(checkpoint_iter))


def getEmbedding(voc):
    # Load model if a loadFilename is provided
    if not os.path.isfile(loadFilename):
        print('Cannot find model file: ' + loadFilename)
    else:
        print('Loading embedding file: ' + loadFilename)
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        #checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        rnn_sd = checkpoint['rnn']

        # Initialize encoder & decoder models
        rnn_model = rnn.RNNModel(RNN_TYPE, voc.num_words, EMBEDDING_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT, TIE_WEIGHT)
        rnn_model.load_state_dict(rnn_sd)
        return rnn_model.encoder


def get_str_embedding(fp, voc, embedding):
    word_vec = np.array([embedding(torch.LongTensor([voc.word2index[char]])).detach().numpy() for char in list(fp)]).squeeze().reshape(-1, 100)
    return np.mean(word_vec, axis=0)


# /**
#      * 将功能点进行划分，分为名词和动词
#      * @param fp
#      * @return "verbs"-动词   "nouns"-名词
#      * @exception FPAnalysisException 功能点解析结果为空、解析后动词、解析后名词为空都会抛出词异常
#      */
#     public static Pair<String, String> fpNameSplit(String fp) throws  FPAnalysisException{
#         fp = fp.trim();
#         //判断数据功能
#         if(fp.endsWith("文件") || fp.endsWith("数据")
#                 || fp.endsWith("信息")
#                 || fp.endsWith("表")
#                 || (fp.endsWith("档") && !fp.endsWith("下档"))
#                 || fp.endsWith("记录")
#                 || fp.endsWith("数据层")
#                 || fp.endsWith("模板")
#                 || fp.endsWith("接口")
#                 || fp.endsWith("日志")
#                 || fp.endsWith("存储")
#                 || fp.endsWith("报文")
#                 || fp.endsWith("系统"))  return new Pair<String, String>(fp, null);
#
#         /*
#         功能点预处理步骤
#          */
#         //1.利用上一步骤进行预处理
#         fp = PreProcessor.filterFPs(fp);
#         if(fp == null) throw new FPAnalysisException("功能点预处理后为空");
#         //2.将预处理按照- 和 _进行部分划分,只保留最后一个部分
#         String[] parts = fp.split("-|_");
#         if(parts.length == 2 && (parts[1].equals("前台") || parts[1].equals("后台"))) fp = parts[0];
#         else fp = parts[parts.length-1];
#
#
#         String verbs = "";
#         String nouns = "";
#
#         //将功能点进行分词
#         CoNLLSentence coNLLSentence = SegmentHelper.parseDependency(fp);
#         CoNLLWord[] words = coNLLSentence.getWordArray();
#
#         //如果解析结果为空，则返回空指针异常
#         if(words == null || words.length == 0) throw new FPAnalysisException("功能点解析结果为空");
#
#         //获取开头词和结尾词
#         CoNLLWord start = words[0];
#         CoNLLWord end = words[words.length - 1];
#
#         //识别：动词+"功能"
#         if (end.LEMMA.equals("功能") && words.length >= 2 && words[words.length - 2].POSTAG.startsWith("v")) {
#             String verb = words[words.length - 2].LEMMA;
#             String noun = fp.substring(0, fp.length() - verb.length() - 2);
#             verbs = verb;
#             nouns = noun;
#         }else {
#             //识别：首尾是动词
#             boolean startVerb = start.POSTAG.startsWith("v");
#             boolean endVerb = end.POSTAG.startsWith("v");
#             if (endVerb) {
#                 String verb = end.LEMMA;
#                 String noun = fp.substring(0, fp.length() - verb.length());
#                 verbs = verb;
#                 nouns = noun;
#             } else if (startVerb) {
#                 String verb = start.LEMMA;
#                 String noun = fp.substring(verb.length(), fp.length());
#                 verbs = verb;
#                 nouns = noun;
#             }
#         }
#
#         //如果功能点解析错误---缺少名词或者动词 则抛出空指针异常
#         if(verbs.length() == 0) verbs = null;
#         if(nouns.length() == 0) nouns = null;
#
#         Pair results = new Pair(nouns, verbs);
#
#         return results;
#     }


def get_fp_verb(fp):
    pairs = pseg.lcut(fp)
    # print(pairs)
    beg_word, beg_pos = pairs[0]
    end_word, end_pos = pairs[-1]
    if end_pos.startswith('v'):
        res = end_word
    elif beg_pos.startswith('v'):
        res = beg_word
    else:
        verbs = []
        for pair in pairs:
            word, pos = pair
            if pos.startswith('v'):
                verbs.append(word)
        if len(verbs) == 0:
            res = end_word
        else:
            res = verbs[-1]
    # print(res)
    return res


def cluster(voc, embedding):
    kmeans_file = os.path.join(save_dir, 'kmeans_model.pickle')
    if os.path.isfile(kmeans_file):
        kmeans_model, X = pickle.load(open(kmeans_file, "rb"))
    else:
        fps = read_fps()
        X = np.array([get_str_embedding(fp, voc, embedding) for fp in fps])
        kmeans_model = KMeans(n_clusters=2).fit(X)
        pickle.dump([kmeans_model, X], open(kmeans_file, "wb"))
    return kmeans_model, X


def verb_cluster(voc, embedding):
    kmeans_file = os.path.join(save_dir, 'verb_kmeans_model.pickle')
    if os.path.isfile(kmeans_file):
        kmeans_model, X = pickle.load(open(kmeans_file, "rb"))
    else:
        verb_file = os.path.join('..', 'data', 'verbs.txt')
        verbs, nums = read_verbs(verb_file)
        X = np.array([get_str_embedding(fp, voc, embedding) for fp in verbs])
        kmeans_model = KMeans(n_clusters=2).fit(X)
        pickle.dump([kmeans_model, X], open(kmeans_file, "wb"))
    return kmeans_model, X


def read_verbs(verb_file):
    verbs = []
    nums = []
    with open(verb_file, "r") as f:
        for line in f:
            line = line[:-1]
            str = line.split(',')
            if int(str[1]) < 500:
                break
            verbs.append(str[0])
            nums.append(str[1])
    return verbs, nums


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b


    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def place_rule(voc, embedding, fps):
    kmeans_model, _ = cluster(voc, embedding)
    cluster_centers = kmeans_model.cluster_centers_
    fp_list = []
    for cluster_center in cluster_centers:
        max_cos = 0.0
        max_fp = ''
        for fp in fps:
            if fp in fp_list:
                continue
            cos = cos_sim(get_str_embedding(fp, voc, embedding), cluster_center)
            if cos > max_cos:
                max_cos = cos
                max_fp = fp
        fp_list.append(max_fp)
    return fp_list


def cluster_rule(stories, cluster_model, voc, embedding):
    new_stories = []
    # embeddings = []
    # ls = []
    for story in stories:
        fp_embedding = np.array([get_str_embedding(fp, voc, embedding) for fp in story.fps])
        labels = cluster_model.predict(fp_embedding)
        # embeddings.extend(fp_embedding)
        # ls.extend(labels)
        new_stories.extend(generate_stories(story, labels, cluster_model.n_clusters))
    # pca(embeddings, ls)
    return new_stories


def verb_cluster_rule(stories, cluster_model, voc, embedding):
    new_stories = []
    # embeddings = []
    # ls = []
    for story in stories:
        verb_embedding = np.array([get_str_embedding(get_fp_verb(fp), voc, embedding) for fp in story.fps])
        labels = cluster_model.predict(verb_embedding)
        # embeddings.extend(verb_embedding)
        # ls.extend(labels)
        new_stories.extend(generate_stories(story, labels, cluster_model.n_clusters))
    # print(len(embeddings))
    # pca(embeddings, ls)
    return new_stories


def man_verb_cluster_rule(stories, cluster_model):
    new_stories = []
    # embeddings = []
    # ls = []
    for story in stories:
        verb = np.array([get_fp_verb(fp) for fp in story.fps])
        labels = cluster_model.predict(verb)
        # embeddings.extend(verb_embedding)
        # ls.extend(labels)
        new_stories.extend(generate_stories(story, labels, cluster_model.n_clusters))
    # print(len(embeddings))
    # pca(embeddings, ls)
    return new_stories


def generate_stories(story, labels, k):
    fps_list = []
    stories = []
    for i in range(k):
        fps_list.append([''])
    for i, label in enumerate(labels):
        if fps_list[label][0] == '':
            fps_list[label][0] = story.fps[i]
        else:
            fps_list[label].append(story.fps[i])
    for fp_c0 in fps_list[0]:
        for fp_c1 in fps_list[1]:
            for fp_c2 in fps_list[2]:
            #     for fp_c3 in fps_list[3]:
                s = Story(story.story_id, story.summary, story.description, story.acceptance)
                s.fps = [fp_c0, fp_c1, fp_c2]
                stories.append(s)
    # print(fps_list, len(stories))
    # print([len(fps) for fps in fps_list], len(stories))
    # for story in stories:
    #     print(story.fps)
    return stories


def pca(x, y):
    pca = PCA(n_components=2)
    reduced_x = pca.fit_transform(x)
    red_x, red_y, red_z = [], [], []
    blue_x, blue_y, blue_z = [], [], []
    green_x, green_y, green_z = [], [], []
    c_x, c_y, c_z = [], [], []
    k_x, k_y, k_z = [], [], []
    m_x, m_y, m_z = [], [], []
    w_x, w_y, w_z = [], [], []
    y_x, y_y, y_z = [], [], []

    for i in range(len(reduced_x)):
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
            # red_z.append(reduced_x[i][2])
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
            # blue_z.append(reduced_x[i][2])
        elif y[i] == 2:
            c_x.append(reduced_x[i][0])
            c_y.append(reduced_x[i][1])
            # c_z.append(reduced_x[i][2])
        elif y[i] == 3:
            k_x.append(reduced_x[i][0])
            k_y.append(reduced_x[i][1])
            # k_z.append(reduced_x[i][2])
        elif y[i] == 4:
            m_x.append(reduced_x[i][0])
            m_y.append(reduced_x[i][1])
            # m_z.append(reduced_x[i][2])
        elif y[i] == 5:
            w_x.append(reduced_x[i][0])
            w_y.append(reduced_x[i][1])
            # w_z.append(reduced_x[i][2])
        elif y[i] == 6:
            y_x.append(reduced_x[i][0])
            y_y.append(reduced_x[i][1])
            # y_z.append(reduced_x[i][2])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
            # green_z.append(reduced_x[i][2])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    plt.scatter(red_x, red_y, c='r', marker='o')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='d')
    plt.scatter(c_x, c_y, c='c', marker='x')
    plt.scatter(k_x, k_y, c='k', marker='v')
    plt.scatter(m_x, m_y, c='m', marker='<')
    plt.scatter(w_x, w_y, c='w', marker='>')
    plt.scatter(y_x, y_y, c='y', marker='.')

    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    plt.show()


def run():
    print(get_fp_verb('客户查看查询信息'))


if __name__ == '__main__':
    run()
