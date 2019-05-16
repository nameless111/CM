# -*- coding: utf-8 -*-
# @File    : evalModel.py
import operator
import os
import torch
import distance
import random

from queue import PriorityQueue

import xlrd
from nltk.translate.bleu_score import sentence_bleu

from baseline.baseline1.baseline1 import rules_based_data_fp_exact
from baseline.baseline2.baseline2 import read_predicted_fps
from data_prepare.prepareData import prepareData, indexesFromSentence, indexesFromFPs, prepareData_1tm
from data_prepare.story import Story
from model import rnn
from model.attn_decoder import LuongAttnDecoderRNN
from model.decoder import DecoderRNN
from model.encoder import EncoderRNN
from model.transfer_encoder import TransferEncoder
from utils.beam_search import BeamSearchNode
from utils.data_preprocess import get_parsed_stories
from utils.matchStoryFp import get_matchest_fp, refine_storyList
from utils.voc import SOS_token, EOS_token

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def cal_metrics(predicted_fps, actual_fps, sim_threshold):
    all_actual_fps = len(actual_fps)
    all_predicted_fps = len(predicted_fps)
    if all_predicted_fps == 0:
        return 0.0, 0.0, 0.0

    TP_precision = 0
    reference = [list(predicted_fp) for predicted_fp in predicted_fps]
    total_score = 0.0
    for actual_fp in actual_fps:
        try:
            score = sentence_bleu(reference, list(actual_fp))
        except ZeroDivisionError:
            score = 0.0
        total_score += score
        print(actual_fp, score)
    # for predicted_fp in predicted_fps:
    #     sim = []
    #     for actual_fp in actual_fps:
    #         similarity = 1 - distance.nlevenshtein(actual_fp, predicted_fp, method=1)
    #         similarity = 0.0 if (len(decoded_words1) == 0 or len(candidate1) == 0) else sentence_bleu(reference1, candidate1)
    #         # if actual_fp in predicted_fp:
    #         #     similarity = 1
    #         sim.append(similarity)
    #     # print(sim)
    #
    #     if len(sim) == 0:
    #         sim = [0]
    #     if max(sim) >= sim_threshold:
    #         TP_precision += 1
    #
    # TP_recall = 0
    # for actual_fp in actual_fps:
    #     sim = []
    #     for predicted_fp in predicted_fps:
    #         similarity = 1 - distance.nlevenshtein(actual_fp, predicted_fp, method=1)
    #         sim.append(similarity)
    #     # print(sim)
    #     if len(sim) == 0:
    #         sim = [0]
    #     if max(sim) >= sim_threshold:
    #         TP_recall += 1
    #
    # # 整体的详情
    # precision = TP_precision / all_predicted_fps
    # recall = TP_recall / all_actual_fps
    # if (precision + recall) == 0:
    #     f1 = 0
    # else:
    #     f1 = 2 * precision * recall / (precision + recall)
    # return precision, recall, f1
    return total_score/len(actual_fps), 0.0, 0.0


def cal_metrics_t(predicted_fps, actual_fps, sim_threshold):
    all_actual_fps = len(actual_fps)
    all_predicted_fps = len(predicted_fps)
    if all_predicted_fps == 0:
        return 0.0, 0.0, 0.0

    TP_precision = 0
    for predicted_fp in predicted_fps:
        sim = []
        for actual_fp in actual_fps:
            # similarity = 1 - distance.nlevenshtein(actual_fp, predicted_fp, method=1)
            try:
                similarity = sentence_bleu([list(predicted_fp)], list(actual_fp))
            except ZeroDivisionError:
                similarity = 0.0
            # if actual_fp in predicted_fp:
            #     similarity = 1
            sim.append(similarity)
        # print(sim)

        if len(sim) == 0:
            sim = [0]
        if max(sim) >= sim_threshold:
            TP_precision += 1

    TP_recall = 0
    for actual_fp in actual_fps:
        sim = []
        for predicted_fp in predicted_fps:
            # similarity = 1 - distance.nlevenshtein(actual_fp, predicted_fp, method=1)
            try:
                similarity = sentence_bleu([list(predicted_fp)], list(actual_fp))
            except ZeroDivisionError:
                similarity = 0.0
            sim.append(similarity)
        # print(sim)
        if len(sim) == 0:
            sim = [0]
        if max(sim) >= sim_threshold:
            TP_recall += 1

    # 整体的详情
    precision = TP_precision / all_predicted_fps
    recall = TP_recall / all_actual_fps
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def beam_decode_pretrain(encoder, decoder, smr_input_seq, smr_input_length):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 5
    topk = 1  # how many sentence do you want to generate

    # Forward input through encoder model
    encoder_outputs, encoder_hidden = encoder(smr_input_seq, smr_input_length)
    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Start with the start of the sentence token
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token

    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # starting node -  hidden vector, previous node, word id, logp, length
    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > 20000: break

        # fetch the best node
        score, n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h

        if n.wordid.item() == EOS_token and n.prevNode != None:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = torch.topk(decoder_output, beam_width)
        nextnodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[0][new_k].view(1, -1)
            log_p = log_prob[0][new_k].item()

            node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))
            # increase qsize
        qsize += len(nextnodes) - 1

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    utterances = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance = []
        utterance.append(n.wordid)
        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid)

        utterance = utterance[::-1]
        utterances.append(utterance)

    return utterances[0]


def beam_decode(encoder, decoder, smr_input_seq, smr_input_length, des_input_seq, des_input_length, accp_input_seq,
                accp_input_length):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 5
    topk = 1  # how many sentence do you want to generate

    # Forward input through encoder model
    encoder_outputs, encoder_hidden = encoder(smr_input_seq, smr_input_length, des_input_seq, des_input_length, accp_input_seq, accp_input_length)
    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Start with the start of the sentence token
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token

    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # starting node -  hidden vector, previous node, word id, logp, length
    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > 200: break

        # fetch the best node
        score, n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h

        if n.wordid.item() == EOS_token and n.prevNode != None:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        # decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = torch.topk(decoder_output, beam_width)
        nextnodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[0][new_k].view(1, -1)
            log_p = log_prob[0][new_k].item()

            node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))
            # increase qsize
        qsize += len(nextnodes) - 1

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    utterances = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance = []
        utterance.append(n.wordid)
        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid)

        utterance = utterance[::-1]
        utterances.append(utterance)

    return utterances[0]


def evaluate_beam_pretrain(encoder, decoder, voc, test_x, test_y):
    ### Format input sentence as a batch
    # words -> indexes
    x_indexes_batch = [indexesFromSentence(voc, test_x)]
    y_indexes_batch = indexesFromFPs(voc, test_y)
    # y_indexes_batch = indexesFromSentence(voc, test_y)
    # Create lengths tensor
    lengths = torch.Tensor([len(indexes) for indexes in x_indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(x_indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens = beam_decode_pretrain(encoder, decoder, input_batch, lengths)
    tokens = tokens[1:]
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    # print(test_x)
    # print(x_indexes_batch)
    # print([voc.index2word[token] for token in x_indexes_batch[0]])
    # print(test_y)
    # print(decoded_words)
    reference = [decoded_words]
    candidate = [voc.index2word[token] for token in y_indexes_batch][:-1]
    print(test_x)
    print(''.join(decoded_words))
    print(''.join(candidate))
    score = sentence_bleu(reference, candidate)
    dis = 1 - distance.nlevenshtein(decoded_words, candidate)
    print(dis)
    print('-' * 80)
    return score, 1 if dis >= 0.6 else 0


def evaluate_beam(encoder, decoder, voc, test1_x, test2_x, test3_x, test_y):
    ### Format input sentence as a batch
    # words -> indexes
    x1_indexes_batch = [indexesFromSentence(voc, test1_x)]
    x2_indexes_batch = [indexesFromSentence(voc, test2_x)]
    x3_indexes_batch = [indexesFromSentence(voc, test3_x)]
    y_indexes_batch = indexesFromFPs(voc, test_y)
    # y_indexes_batch = indexesFromSentence(voc, test_y)
    # Create lengths tensor
    lengths1 = torch.Tensor([len(indexes) for indexes in x1_indexes_batch])
    lengths2 = torch.Tensor([len(indexes) for indexes in x2_indexes_batch])
    lengths3 = torch.Tensor([len(indexes) for indexes in x3_indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch1 = torch.LongTensor(x1_indexes_batch).transpose(0, 1)
    input_batch2 = torch.LongTensor(x2_indexes_batch).transpose(0, 1)
    input_batch3 = torch.LongTensor(x3_indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch1 = input_batch1.to(device)
    input_batch2 = input_batch2.to(device)
    input_batch3 = input_batch3.to(device)
    lengths1 = lengths1.to(device)
    lengths2 = lengths2.to(device)
    lengths3 = lengths3.to(device)
    # Decode sentence with searcher
    tokens = beam_decode(encoder, decoder, input_batch1, lengths1, input_batch2, lengths2, input_batch3, lengths3)
    # tokens = tokens[1:-1]
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    # print(test_x)
    # print(x_indexes_batch)
    # print([voc.index2word[token] for token in x_indexes_batch[0]])
    # print(test_y)
    # print(decoded_words)
    reference = [decoded_words]
    candidate = [voc.index2word[token] for token in y_indexes_batch][:-1]
    print(''.join(decoded_words))
    print(''.join(candidate))
    score = sentence_bleu(reference, candidate)
    return score


class GreedySearchDecoder(torch.nn.Module):
    def __init__(self, transfer_encoder, fp_decoder):
        super(GreedySearchDecoder, self).__init__()
        self.transfer_encoder = transfer_encoder
        # self.smr_encoder = smr_encoder
        # self.des_encoder = des_encoder
        # self.accp_encoder = accp_encoder
        self.fp_decoder = fp_decoder

    def forward(self, smr_input_seq, smr_input_length, des_input_seq, des_input_length, accp_input_seq,
                accp_input_length, max_length):
        # Forward pass through encoder
        story_summary_encoder_outputs, story_encoder_hidden = self.transfer_encoder(smr_input_seq, smr_input_length,
                                                                               des_input_seq, des_input_length,
                                                                               accp_input_seq, accp_input_length)

        # smr_encoder_outputs, smr_encoder_hidden = self.smr_encoder(smr_input_seq, smr_input_length)
        # des_encoder_outputs, des_encoder_hidden = self.des_encoder(des_input_seq, des_input_length)
        # accp_encoder_outputs, accp_encoder_hidden = self.accp_encoder(accp_input_seq, accp_input_length)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = story_encoder_hidden[:self.fp_decoder.n_layers]

        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.fp_decoder(decoder_input, decoder_hidden, story_summary_encoder_outputs)

            # print(encoder_decoder_outputs.size())
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
            if decoder_input == EOS_token:
                break
        # Return collections of word tokens and scores
        return all_tokens, all_scores


class GreedySearchDecoder_pretrain(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder_pretrain, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
            if decoder_input == EOS_token:
                break
        # Return collections of word tokens and scores
        return all_tokens, all_scores


class GreedySearchDecoder_1tm(torch.nn.Module):
    def __init__(self, encoder, decoder, k):
        super(GreedySearchDecoder_1tm, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        #self.decoder1 = decoder[0]
        #self.decoder2 = decoder[1]
        #self.decoder3 = decoder[2]
        #self.decoder4 = decoder[3]
        self.k = k

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder[0].n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens, all_scores = [], []
        for i in range(self.k):
            all_tokens.append(torch.zeros([0], device=device, dtype=torch.long))
            all_scores.append(torch.zeros([0], device=device))

        #all_tokens1 = torch.zeros([0], device=device, dtype=torch.long)
        #all_scores1 = torch.zeros([0], device=device)

        #all_tokens2 = torch.zeros([0], device=device, dtype=torch.long)
        #all_scores2 = torch.zeros([0], device=device)

        #all_tokens3 = torch.zeros([0], device=device, dtype=torch.long)
        #all_scores3 = torch.zeros([0], device=device)

        #all_tokens4 = torch.zeros([0], device=device, dtype=torch.long)
        #all_scores4 = torch.zeros([0], device=device)
        for i in range(self.k):
            # Iteratively decode one word token at a time
            for _ in range(max_length):
                # Forward pass through decoder
                decoder_output, decoder_hidden = self.decoder[i](decoder_input, decoder_hidden, encoder_outputs)
                # Obtain most likely word token and its softmax score
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
                # Record token and score
                all_tokens[i] = torch.cat((all_tokens[i], decoder_input), dim=0)
                all_scores[i] = torch.cat((all_scores[i], decoder_scores), dim=0)
                # Prepare current token to be next decoder input (add a dimension)
                decoder_input = torch.unsqueeze(decoder_input, 0)
                if decoder_input == EOS_token:
                    break

        #for _ in range(max_length):
            # Forward pass through decoder
            #decoder_output, decoder_hidden = self.decoder2(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            #decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            #all_tokens2 = torch.cat((all_tokens2, decoder_input), dim=0)
            #all_scores2 = torch.cat((all_scores2, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            #decoder_input = torch.unsqueeze(decoder_input, 0)
            #if decoder_input == EOS_token:
            #    break

        #for _ in range(max_length):
            # Forward pass through decoder
            #decoder_output, decoder_hidden = self.decoder3(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            #decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            #all_tokens3 = torch.cat((all_tokens3, decoder_input), dim=0)
            #all_scores3 = torch.cat((all_scores3, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            #decoder_input = torch.unsqueeze(decoder_input, 0)
            #if decoder_input == EOS_token:
            #    break

        #for _ in range(max_length):
            # Forward pass through decoder
            #decoder_output, decoder_hidden = self.decoder4(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            #decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            #all_tokens4 = torch.cat((all_tokens4, decoder_input), dim=0)
            #all_scores4 = torch.cat((all_scores4, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            #decoder_input = torch.unsqueeze(decoder_input, 0)
            #if decoder_input == EOS_token:
            #    break
        # Return collections of word tokens and scores
        # return (all_tokens1, all_tokens2, all_tokens3), (all_scores1, all_scores2, all_scores3)
        return all_tokens, all_scores


def evaluate(searcher, voc, test1_x, test2_x, test3_x, test_y):
    ### Format input sentence as a batch
    # words -> indexes
    x1_indexes_batch = [indexesFromSentence(voc, test1_x)]
    x2_indexes_batch = [indexesFromSentence(voc, test2_x)]
    x3_indexes_batch = [indexesFromSentence(voc, test3_x)]
    y_indexes_batch = indexesFromFPs(voc, test_y)
    # y_indexes_batch = indexesFromSentence(voc, test_y)
    # Create lengths tensor
    lengths1 = torch.Tensor([len(indexes) for indexes in x1_indexes_batch])
    lengths2 = torch.Tensor([len(indexes) for indexes in x2_indexes_batch])
    lengths3 = torch.Tensor([len(indexes) for indexes in x3_indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch1 = torch.LongTensor(x1_indexes_batch).transpose(0, 1)
    input_batch2 = torch.LongTensor(x2_indexes_batch).transpose(0, 1)
    input_batch3 = torch.LongTensor(x3_indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch1 = input_batch1.to(device)
    input_batch2 = input_batch2.to(device)
    input_batch3 = input_batch3.to(device)
    lengths1 = lengths1.to(device)
    lengths2 = lengths2.to(device)
    lengths3 = lengths3.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch1, lengths1, input_batch2, lengths2, input_batch3, lengths3, 50)
    tokens = tokens[:-1]
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    # print(test_x)
    # print(x_indexes_batch)
    # print([voc.index2word[token] for token in x_indexes_batch[0]])
    # print(test_y)
    # print(decoded_words)
    reference = [decoded_words]
    candidate = [voc.index2word[token] for token in y_indexes_batch][:-1]
    print(test1_x)
    print(''.join(decoded_words))
    print(''.join(candidate))
    # print(test_y)
    score = sentence_bleu(reference, candidate)
    dis = 1 - distance.nlevenshtein(decoded_words, candidate)
    print(dis)
    print('-' * 80)
    return score, 1 if dis >= 0.6 else 0


def evaluate_pretain(searcher, voc, test_x, test_y):
    ### Format input sentence as a batch
    # words -> indexes
    x_indexes_batch = [indexesFromSentence(voc, test_x)]
    y_indexes_batch = indexesFromFPs(voc, test_y)
    # y_indexes_batch = indexesFromSentence(voc, test_y)
    # y_indexes_batch = indexesFromSentence(voc, test_y)
    # Create lengths tensor
    lengths = torch.Tensor([len(indexes) for indexes in x_indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(x_indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, 100)
    tokens = tokens[:-1]
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    # print(test_x)
    # print(x_indexes_batch)
    # print([voc.index2word[token] for token in x_indexes_batch[0]])
    # print(test_y)
    # print(decoded_words)
    reference = [decoded_words]
    candidate = [voc.index2word[token] for token in y_indexes_batch][:-1]
    print(test_x)
    print(''.join(decoded_words))
    print(''.join(candidate))
    # print(test_y)
    score = sentence_bleu(reference, candidate)
    dis = 1 - distance.nlevenshtein(decoded_words, candidate)
    print(dis)
    print('-' * 80)
    return score, 1 if dis>=0.6 else 0


def evaluate_1tm(searcher, voc, test_x, test_y, k):
    ### Format input sentence as a batch
    # words -> indexes
    x_indexes_batch = [indexesFromSentence(voc, test_x)]
    y_indexes_batch = [indexesFromSentence(voc, y) for y in test_y]
    # y_indexes_batch1 = indexesFromSentence(voc, test_y[0])
    # y_indexes_batch2 = indexesFromSentence(voc, test_y[1])
    # y_indexes_batch3 = indexesFromSentence(voc, test_y[2])

    # Create lengths tensor
    lengths = torch.Tensor([len(indexes) for indexes in x_indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(x_indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, 100)
    tokens_list = [token[:-1] for token in tokens]
    # tokens1 = tokens[0][:-1]
    # tokens2 = tokens[1][:-1]
    # tokens3 = tokens[2][:-1]

    # indexes -> words
    decoded_words = [[voc.index2word[token.item()] for token in tokens] for tokens in tokens_list]
    # decoded_words1 = [voc.index2word[token.item()] for token in tokens1]
    # decoded_words2 = [voc.index2word[token.item()] for token in tokens2]
    # decoded_words3 = [voc.index2word[token.item()] for token in tokens3]

    # reference = [decoded_word for decoded_word in decoded_words]
    # reference1 = [decoded_words1]
    # reference2 = [decoded_words2]
    # reference3 = [decoded_words3]
    # candidate = [[voc.index2word[token] for token in y_index_batch][:-1] for y_index_batch in y_indexes_batch]
    # candidate1 = [voc.index2word[token] for token in y_indexes_batch1][:-1]
    # candidate2 = [voc.index2word[token] for token in y_indexes_batch2][:-1]
    # candidate3 = [voc.index2word[token] for token in y_indexes_batch3][:-1]

    predicted_fps = []
    actual_fps = []
    sim_threshold = 0.5

    for i in range(k):
        if test_y[i] != '':
            actual_fps.append(test_y[i])
        if ''.join(decoded_words[i]) != '':
            predicted_fps.append(''.join(decoded_words[i]))

    # print(test_x)
    # print(actual_fps)
    # print([''.join(decoded_word) for decoded_word in decoded_words])
    #print(predicted_fps)

    p, r, _ = cal_metrics_t(predicted_fps, actual_fps, sim_threshold)
    # bleu, _ , _ = cal_metrics(predicted_fps, actual_fps, sim_threshold)

    # score1 = 0.0 if (len(decoded_words1) == 0 or len(candidate1) == 0) else sentence_bleu(reference1, candidate1)
    # dis1 = 1 - distance.nlevenshtein(decoded_words1, candidate1)
    # score2 = 0.0 if (len(decoded_words2) == 0 or len(candidate2) == 0) else sentence_bleu(reference2, candidate2)
    # dis2 = 1 - distance.nlevenshtein(decoded_words2, candidate2)
    # score3 = 0.0 if (len(decoded_words3) == 0 or len(candidate3) == 0) else sentence_bleu(reference3, candidate3)
    # dis3 = 1 - distance.nlevenshtein(decoded_words3, candidate3)
    # dis = (dis1 + dis2 + dis3)/3
    # score = (score1 + score2 + score3)/3
    # print(dis)
    # print(p, r)
    # print('-' * 80)
    # return score, 1 if dis>=0.6 else 0
    return p, r


def evaluate_SAFE(test_x, test_y):

    predicted_fps = []
    actual_fps = []
    sim_threshold = 0.5

    sentences = test_x.split()
    for sentence in sentences:
        predicted_fps.extend(rules_based_data_fp_exact(sentence))

    for fp in test_y:
        if fp != '':
            actual_fps.append(fp)
    # print(test_x)
    # print(predicted_fps)
    # print(actual_fps)
    p, r, _ = cal_metrics_t(predicted_fps, actual_fps, sim_threshold)
    # print("-" * 80)
    return p, r


def evaluate_MOSES(test_x, test_y, fps):

    predicted_fps = fps
    actual_fps = []
    sim_threshold = 0.5

    for fp in test_y:
        if fp != '':
            actual_fps.append(fp)
    # print(test_x)
    # print(predicted_fps)
    # print(actual_fps)
    p, r, _ = cal_metrics_t(predicted_fps, actual_fps, sim_threshold)
    # print("-" * 80)
    return p, r


def run():
    hidden_size = 100
    encoder_n_layers = 2
    decoder_n_layers = 2

    save_dir = os.path.join("..", "data", "save")
    option = 'story_fp'
    model_name = option + '_model_Attn'
    attn_model = 'general'
    checkpoint_iter = 120000
    loadFilename = os.path.join(save_dir, model_name,
                                '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                                '{}_checkpoint.tar'.format(checkpoint_iter))
    # Load model if a loadFilename is provided
    if not os.path.isfile(loadFilename):
        print('Cannot find model file: ' + loadFilename)
    else:
        print('Loading model file: ' + loadFilename)
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        transfer_encoder_sd = checkpoint['transfer_en']
        # smr_encoder_sd = checkpoint['smr_en']
        # des_encoder_sd = checkpoint['des_en']
        # accp_encoder_sd = checkpoint['accp_en']
        fp_decoder_sd = checkpoint['fp_de']
        # encoder_optimizer_sd = checkpoint['en_opt']
        # decoder_optimizer_sd = checkpoint['de_opt']
        # embedding_sd = checkpoint['embedding']

        # prepare vocabulary and test data
        voc, _, test_pairs = prepareData(os.path.join(save_dir, 'vocab.pickle'), os.path.join(save_dir, option + '.pickle'),
                                 option)
        print("Read {!s} testing pairs".format(len(test_pairs)))

        print('Building encoder and decoder ...')

        # Initialize word embeddings
        embedding = torch.nn.Embedding(voc.num_words, hidden_size)
        # embedding.load_state_dict(embedding_sd)

        # Initialize encoder & decoder models
        # smr_encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers)
        smr_encoder = rnn.RNNModel('GRU', voc.num_words, 100, 100, 2)
        des_encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers)
        accp_encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers)
        fp_decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers)
        # fp_decoder = DecoderRNN(embedding, hidden_size, voc.num_words, decoder_n_layers)
        fp_decoder.load_state_dict(fp_decoder_sd)

        # transfer_encoder = TransferEncoder(smr_encoder, des_encoder, accp_encoder, fp_decoder.hidden_size)
        transfer_encoder = TransferEncoder(smr_encoder, smr_encoder, smr_encoder, fp_decoder.hidden_size)
        transfer_encoder.load_state_dict(transfer_encoder_sd)
        # smr_encoder.load_state_dict(smr_encoder_sd)
        # des_encoder.load_state_dict(des_encoder_sd)
        # accp_encoder.load_state_dict(accp_encoder_sd)

        # for name, para in encoder_decoder.named_parameters():
        #     print(name, ':', para)

        # encoder.load_state_dict(encoder_sd)
        # decoder.load_state_dict(decoder_sd)

        # Use appropriate device
        transfer_encoder = transfer_encoder.to(device)
        # smr_encoder = smr_encoder.to(device)
        # des_encoder = des_encoder.to(device)
        # accp_encoder = accp_encoder.to(device)
        fp_decoder = fp_decoder.to(device)

        # Set dropout layers to eval mode
        transfer_encoder.eval()
        # smr_encoder.train()
        # des_encoder.train()
        # accp_encoder.train()
        fp_decoder.eval()

        # Initialize search module
        searcher = GreedySearchDecoder(transfer_encoder, fp_decoder)

        print('Evaluating {!s} test pairs ...'.format(len(test_pairs)))
        score = 0
        dis = 0
        num = 0
        for test_pair in test_pairs:
            try:
                s, d = evaluate(searcher, voc, test_pair[0], test_pair[1], test_pair[2], test_pair[3])
                # score += evaluate_beam(transfer_encoder, fp_decoder, voc, test_pair[0], test_pair[1], test_pair[2], test_pair[3])
                score += s
                dis += d
                num += 1
            except ZeroDivisionError:
                continue
        print('Total BLEU score: ' + str(score / num))
        print('Total levenshtein distance: ' + str(dis / num))


def run_pretrain():
    hidden_size = 100
    encoder_n_layers = 2
    decoder_n_layers = 2

    save_dir = os.path.join("..", "data", "save")
    option = 'story_fp'
    model_name = option + '_nd_model_Attn'
    attn_model = 'general'
    checkpoint_iter = 40000
    loadFilename = os.path.join(save_dir, model_name,
                                '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                                '{}_checkpoint.tar'.format(checkpoint_iter))
    # Load model if a loadFilename is provided
    if not os.path.isfile(loadFilename):
        print('Cannot find model file: ' + loadFilename)
    else:
        print('Loading model file: ' + loadFilename)
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        # encoder_optimizer_sd = checkpoint['en_opt']
        # decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']

        # prepare vocabulary and test data
        voc, _, test_pairs = prepareData(os.path.join(save_dir, 'vocab.pickle'),
                                                   os.path.join(save_dir, option + '.pickle'),
                                                   option)
        # test_pairs = read_stories_from_xls(os.path.join('..', 'data', 'compare.xlsx'))

        # test_pairs = [random.choice(test_pairs) for _ in range(10)]
        print("Read {!s} testing pairs".format(len(test_pairs)))

        print('Building encoder and decoder ...')

        # Initialize word embeddings
        embedding = torch.nn.Embedding(voc.num_words, hidden_size)
        embedding.load_state_dict(embedding_sd)

        # Initialize encoder & decoder models
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers)
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers)
        # decoder = DecoderRNN(embedding, hidden_size, voc.num_words, decoder_n_layers)
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

        # Use appropriate device
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()

        # Initialize search module
        searcher = GreedySearchDecoder_pretrain(encoder, decoder)

        print('Evaluating {!s} test pairs ...'.format(len(test_pairs)))
        score = 0
        dis = 0
        num = 0
        for test_pair in test_pairs:
            try:
                s, d = evaluate_pretain(searcher, voc, test_pair[0], test_pair[3])
                # s, d = evaluate_beam_pretrain(encoder, decoder, voc, test_pair[0], test_pair[3])
                score += s
                dis += d
                num += 1
            except ZeroDivisionError:
                continue
        print('Total BLEU score: ' + str(score / num))
        print('Total levenshtein distance: ' + str(dis / num))


def run_1tm(n_cluster, n_exp, ed, hs):
    embedding_dim = ed
    hidden_size = hs
    encoder_n_layers = 2
    decoder_n_layers = 2

    save_dir = os.path.join("..", "data", "save", "cluster" + str(n_cluster), "exp" + str(n_exp))
    option = 'story_fp_1tm'
    model_name = option + '_model_Attn_embedding_dim' + str(ed)
    attn_model = 'general'
    checkpoint_iter = 160000
    k = n_cluster
    loadFilename = os.path.join(save_dir, model_name,
                                '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                                '{}_checkpoint.tar'.format(checkpoint_iter))
    # Load model if a loadFilename is provided
    if not os.path.isfile(loadFilename):
        print('Cannot find model file: ' + loadFilename)
    else:
        print('Loading model file: ' + loadFilename)
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

        encoder_sd = checkpoint['en']
        decoder_sd = [checkpoint['de' + str(i+1)] for i in range(k)]
        #decoder_sd1 = checkpoint['de1']
        #decoder_sd2 = checkpoint['de2']
        #decoder_sd3 = checkpoint['de3']
        #decoder_sd4 = checkpoint['de4']

        # encoder_optimizer_sd = checkpoint['en_opt']
        # decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']

        # prepare vocabulary and test data
        voc, _, test_pairs = prepareData_1tm(os.path.join("..", "data", "save", 'vocab.pickle'),
                                                   os.path.join(save_dir, option + '.pickle'))
        # test_pairs = read_stories_from_xls(os.path.join('..', 'data', 'compare.xlsx'))

        # test_pairs = [random.choice(test_pairs) for _ in range(10)]
        print("Read {!s} testing pairs".format(len(test_pairs)))

        print('Building encoder and decoder ...')

        # Initialize word embeddings
        embedding = torch.nn.Embedding(voc.num_words, embedding_dim)
        embedding.load_state_dict(embedding_sd)

        # Initialize encoder & decoder models
        encoder = EncoderRNN(embedding_dim, hidden_size, embedding, encoder_n_layers)
        decoder = [LuongAttnDecoderRNN(attn_model, embedding, embedding_dim, hidden_size, voc.num_words, decoder_n_layers) for _ in range(k)]
        #decoder1 = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers)
        #decoder2 = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers)
        #decoder3 = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers)
        #decoder4 = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers)

        # decoder = DecoderRNN(embedding, hidden_size, voc.num_words, decoder_n_layers)
        encoder.load_state_dict(encoder_sd)
        [decoder[i].load_state_dict(decoder_sd[i]) for i in range(k)]
        #decoder1.load_state_dict(decoder_sd1)
        #decoder2.load_state_dict(decoder_sd2)
        #decoder3.load_state_dict(decoder_sd3)
        #decoder4.load_state_dict(decoder_sd4)

        # Use appropriate device
        encoder = encoder.to(device)
        [decoder[i].to(device) for i in range(k)]
        #decoder1 = decoder1.to(device)
        #decoder2 = decoder2.to(device)
        #decoder3 = decoder3.to(device)
        #decoder4 = decoder4.to(device)

        # Set dropout layers to eval mode
        encoder.eval()
        [decoder[i].eval() for i in range(k)]
        #decoder1.eval()
        #decoder2.eval()
        #decoder3.eval()
        #decoder4.eval()

        # Initialize search module
        searcher = GreedySearchDecoder_1tm(encoder, decoder, k)

        print('Evaluating {!s} test pairs ...'.format(len(test_pairs)))
        OMEG_score = 0
        OMEG_dis = 0
        SAFE_score = 0
        SAFE_dis = 0
        MOSES_score = 0
        MOSES_dis = 0
        num = 0
        # fps_list = read_predicted_fps(save_dir)
        for i, test_pair in enumerate(test_pairs):
            num += 1
            # try:
            OMEG_s, OMEG_d = evaluate_1tm(searcher, voc, test_pair[0], test_pair[1], k)
            # SAFE_s, SAFE_d = evaluate_SAFE(test_pair[0], test_pair[1])
            # MOSES_s, MOSES_d = evaluate_MOSES(test_pair[0], test_pair[1], fps_list[i])
            # s, d = evaluate_beam_pretrain(encoder, decoder, voc, test_pair[0], test_pair[3])
            OMEG_score += OMEG_s
            OMEG_dis += OMEG_d
            # SAFE_score += SAFE_s
            # SAFE_dis += SAFE_d
            # MOSES_score += MOSES_s
            # MOSES_dis += MOSES_d
            # except ZeroDivisionError:
            #     continue
        print('\n\n' + '-' * 40 + "cluster" + str(n_cluster), "exp" + str(n_exp) + '_embedding' + str(ed) + '_hidden' + str(hs) + '-' * 40)
        print('OMEG Total Precision: ' + str(OMEG_score / num))
        print('OMEG Total Recall: ' + str(OMEG_dis / num))

        # print('SAFE Total Precision: ' + str(SAFE_score / num))
        # print('SAFE Total Recall: ' + str(SAFE_dis / num))
        #
        # print('MOSES Total Precision: ' + str(MOSES_score / num))
        # print('MOSES Total Recall: ' + str(MOSES_dis / num))
        print('-' * 40 + "cluster" + str(n_cluster), "exp" + str(n_exp) + '_embedding' + str(ed) + '_hidden' + str(hs) + '-' * 40 + '\n\n')
        #
        # with open(os.path.join("..", "data", "result.csv"), 'a') as f:
        #     f.write('cluster' + str(n_cluster) + '_exp' + str(n_exp) + ',')
        #     f.write(str(OMEG_score / num) + ',')
        #     f.write(str(OMEG_dis / num) + ',')
        #     f.write(str(SAFE_score / num) + ',')
        #     f.write(str(SAFE_dis / num) + ',')
        #     f.write(str(MOSES_score / num) + ',')
        #     f.write(str(MOSES_dis / num) + ',')
        #     f.write('\n')
        # with open(os.path.join("..", "data", "result_embedding_hidden.csv"), 'a') as f:
        #     f.write('cluster' + str(n_cluster) + '_embedding' + str(ed) + '_hidden' + str(hs) + ',')
        #     f.write(str(OMEG_score / num) + ',')
        #     f.write(str(OMEG_dis / num) + ',')
        #     f.write('\n')

def read_stories_from_xls(file_path):
    file = xlrd.open_workbook(file_path)
    table = file.sheets()[1]
    nrows = table.nrows
    storyList = []
    for i in range(0, nrows):
        story_id = table.cell(i, 0).value
        summary = table.cell(i, 2).value
        description = table.cell(i, 3).value
        acceptance = table.cell(i, 4).value
        fps = table.cell(i, 7).value[1:-1].split(', ')
        r_str = table.cell(i, 12).value
        recall = table.cell(i, 14).value

        # if recall >= 0.6:
        #     continue

        r_strs = r_str.split('    ')
        r = []
        for r_str in r_strs:
            r_list = r_str[1:-1].split()
            for r_l in r_list:
                r.append(r_l[:r_str.index('(') - 1])
        story = Story(story_id, summary, description, acceptance)
        story.fps.extend(fps)
        story.r.extend(r)
        storyList.append(story)
    pairs = []
    storyList = get_parsed_stories(storyList)
    for story in storyList:
        pairs.append([story.summary, '', '', [get_matchest_fp(story)]])

    return pairs


if __name__ == '__main__':
    n_cluster = 3
    n_exp = 'params'
    # for i in n_cluster:
    #     for j in range(n_exp):
    #         run_1tm(i, j)
    embedding_dim = [100]
    hidden_size = [300, 500]
    for ed in embedding_dim:
        for hs in hidden_size:
            run_1tm(n_cluster, n_exp, ed, hs)
