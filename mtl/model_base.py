# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     sentiment_classifier
   Description :
   Author :       xmz
   date：          19-4-10
-------------------------------------------------
"""
from typing import Dict, Optional

import numpy as np
from allennlp.modules.attention import DotProductAttention, LinearAttention
from allennlp.modules.encoder_base import _EncoderBase
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder
from allennlp.nn.util import get_final_encoder_states
from allennlp.training.util import get_batch_size
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, Seq2SeqEncoder, InputVariationalDropout
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Auc
from torch import nn
from torch.autograd import Function
from torch.nn import Dropout

from mtl.attention import Attention
from mtl.logger import logger
from train import TASKS_NAME


class CNNEncoder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 shared_encoder: Seq2VecEncoder,
                 private_encoder: Seq2VecEncoder,
                 input_dropout: float = 0.0,
                 regularizer: RegularizerApplicator = None) -> None:
        super(CNNEncoder, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._shared_encoder = shared_encoder
        self._private_encoder = private_encoder
        # self._U = nn.Linear()
        self._attention = DotProductAttention()
        self._input_dropout = Dropout(input_dropout)

    def forward(self,
                task_index: torch.IntTensor,
                tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        embedded_text_input = self._text_field_embedder(tokens)
        tokens_mask = util.get_text_field_mask(tokens)
        batch_size = get_batch_size(tokens)
        embedded_text_input = self._input_dropout(embedded_text_input)

        shared_encoded_text = self._shared_encoder(embedded_text_input, tokens_mask)
        private_encoded_text = self._private_encoder(embedded_text_input, tokens_mask)
        output_dict = dict()
        output_dict["share_embedding"] = shared_encoded_text
        output_dict["private_embedding"] = private_encoded_text

        embedded_text = torch.cat([shared_encoded_text, private_encoded_text], -1)
        output_dict["embedded_text"] = embedded_text
        return output_dict

    def get_output_dim(self):
        return self._shared_encoder.get_output_dim() + self._private_encoder.get_output_dim()


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return self.lambd * grad_output.neg()


def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


class Discriminator(nn.Module):
    def __init__(self, source_size, target_size):
        super(Discriminator, self).__init__()
        self._classifier = nn.Linear(int(source_size), target_size)

    def forward(self, representation, epoch_trained=None, reverse=torch.tensor(False), lambd=1.0):
        if reverse.all():
            # TODO increase lambda from 0
            representation = grad_reverse(representation, lambd)
        return self._classifier(representation)


class RNNEncoder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 shared_encoder: Seq2SeqEncoder,
                 private_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.0,
                 regularizer: RegularizerApplicator = None) -> None:
        super(RNNEncoder, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._shared_encoder = shared_encoder
        self._private_encoder = private_encoder
        self._seq2vec = CnnEncoder(embedding_dim=self._shared_encoder.get_output_dim(),
                                   num_filters=int(shared_encoder.get_output_dim() / 4))
        self._input_dropout = Dropout(input_dropout)
        self._s_query = nn.Parameter(torch.randn(1, 100))
        self._s_att = Attention(heads=3, attn_size=300, query_size=100, value_size=shared_encoder.get_output_dim(),
                                key_size=shared_encoder.get_output_dim(), dropout=0.1)
        self._p_query = nn.Parameter(torch.randn(1, 100))
        self._p_att = Attention(heads=3, attn_size=300, query_size=100, value_size=private_encoder.get_output_dim(),
                                key_size=private_encoder.get_output_dim(), dropout=0.1)

    @overrides
    def forward(self,
                task_index: torch.IntTensor,
                tokens: Dict[str, torch.LongTensor],
                epoch_trained: torch.IntTensor,
                valid_discriminator: Discriminator,
                reverse: torch.ByteTensor,
                for_training: torch.ByteTensor,
                text_id: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        embedded_text_input = self._text_field_embedder(tokens)
        tokens_mask = util.get_text_field_mask(tokens)
        output_dict = dict()
        embedded_text_input = self._input_dropout(embedded_text_input)

        shared_encoded_text = self._shared_encoder(embedded_text_input, tokens_mask)
        shared_encoded_text, s_weights = self._s_att(self._s_query, shared_encoded_text, shared_encoded_text)

        # shared_encoded_text = self._seq2vec(shared_encoded_text, tokens_mask)
        # shared_encoded_text = get_final_encoder_states(shared_encoded_text, tokens_mask, bidirectional=True)
        output_dict["share_embedding"] = shared_encoded_text.squeeze()

        private_encoded_text = self._private_encoder(embedded_text_input, tokens_mask)
        private_encoded_text, p_weights = self._p_att(self._s_query, private_encoded_text, private_encoded_text)
        print(p_weights.shape)
        # private_encoded_text = self._seq2vec(private_encoded_text, tokens_mask)
        # private_encoded_text = get_final_encoder_states(private_encoded_text, tokens_mask, bidirectional=True)
        output_dict["private_embedding"] = private_encoded_text.squeeze()

        if not for_training:
            with open("attn.txt", "a") as f:
                f.write(f"Task: {TASKS_NAME[task_index.cpu().item()]}\nLine ID: ")
                f.write(" ".join(list(map(str, text_id.cpu().detach().numpy()))))
                f.write("\nShared Encoder Att: ")
                f.write(" ".join(list(map(str, s_weights.squeeze().cpu().detach().numpy()))))
                f.write("\nPrivate Encoder Att: ")
                f.write(" ".join(list(map(str, p_weights.squeeze().cpu().detach().numpy()))))
                f.write("\n\n\n")
        embedded_text = torch.cat([shared_encoded_text, private_encoded_text], -1).squeeze(1)
        output_dict["embedded_text"] = embedded_text
        return output_dict

    def get_output_dim(self):
        return self._shared_encoder.get_output_dim() + self._private_encoder.get_output_dim()


class SentimentClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 share_encoder: _EncoderBase,
                 private_encoder: _EncoderBase,
                 s_domain_discriminator: Discriminator,
                 p_domain_discriminator: Discriminator,
                 valid_discriminator: Discriminator,
                 dropout: float = 0.0,
                 input_dropout: float = 0.0,
                 label_smoothing: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SentimentClassifier, self).__init__(vocab, regularizer)

        if isinstance(share_encoder, Seq2VecEncoder) and isinstance(private_encoder, Seq2VecEncoder):
            self._encoder = CNNEncoder(vocab, text_field_embedder, share_encoder, private_encoder,
                                       input_dropout=input_dropout)
        else:
            self._encoder = RNNEncoder(vocab, text_field_embedder, share_encoder, private_encoder,
                                       input_dropout=input_dropout)

        self._num_classes = self.vocab.get_vocab_size("label")
        self._sentiment_discriminator = Discriminator(self._encoder.get_output_dim(), self._num_classes)
        self._s_domain_discriminator = s_domain_discriminator
        self._p_domain_discriminator = p_domain_discriminator
        self._valid_discriminator = valid_discriminator
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        self._label_smoothing = label_smoothing

        self.metrics = {
            "prf": F1Measure(positive_label=1),
            "auc": Auc(positive_label=1),
            "acc": CategoricalAccuracy(),
        }

        self._loss = torch.nn.CrossEntropyLoss()
        self._domain_loss = torch.nn.CrossEntropyLoss()
        # TODO torch.nn.BCELoss
        self._valid_loss = torch.nn.BCEWithLogitsLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                task_index: torch.IntTensor,
                reverse: torch.ByteTensor,
                epoch_trained: torch.IntTensor,
                for_training: torch.ByteTensor,
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                text_id: torch.IntTensor = None) -> Dict[str, torch.Tensor]:

        embeddeds = self._encoder(task_index, tokens, epoch_trained, self._valid_discriminator, reverse, for_training,
                                  text_id)
        batch_size = get_batch_size(embeddeds["embedded_text"])

        sentiment_logits = self._sentiment_discriminator(embeddeds["embedded_text"])

        p_domain_logits = self._p_domain_discriminator(embeddeds["private_embedding"])

        # TODO set reverse = true
        s_domain_logits = self._s_domain_discriminator(embeddeds["share_embedding"], reverse=reverse)

        logits = [sentiment_logits, p_domain_logits, s_domain_logits]

        # domain_logits = self._domain_discriminator(embedded_text)
        output_dict = {'logits': sentiment_logits}
        if label is not None:
            loss = self._loss(sentiment_logits, label)
            # task_index = task_index.unsqueeze(0)
            task_index = task_index.expand(batch_size)
            # targets = [label, label, label, task_index, task_index]
            # print(p_domain_logits.shape, task_index, task_index.shape)
            p_domain_loss = self._domain_loss(p_domain_logits, task_index)
            s_domain_loss = self._domain_loss(s_domain_logits, task_index)
            # logger.info("Share domain logits standard variation is {}",
            #             torch.mean(torch.std(F.softmax(s_domain_logits), dim=-1)))
            output_dict["tokens"] = tokens
            output_dict['stm_loss'] = loss
            output_dict['p_d_loss'] = p_domain_loss
            output_dict['s_d_loss'] = s_domain_loss
            # TODO add share domain logits std loss
            output_dict['loss'] = loss + 0.06 * p_domain_loss + 0.04 * s_domain_loss

            for (metric_name, metric) in zip(self.metrics.keys(), self.metrics.values()):
                if "auc" in metric_name:
                    metric(self.decode(output_dict)["label"], label)
                    continue
                metric(sentiment_logits, label)
        print("for training", for_training)
        if not for_training:
            with open("class_probabilities.txt", "a", encoding="utf8") as f:
                f.write(f"Task: {TASKS_NAME[task_index[0].detach()]}\nLine ID: ")
                f.write(" ".join(list(map(str, text_id.cpu().detach().numpy()))))
                f.write("\nProb: ")
                f.write(" ".join(list(map(str, F.softmax(sentiment_logits, dim=-1).cpu().detach().numpy()))))
                f.write("\nLabel: " + " ".join(list(map(str, label.cpu().detach().numpy()))) + "\n")
                f.write("\n\n\n")
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities
        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        # labels = [self.vocab.get_token_from_index(x, namespace="label")
        #           for x in argmax_indices]
        output_dict['label'] = torch.IntTensor(argmax_indices)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_dict = dict()
        for metric_name, metric in self.metrics.items():
            if "prf" in metric_name:
                p, r, f = metric.get_metric(reset)
                metrics_dict["precision"] = p
                metrics_dict["recall"] = r
                metrics_dict["f1"] = f
                continue
            metrics_dict[metric_name] = metric.get_metric(reset)
        return metrics_dict
