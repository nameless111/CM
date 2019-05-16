# -*- coding: utf-8 -*-
# @File    : trainModel.py
import os
import torch
import random

from torch import optim
from data_prepare.prepareData import batch2TrainData_1tm, prepareData_1tm
from model.attn_decoder import LuongAttnDecoderRNN
from model.decoder import DecoderRNN
from model.encoder import EncoderRNN
from utils.voc import SOS_token


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


######################################################################
# Define Training Procedure
# -------------------------
#
# Masked loss
# ~~~~~~~~~~~
#
# Since we are dealing with batches of padded sequences, we cannot simply
# consider all elements of the tensor when calculating loss. We define
# ``maskNLLLoss`` to calculate our loss based on our decoder’s output
# tensor, the target tensor, and a binary mask tensor describing the
# padding of the target tensor. This loss function calculates the average
# negative log likelihood of the elements that correspond to a *1* in the
# mask tensor.
#

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


######################################################################
# Single training iteration
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``train`` function contains the algorithm for a single training
# iteration (a single batch of inputs).
#
# We will use a couple of clever tricks to aid in convergence:
#
# -  The first trick is using **teacher forcing**. This means that at some
#    probability, set by ``teacher_forcing_ratio``, we use the current
#    target word as the decoder’s next input rather than using the
#    decoder’s current guess. This technique acts as training wheels for
#    the decoder, aiding in more efficient training. However, teacher
#    forcing can lead to model instability during inference, as the
#    decoder may not have a sufficient chance to truly craft its own
#    output sequences during training. Thus, we must be mindful of how we
#    are setting the ``teacher_forcing_ratio``, and not be fooled by fast
#    convergence.
#
# -  The second trick that we implement is **gradient clipping**. This is
#    a commonly used technique for countering the “exploding gradient”
#    problem. In essence, by clipping or thresholding gradients to a
#    maximum value, we prevent the gradients from growing exponentially
#    and either overflow (NaN), or overshoot steep cliffs in the cost
#    function.
#
#
# **Sequence of Operations:**
#
#    1) Forward pass entire input batch through encoder.
#    2) Initialize decoder inputs as SOS_token, and hidden state as the encoder's final hidden state.
#    3) Forward input batch sequence through decoder one time step at a time.
#    4) If teacher forcing: set next decoder input as the current target; else: set next decoder input as current decoder output.
#    5) Calculate and accumulate loss.
#    6) Perform backpropagation.
#    7) Clip gradients.
#    8) Update encoder and decoder model parameters.
#
#
# .. Note ::
#
#   PyTorch’s RNN modules (``RNN``, ``LSTM``, ``GRU``) can be used like any
#   other non-recurrent layers by simply passing them the entire input
#   sequence (or batch of sequences). We use the ``GRU`` layer like this in
#   the ``encoder``. The reality is that under the hood, there is an
#   iterative process looping over each time step calculating hidden states.
#   Alternatively, you ran run these modules one time-step at a time. In
#   this case, we manually loop over the sequences during the training
#   process like we must do for the ``decoder`` model. As long as you
#   maintain the correct conceptual model of these modules, implementing
#   sequential models can be very straightforward.
#
#
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder1, decoder2, decoder3, embedding,
          encoder_optimizer, decoder_optimizer1, decoder_optimizer2, decoder_optimizer3, batch_size, teacher_forcing_ratio, clip):
#def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder1, decoder2,
#          embedding,
#          encoder_optimizer, decoder_optimizer1, decoder_optimizer2, batch_size,
#          teacher_forcing_ratio, clip):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer1.zero_grad()
    decoder_optimizer2.zero_grad()
    decoder_optimizer3.zero_grad()
    #decoder_optimizer4.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable1 = target_variable[0].to(device)
    target_variable2 = target_variable[1].to(device)
    target_variable3 = target_variable[2].to(device)
    #target_variable4 = target_variable[3].to(device)

    mask1 = mask[0].to(device)
    mask2 = mask[1].to(device)
    mask3 = mask[2].to(device)
    #mask4 = mask[3].to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder1.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len[0]):
            decoder_output, decoder_hidden = decoder1(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable1[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable1[t], mask1[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len[0]):
            decoder_output, decoder_hidden = decoder1(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable1[t], mask1[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len[1]):
            decoder_output, decoder_hidden = decoder2(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable2[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable2[t], mask2[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len[1]):
            decoder_output, decoder_hidden = decoder2(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable2[t], mask2[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len[2]):
            decoder_output, decoder_hidden = decoder3(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable3[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable3[t], mask3[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len[2]):
            decoder_output, decoder_hidden = decoder3(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable3[t], mask3[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    ## Forward batch of sequences through decoder one time step at a time
    #if use_teacher_forcing:
    #    for t in range(max_target_len[3]):
    #        decoder_output, decoder_hidden = decoder4(
    #            decoder_input, decoder_hidden, encoder_outputs
    #        )
    #        # Teacher forcing: next input is current target
    #        decoder_input = target_variable4[t].view(1, -1)
    #        # Calculate and accumulate loss
    #        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable4[t], mask4[t])
    #        loss += mask_loss
    #        print_losses.append(mask_loss.item() * nTotal)
    #        n_totals += nTotal
    #else:
    #    for t in range(max_target_len[3]):
    #        decoder_output, decoder_hidden = decoder4(
    #            decoder_input, decoder_hidden, encoder_outputs
    #        )
    #        # No teacher forcing: next input is decoder's own current output
    #        _, topi = decoder_output.topk(1)
    #        decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
    #        decoder_input = decoder_input.to(device)
    #        # Calculate and accumulate loss
    #        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable4[t], mask4[t])
    #        loss += mask_loss
    #        print_losses.append(mask_loss.item() * nTotal)
    #        n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder1.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder2.parameters(), clip)
    # _ = torch.nn.utils.clip_grad_norm_(decoder3.parameters(), clip)
    #_ = torch.nn.utils.clip_grad_norm_(decoder4.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer1.step()
    decoder_optimizer2.step()
    # decoder_optimizer3.step()
    #decoder_optimizer4.step()

    return sum(print_losses) / n_totals


######################################################################
# Training iterations
# ~~~~~~~~~~~~~~~~~~~
#
# It is finally time to tie the full training procedure together with the
# data. The ``trainIters`` function is responsible for running
# ``n_iterations`` of training given the passed models, optimizers, data,
# etc. This function is quite self explanatory, as we have done the heavy
# lifting with the ``train`` function.
#
# One thing to note is that when we save our model, we save a tarball
# containing the encoder and decoder state_dicts (parameters), the
# optimizers’ state_dicts, the loss, the iteration, etc. Saving the model
# in this way will give us the ultimate flexibility with the checkpoint.
# After loading a checkpoint, we will be able to use the model parameters
# to run inference, or we can continue training right where we left off.
#

def trainIters(model_name, voc, pairs, encoder, decoder1, decoder2, decoder3, encoder_optimizer, decoder_optimizer1, decoder_optimizer2, decoder_optimizer3, embedding, encoder_n_layers, decoder_n_layers, hidden_size, save_dir, n_iteration, batch_size, print_every, save_every, teacher_forcing_ratio, clip, loadFilename, checkpoint_iter, checkpoint_iteration, option):
#def trainIters(model_name, voc, pairs, encoder, decoder1, decoder2, encoder_optimizer, decoder_optimizer1,
#                   decoder_optimizer2, embedding, encoder_n_layers, decoder_n_layers, hidden_size,
#                   save_dir, n_iteration, batch_size, print_every, save_every, teacher_forcing_ratio, clip,
#                   loadFilename, checkpoint_iter, checkpoint_iteration, option):

    # Load batches for each iteration
    training_batches = [batch2TrainData_1tm(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration - checkpoint_iter)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint_iteration + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - start_iteration]

        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        # print(input_variable[:, 0].size())
        # print(''.join([voc.index2word[input.item()] for input in input_variable[:, 0]]))
        # print(target_variable[:, 0])
        # print(''.join([voc.index2word[input.item()] for input in target_variable[:, 0]]))

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder1, decoder2, decoder3, embedding, encoder_optimizer, decoder_optimizer1, decoder_optimizer2, decoder_optimizer3, batch_size, teacher_forcing_ratio, clip)
        #loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
        #             decoder1, decoder2, embedding, encoder_optimizer, decoder_optimizer1, decoder_optimizer2,
        #             batch_size, teacher_forcing_ratio, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de1': decoder1.state_dict(),
                'de2': decoder2.state_dict(),
                'de3': decoder3.state_dict(),
                #'de4': decoder4.state_dict(),

                'en_opt': encoder_optimizer.state_dict(),
                'de_opt1': decoder_optimizer1.state_dict(),
                'de_opt2': decoder_optimizer2.state_dict(),
                'de_opt3': decoder_optimizer3.state_dict(),
                #'de_opt4': decoder_optimizer4.state_dict(),

                'loss': loss,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def run(j, i, k, ed, hs):
    save_dir = os.path.join("..", "data", "save", "cluster" + str(k), "exp" + str(j))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    option = 'story_fp_1tm'
    # if option not in ['story_summary', 'story_description', 'story_acceptance', 'fp', 'story_fp', 'story_fp_1tm']:
    #     raise ValueError(option, "is not an appropriate corpus type.")
    voc, pairs, _ = prepareData_1tm(os.path.join("..", "data", "save", 'vocab.pickle'), os.path.join(save_dir, option + '.pickle'))
    print("Read {!s} training pairs".format(len(pairs)))

    # Configure models
    model_name = option + '_model_Attn_embedding_dim' + str(ed)
    # attn_model = 'dot'
    attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = hs
    embedding_dim = ed
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
    checkpoint_iter = i * 40000
    n_iteration = checkpoint_iter + 40000
    if i == 0:
        loadFilename = None
    else:
        loadFilename = os.path.join(save_dir, model_name,
                                    '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                                    '{}_checkpoint.tar'.format(checkpoint_iter))


    checkpoint_iteration = 0
    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd1 = checkpoint['de1']
        decoder_sd2 = checkpoint['de2']
        decoder_sd3 = checkpoint['de3']
        #decoder_sd4 = checkpoint['de4']

        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd1 = checkpoint['de_opt1']
        decoder_optimizer_sd2 = checkpoint['de_opt2']
        decoder_optimizer_sd3 = checkpoint['de_opt3']
        #decoder_optimizer_sd4 = checkpoint['de_opt4']

        embedding_sd = checkpoint['embedding']

        checkpoint_iteration = checkpoint['iteration']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = torch.nn.Embedding(voc.num_words, embedding_dim)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(embedding_dim, hidden_size, embedding, encoder_n_layers, dropout)
    decoder1 = LuongAttnDecoderRNN(attn_model, embedding, embedding_dim, hidden_size, voc.num_words, decoder_n_layers)
    decoder2 = LuongAttnDecoderRNN(attn_model, embedding, embedding_dim, hidden_size, voc.num_words, decoder_n_layers)
    decoder3 = LuongAttnDecoderRNN(attn_model, embedding, embedding_dim, hidden_size, voc.num_words, decoder_n_layers)
    #decoder4 = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers)

    # decoder = DecoderRNN(embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder1.load_state_dict(decoder_sd1)
        decoder2.load_state_dict(decoder_sd2)
        decoder3.load_state_dict(decoder_sd3)
        #decoder4.load_state_dict(decoder_sd4)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder1 = decoder1.to(device)
    decoder2 = decoder2.to(device)
    decoder3 = decoder3.to(device)
    #decoder4 = decoder4.to(device)

    print('Models built and ready to go!')

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    print_every = 100
    save_every = 40000

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder1.train()
    decoder2.train()
    decoder3.train()
    #decoder4.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer1 = optim.Adam(decoder1.parameters(), lr=learning_rate * decoder_learning_ratio)
    decoder_optimizer2 = optim.Adam(decoder2.parameters(), lr=learning_rate * decoder_learning_ratio)
    decoder_optimizer3 = optim.Adam(decoder3.parameters(), lr=learning_rate * decoder_learning_ratio)
    #decoder_optimizer4 = optim.Adam(decoder4.parameters(), lr=learning_rate * decoder_learning_ratio)

    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer1.load_state_dict(decoder_optimizer_sd1)
        decoder_optimizer2.load_state_dict(decoder_optimizer_sd2)
        decoder_optimizer3.load_state_dict(decoder_optimizer_sd3)
        #decoder_optimizer4.load_state_dict(decoder_optimizer_sd4)

    # Run training iterations
    print("Starting Training!")
    trainIters(model_name, voc, pairs, encoder, decoder1, decoder2, decoder3, encoder_optimizer, decoder_optimizer1, decoder_optimizer2, decoder_optimizer3,
               embedding, encoder_n_layers, decoder_n_layers,hidden_size, save_dir, n_iteration, batch_size,
               print_every, save_every, teacher_forcing_ratio, clip, loadFilename, checkpoint_iter, checkpoint_iteration, option)
    #trainIters(model_name, voc, pairs, encoder, decoder1, decoder2, encoder_optimizer, decoder_optimizer1, decoder_optimizer2,
    #           embedding, encoder_n_layers, decoder_n_layers,hidden_size, save_dir, n_iteration, batch_size,
    #           print_every, save_every, teacher_forcing_ratio, clip, loadFilename, checkpoint_iter, checkpoint_iteration, option)


if __name__ == '__main__':
    n_cluster = 3
    embedding_dim = [100]
    hidden_size = [300, 500]
    # for j in range(2, 5):
    #     for i in range(4):
    #         run(j, i, k)
    for ed in embedding_dim:
        for hs in hidden_size:
            for i in range(4):
                run('params', i, n_cluster, ed, hs)
