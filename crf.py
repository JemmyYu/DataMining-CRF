import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np


class CRF(nn.Module):

    def __init__(self, n_dice, log_likelihood):
        super(CRF, self).__init__()

        self.n_states = n_dice
        self.transition = nn.init.normal_(nn.Parameter(torch.randn(n_dice, n_dice+1)), -1, 0.1)
        self.loglikelihood = log_likelihood

    def to_scalar(self, var):
        return var.view(-1).data.tolist()[0]

    def argmax(self, vec):
        _1, idx = torch.max(vec, 1)
        return self.to_scalar(idx)

    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _data_to_likelihood(self, rolls):
        return torch.FloatTensor(self.loglikelihood[rolls])

    def _compute_likelihood_numerator(self, loglikelihoods, states):
        prev_state = self.n_states
        score = torch.Tensor([0])
        for index, state in enumerate(states):
            score += self.transition[state, prev_state] + loglikelihoods[index, state]
            prev_state = state
        return state

    def _compute_likelihood_denominator(self, loglikelihoods):
        prev_alpha = self.transition[:, self.n_states] + loglikelihoods[0].view(1, -1)
        for roll in loglikelihoods[1:]:
            alpha_t = []
            for next_state in range(self.n_states):
                feature_fuction = self.transition[next_state, :self.n_states].view(1, -1) +\
                                  roll[next_state].view(1, -1).expand(1, self.n_states)
                alpha_t_next_state = prev_alpha + feature_fuction
                alpha_t.append(self.log_sum_exp(alpha_t_next_state))
            prev_alpha = torch.stack(alpha_t).view(1, -1)
        return self.log_sum_exp(prev_alpha)

    def _viterbi_algorithm(self, loglikelihoods):
        argmaxes = []
        prev_delta = self.transition[:, self.n_states].contiguous().view(1, -1) +\
                     loglikelihoods[0].view(1, -1)
        for roll in loglikelihoods[1:]:
            local_argmaxes = []
            next_delta = []
            for next_state in range(self.n_states):
                feature_function = self.transition[next_state, :self.n_states].view(1, -1) +\
                                   roll.view(1, -1) + prev_delta
                most_likely_state = self.argmax(feature_function)
                score = feature_function[0][most_likely_state]
                next_delta.append(score)
                local_argmaxes.append(most_likely_state)
            prev_delta = torch.stack(next_delta).view(1, -1)
            argmaxes.append(local_argmaxes)

        final_state = self.argmax(prev_delta)
        final_score = prev_delta[0][final_state]
        path_list = [final_state]

        for states in reversed(argmaxes):
            final_state = states[final_state]
            path_list.append(final_state)

        return np.array(path_list), final_score

    def neg_log_likelihood(self, rolls, states):
        loglikelihoods = self._data_to_likelihood(rolls)
        states = torch.LongTensor(states)
        sequence_loglik = self._compute_likelihood_numerator(loglikelihoods, states)
        denominator = self._compute_likelihood_denominator(loglikelihoods)
        return denominator - sequence_loglik

    def forward(self, rolls):
        loglikelihoods = self._data_to_likelihood(rolls)
        return self._viterbi_algorithm(loglikelihoods)


def crf_train_loop(model, rolls, targets, n_epochs, learning_rate=0.01):
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    for epoch in range(n_epochs):
        batch_loss = []
        N = rolls.shape[0]
        model.zero_grad()
        for index, (roll, labels) in enumerate(zip(rolls, targets)):
            neg_log_likelihood = model.neg_log_likelihood(roll, labels)
            batch_loss.append(neg_log_likelihood)
            if index % 50 == 0:
                ll = torch.stack(batch_loss).mean()
                ll.backward()
                optimizer.step()
                print("Epoch {}: Batch {}/{} loss is ".format(epoch, index // 50, N // 50), ll.data.numpy())
                # print("Epoch {}: Batch {}/{} loss is {:.4f}".format(epoch, index // 50, N // 50, ll.data.numpy()[0]))
                batch_loss = []
    return model
