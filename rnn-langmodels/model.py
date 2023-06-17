import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        #self.embedding = None
        #self.rnn = None
        #self.linear = None
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size,
                                      padding_idx=dataset.pad_id)
        self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)


    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        # This is a placeholder, you may remove it.
        #logits = torch.randn(
         #   indices.shape[0], indices.shape[1], self.vocab_size,
          #  device=indices.device
        #)

        embeds = self.embedding(indices)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed_embeds)
        outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
        logits = self.linear(outputs)


        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        device = next(self.parameters()).device
        pref_indices = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        pref_indices = torch.tensor(pref_indices).unsqueeze(0).to(device)

        embeds = self.embedding(pref_indices)
        output, hidden = self.rnn(embeds)
        logits = self.linear(output) / temp

        new_tokens = Categorical(logits=logits[:, -1:]).sample()
        pref_indices = torch.cat([pref_indices, new_tokens], dim=1)

        while pref_indices.shape[1] < self.max_length:
            if new_tokens.item() == self.dataset.eos_id:
                break

            embeds = self.embedding(new_tokens)
            output, hidden = self.rnn(embeds, hidden)
            logits = self.linear(output) / temp
            
            new_tokens = Categorical(logits=logits[:, -1:]).sample()
            pref_indices = torch.cat([pref_indices, new_tokens], dim=1)

        # decode result to a string
        # This is a placeholder, you may remove it.
        # generated = prefix + ', а потом купил мужик шляпу, а она ему как раз.'
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        return self.dataset.ids2text(pref_indices.squeeze())

        #return generated
