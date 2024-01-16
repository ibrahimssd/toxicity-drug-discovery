from transformers import PreTrainedTokenizer
from typing import List, Optional, Union
import os
import collections
import codecs
import torch
import pickle
from SmilesPE.tokenizer import *
from SmilesPE.pretokenizer import atomwise_tokenizer
import chardet
from bertviz_repo.bertviz import head_view
from captum.attr import visualization as viz

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    encoding = detect_encoding(vocab_file) # "utf-8"
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding=encoding) as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

def save_model(model,model_path_name):
    # pickle.dump(arab_morfessor_tokenizer, open('./models/morfessor_models/arab/'+ arab_morf_name,'wb'))
    pickle.dump(model, open(model_path_name,'wb'))
    
def load_model(model_path_name):
    chem_morfessor_tokenizer= pickle.load(open(model_path_name, 'rb'))
    return chem_morfessor_tokenizer

class Toxic_Tokenizer(PreTrainedTokenizer):
    r"""
    Constructs a SMILES tokenizer. Based on SMILES Pair Encoding (https://github.com/XinhaoLi74/SmilesPE).
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary.
        spe_file (:obj:`string`):
            File containing the trained SMILES Pair Encoding vocabulary.
        unk_token (:obj:`string`, `optional`, defaults to "[UNK]"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`string`, `optional`, defaults to "[PAD]"):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`string`, `optional`, defaults to "[MASK]"):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    def __init__(
        self,
        vocab_file,
        spe_file='./vocabs/spe_vocab/SPE_ChEMBL.txt',
        tokenizer_type=None,
        tokenizer_path=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file)
            )
        if not os.path.isfile(spe_file):
            raise ValueError(
                "Can't find a SPE vocabulary file at path '{}'.".format(spe_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.spe_vocab = codecs.open(spe_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.spe_tokenizer = SPE_Tokenizer(self.spe_vocab)
        self.tokenizer_type = tokenizer_type
        self.tokenizer_path = tokenizer_path
        

        # load tokenizers based on tokenizer_type
        if self.tokenizer_type == "Morfessor":
            self.tokenizer = load_model(self.tokenizer_path)
        elif self.tokenizer_type == "BPE":
            self.tokenizer =  load_model(self.tokenizer_path)
        elif self.tokenizer_type == "WordPiece":
            self.tokenizer =  load_model(self.tokenizer_path)
        else:
            self.tokenizer = None

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)
    
    # customization for SMILES
    def _tokenize(self, text):
        """ Tokenize a string. """
        if self.tokenizer_type == "Atom-wise":
            return atomwise_tokenizer(text)
        elif self.tokenizer_type == "MacFrag":
            tokens =text.split(',')
            return tokens
        elif self.tokenizer_type == "SPE":
            return self.spe_tokenizer.tokenize(text).split(' ')
        elif self.tokenizer_type == "Morfessor":
            return self.tokenizer.viterbi_segment(text)[0]
        elif self.tokenizer_type == "BPE":
            return self.tokenizer.encode(text).tokens
        elif self.tokenizer_type == "WordPiece":
            return self.tokenizer.encode(text).tokens 
        else:
            return text.split(' ')

        
        
            


    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string
    

    # handle special tokens and sequence classification tasks
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
    

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        """
        Converts a sequence of token IDs into a single string.
        Args:
            token_ids (List[int] or List[List[int]]):
                The token IDs to be decoded.
            skip_special_tokens (bool, optional):
                Whether to skip special tokens when decoding.
                Defaults to False.
            clean_up_tokenization_spaces (bool, optional):
                Whether to clean up the tokenization spaces.
                Defaults to True.
        Returns:
            str or List[str]:
                The decoded sequence(s).
        """
        # handle token_ids if torch.Tensor
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, List):
            if isinstance(token_ids[0], int):
                # Convert token IDs to tokens
                tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
                if skip_special_tokens:
                    # Remove special tokens
                    tokens = self._remove_special_tokens(tokens)
                # Join tokens into a string
                text = self.convert_tokens_to_string(tokens)
                if clean_up_tokenization_spaces:
                    # Clean up tokenization spaces
                    text = self.clean_up_tokenization(text)
                return tokens 
            elif isinstance(token_ids[0], List):
                # Recursively decode each element of the list
                return [self.decode(ids, skip_special_tokens, clean_up_tokenization_spaces) for ids in token_ids]
        raise ValueError("Unsupported input format. Input must be a list of integers or a nested list of integers.")





