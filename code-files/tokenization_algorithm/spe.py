import codecs
from SmilesPE.learner import learn_SPE
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import codecs
from SmilesPE.tokenizer import *

class SMILESLearner():
    def __init__(self, file_name, output_file, num_symbols, min_frequency, augmentation=1, verbose=True, total_symbols=True):
        self.file_name = file_name
        self.output_file = output_file
        self.num_symbols = num_symbols # maximum total number of SPE symbols, set to 30,000
        self.min_frequency = min_frequency # the minimum frequency of SPE symbols appears, set to 2,000.
        self.augmentation = augmentation #times of SMILES augmentation, set to 1. The final data set is ~2 times larger than the original one.
        self.verbose = verbose # print the learning process or merging process
        self.total_symbols = total_symbols # whether to use the total number of symbols or the number of unique symbols in the data set.
    
    def learn_SMILES(self):
        with open(self.file_name, "r") as ins:
            SMILES = []
            for line in ins:
                SMILES.append(line.split('\n')[0])
        
        print('Number of training SMILES :', len(SMILES))
        output = codecs.open(self.output_file, 'w')
        learn_SPE(SMILES, output, self.num_symbols, min_frequency=self.min_frequency, augmentation=self.augmentation, verbose=self.verbose, total_symbols=self.total_symbols)
        output.close()
        print("saving vocabulary to", self.output_file)
        spe_vob= codecs.open(self.output_file)
        spe = SPE_Tokenizer(spe_vob)
        return spe , spe_vob

    def learn_BPE(self):
        voc_size = self.num_symbols
        file_name = self.file_name
        print("Training BPE with {} tokens".format(voc_size))
        print("Input file: {}".format(self.file_name))
        print("min_frequency: {}".format(self.min_frequency))
        input_file = [file_name]
        tokenizer = Tokenizer(BPE()) 
        # tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size= voc_size)
        tokenizer.train(input_file, trainer)
        # print vocab size 
        print("Vocab size: {}".format(tokenizer.get_vocab_size()))
        # save tokenizer 
        name= file_name.split("/")[-1].split(".")[0]
        print("saving BPE Tokenizer")
        # tokenizer.save("./models/tokenizers/bpe_tokenizer_"+name+"_"+str(voc_size)+".json")
        return tokenizer , tokenizer.get_vocab()

