from tokenizers import Tokenizer, models, trainers, pre_tokenizers

class WordPieceTrainer:
    def __init__(self, vocab_size, min_frequency):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = Tokenizer(models.WordPiece())

    def train(self, file_path):
        try:
            # Read molecules from file
            with open(file_path, "r") as file:
                molecules = file.readlines()

            # Train the tokenizer
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[UNK]"],  # Consider adding more special tokens if required
                min_frequency=self.min_frequency
            )
            self.tokenizer.train_from_iterator(molecules, trainer=trainer)

            # return tokenizer and vocab
            return self.tokenizer, self.tokenizer.get_vocab()
        
        except Exception as e:
            print(f"Error during training: {e}")
            return None, None

    def save(self, save_path):
        try:
            self.tokenizer.save(save_path)
        except Exception as e:
            print(f"Error saving tokenizer: {e}")

    def load(self, tokenizer_path):
        try:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")

    def tokenize(self, text):
        if not self.tokenizer:
            print("Tokenizer has not been trained or loaded.")
            return []

        encoding = self.tokenizer.encode(text)
        return encoding.tokens
