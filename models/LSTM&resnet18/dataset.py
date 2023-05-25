from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from torch.utils.data import Dataset
import torch

class ImageCaptionDataset(Dataset):
    
    def __init__(self, df, tokenizer, image_features, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.image_features = image_features
        self.max_length = max_length

        # Preprocess all captions to sequences of input-output pairs
        self.pairs = []
        for _, row in self.df.iterrows():
            image_id = row['image']
            image_feature = self.image_features.get(image_id)
            if image_feature is None:
                continue

            caption = row['caption']
            sequence = self.tokenizer.texts_to_sequences([caption])[0]
            
            for i in range(1, len(sequence)):
                in_seq = sequence[:i]
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                self.pairs.append((image_feature, in_seq))
        
        del self.image_features #finished processing, saved in pairs already
        del self.tokenizer
        del self.max_length
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_feature, in_seq = self.pairs[idx]
        X1, X2 = torch.tensor(image_feature, dtype=torch.float32), torch.tensor(in_seq, dtype=torch.long)
        caption = self.df.iloc[idx]["caption"]
        return X1, X2,caption
