import random
import numpy as np
from typing import List, Tuple
import torch

class DataAugmentation:
    """Data augmentation techniques for text sentiment analysis"""
    
    def __init__(self, seed=42):
        """
        Initialize augmentation class
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Common English synonyms for simple replacement
        self.synonym_dict = {
            'good': ['great', 'excellent', 'nice', 'fine', 'wonderful'],
            'bad': ['terrible', 'awful', 'horrible', 'poor', 'unpleasant'],
            'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
            'small': ['tiny', 'little', 'mini', 'miniature', 'petite'],
            'happy': ['joyful', 'cheerful', 'delighted', 'pleased', 'content'],
            'sad': ['unhappy', 'miserable', 'depressed', 'gloomy', 'sorrowful'],
            'beautiful': ['gorgeous', 'stunning', 'lovely', 'attractive', 'pretty'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
            'love': ['adore', 'cherish', 'treasure', 'appreciate', 'enjoy'],
            'hate': ['despise', 'detest', 'loathe', 'dislike', 'abhor'],
            'amazing': ['incredible', 'awesome', 'fantastic', 'wonderful', 'marvelous'],
            'boring': ['dull', 'tedious', 'monotonous', 'uninteresting', 'tiresome'],
        }
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap n pairs of words in the sentence
        
        Args:
            text: Input text
            n: Number of swaps to perform
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p
        
        Args:
            text: Input text
            p: Probability of deleting each word
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        # If all words are deleted, return a random word
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert n words into the sentence
        
        Args:
            text: Input text
            n: Number of insertions
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        for _ in range(n):
            new_word = random.choice(words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, new_word)
        
        return ' '.join(words)
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Replace n words with their synonyms
        
        Args:
            text: Input text
            n: Number of words to replace
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        # Find replaceable words
        replaceable_indices = [
            i for i, word in enumerate(words) 
            if word in self.synonym_dict
        ]
        
        if len(replaceable_indices) == 0:
            return text
        
        # Randomly select words to replace
        n = min(n, len(replaceable_indices))
        indices_to_replace = random.sample(replaceable_indices, n)
        
        for idx in indices_to_replace:
            word = words[idx]
            synonym = random.choice(self.synonym_dict[word])
            words[idx] = synonym
        
        return ' '.join(words)
    
    def back_translation(self, text: str) -> str:
        """
        Simulate back translation (English -> French -> English)
        Note: This is a placeholder. In production, use actual translation models
        
        Args:
            text: Input text
            
        Returns:
            Augmented text (in this case, just slightly modified)
        """
        # For now, we'll just do a combination of other techniques
        # In production, use MarianMT or similar models
        augmented = self.random_swap(text, n=1)
        augmented = self.synonym_replacement(augmented, n=1)
        return augmented
    
    def easy_data_augmentation(self, text: str, alpha: float = 0.1, num_aug: int = 4) -> List[str]:
        """
        Easy Data Augmentation (EDA) technique
        Combines multiple augmentation methods
        
        Args:
            text: Input text
            alpha: Parameter controlling augmentation intensity
            num_aug: Number of augmented sentences to generate
            
        Returns:
            List of augmented texts
        """
        augmented_texts = []
        num_words = len(text.split())
        
        # Calculate number of operations based on sentence length
        n = max(1, int(alpha * num_words))
        
        for _ in range(num_aug):
            # Randomly choose augmentation technique
            choice = random.randint(0, 3)
            
            if choice == 0:
                aug_text = self.synonym_replacement(text, n)
            elif choice == 1:
                aug_text = self.random_insertion(text, n)
            elif choice == 2:
                aug_text = self.random_swap(text, n)
            else:
                aug_text = self.random_deletion(text, alpha)
            
            augmented_texts.append(aug_text)
        
        return augmented_texts
    
    def augment_dataset(self, 
                       texts: List[str], 
                       labels: List[int], 
                       method: str = 'eda',
                       num_aug: int = 2,
                       balance_classes: bool = True) -> Tuple[List[str], List[int]]:
        """
        Augment entire dataset
        
        Args:
            texts: List of input texts
            labels: List of labels
            method: Augmentation method ('eda', 'swap', 'delete', 'insert', 'synonym')
            num_aug: Number of augmented samples per original sample
            balance_classes: Whether to balance classes through augmentation
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        
        if balance_classes:
            # Find majority class size
            from collections import Counter
            label_counts = Counter(labels)
            max_count = max(label_counts.values())
            
            # Augment minority classes
            for label, count in label_counts.items():
                if count < max_count:
                    # Find indices of this class
                    indices = [i for i, l in enumerate(labels) if l == label]
                    
                    # Calculate how many augmentations needed
                    needed = max_count - count
                    aug_per_sample = needed // len(indices) + 1
                    
                    for idx in indices:
                        text = texts[idx]
                        
                        if method == 'eda':
                            aug_texts = self.easy_data_augmentation(text, num_aug=aug_per_sample)
                        elif method == 'swap':
                            aug_texts = [self.random_swap(text) for _ in range(aug_per_sample)]
                        elif method == 'delete':
                            aug_texts = [self.random_deletion(text) for _ in range(aug_per_sample)]
                        elif method == 'insert':
                            aug_texts = [self.random_insertion(text) for _ in range(aug_per_sample)]
                        elif method == 'synonym':
                            aug_texts = [self.synonym_replacement(text) for _ in range(aug_per_sample)]
                        else:
                            aug_texts = [text]
                        
                        augmented_texts.extend(aug_texts[:needed])
                        augmented_labels.extend([label] * min(len(aug_texts), needed))
                        needed -= len(aug_texts)
                        
                        if needed <= 0:
                            break
        else:
            # Augment all samples equally
            for text, label in zip(texts, labels):
                if method == 'eda':
                    aug_texts = self.easy_data_augmentation(text, num_aug=num_aug)
                elif method == 'swap':
                    aug_texts = [self.random_swap(text) for _ in range(num_aug)]
                elif method == 'delete':
                    aug_texts = [self.random_deletion(text) for _ in range(num_aug)]
                elif method == 'insert':
                    aug_texts = [self.random_insertion(text) for _ in range(num_aug)]
                elif method == 'synonym':
                    aug_texts = [self.synonym_replacement(text) for _ in range(num_aug)]
                else:
                    aug_texts = []
                
                augmented_texts.extend(aug_texts)
                augmented_labels.extend([label] * len(aug_texts))
        
        return augmented_texts, augmented_labels


class MixUpAugmentation:
    """MixUp augmentation for text embeddings"""
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp augmentation
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def mixup(self, 
              embeddings: torch.Tensor, 
              labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation to embeddings
        
        Args:
            embeddings: Batch of embeddings [batch_size, embedding_dim]
            labels: Batch of labels [batch_size]
            
        Returns:
            Tuple of (mixed_embeddings, labels_a, labels_b, lambda)
        """
        batch_size = embeddings.size(0)
        
        # Sample lambda from beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Random permutation
        index = torch.randperm(batch_size).to(embeddings.device)
        
        # Mix embeddings
        mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index, :]
        
        # Return mixed embeddings and both labels for loss calculation
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_embeddings, labels_a, labels_b, lam


if __name__ == "__main__":
    # Example usage
    augmenter = DataAugmentation()
    
    sample_text = "This movie is really good and I enjoyed it a lot"
    print("Original:", sample_text)
    print("\nRandom Swap:", augmenter.random_swap(sample_text))
    print("Random Deletion:", augmenter.random_deletion(sample_text))
    print("Random Insertion:", augmenter.random_insertion(sample_text))
    print("Synonym Replacement:", augmenter.synonym_replacement(sample_text))
    
    print("\nEDA Augmentation:")
    aug_texts = augmenter.easy_data_augmentation(sample_text, num_aug=3)
    for i, text in enumerate(aug_texts, 1):
        print(f"{i}. {text}")
