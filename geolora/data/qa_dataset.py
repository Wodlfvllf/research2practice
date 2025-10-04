# geolora/data/qa_dataset.py
import torch
from torch.utils.data import Dataset
from typing import List, Dict

class QADataset(Dataset):
    """
    A dummy Question-Answering dataset.
    """
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self.questions, self.answers = self._generate_dummy_data()

    def _generate_dummy_data(self) -> (List[str], List[str]):
        """Generates synthetic QA pairs."""
        questions = [f"What is the capital of imaginary country {i}?" for i in range(self.num_samples)]
        answers = [f"The capital is City {i}." for i in range(self.num_samples)]
        return questions, answers

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"question": self.questions[idx], "answer": self.answers[idx]}

def get_qa_dataset(num_samples: int = 100) -> QADataset:
    """Returns an instance of the dummy QA dataset."""
    return QADataset(num_samples=num_samples)
