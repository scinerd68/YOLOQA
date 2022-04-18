import json
# import re
import numpy as np
import torch


class LoadData:
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
    
    def create_data(self):
        contexts = []
        questions = []
        answers = []
        for group in self.data['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        answers.append(answer)
                        questions.append(question)
                        contexts.append(context)

        return answers, questions, contexts


class PreprocessData:
    def __init__(self, tokenizer, answers, questions, contexts):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer

    # def reg_tokenize(self, text):
    #     WORD = re.compile(r'\w+')
    #     words = WORD.findall(text)
    #     return words

    def create_encodings(self):
        encodings = self.tokenizer(self.contexts, self.questions, truncation=True, padding=True)
        self.__add_answer_end_pos()
        encodings = self.__convert_to_token_start_end_pos(encodings)
        encodings = self.__create_labels(encodings)
        return encodings
        
    def __add_answer_end_pos(self):
        for answer, text in zip(self.answers, self.contexts):
            real_answer = answer['text']
            start_idx = answer['answer_start']
            # Get the real end index
            end_idx = start_idx + len(real_answer)

            # Deal with the problem of 1 or 2 more characters 
            if text[start_idx:end_idx] == real_answer:
                answer['answer_end'] = end_idx
            # When the real answer is more by one character
            elif text[start_idx-1:end_idx-1] == real_answer:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1  
            # When the real answer is more by two characters  
            elif text[start_idx-2:end_idx-2] == real_answer:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2 

    def __convert_to_token_start_end_pos(self, encodings):
        start_token_positions = []
        end_token_positions = []
        for index, answer in enumerate(self.answers):
            start = encodings.char_to_token(index, answer['answer_start'])
            end = encodings.char_to_token(index, answer['answer_end'])

            # if start = None, the answers have been truncated
            if start == None:
                start = self.tokenizer.model_max_length

            # if end == None, the 'char_to_token' function points to the space after the correct token, so add - 1
            if end == None:
                end = encodings.char_to_token(index, answer['answer_end'] - 1)
                # if end is still None, the answers have been truncated
                if end == None:
                    end = self.tokenizer.model_max_length

            start_token_positions.append(start)
            end_token_positions.append(end)

        encodings['start_positions'] = start_token_positions
        encodings['end_positions'] = end_token_positions
        return encodings

    def __create_labels(self, encodings):
        encodings['answer_length'] = np.array(encodings['end_positions'])\
         - np.array(encodings['start_positions']) + 1
        labels = np.zeros((len(self.answers), self.tokenizer.model_max_length, 
            2)) # num_example * seq_length * 2

        for example_idx, start in enumerate(encodings['start_positions']):
            if start < self.tokenizer.model_max_length: # if the answer is not truncated
                labels[example_idx, start, 0] = 1
                labels[example_idx, start, 1] = encodings['answer_length'][example_idx]

        encodings['labels'] = labels
        return encodings


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)