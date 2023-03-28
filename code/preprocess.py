import re
from random import randrange
from random import randint
from random import shuffle
import random
import Utils

class BertPreprocess():
    def __init__(self, inputCorpusList):
        self.wordList , self.sentences= self.CleanAndGetWordList(inputCorpusList)

        self.wordDict, self.numberDict, self.vocabSize= self.BuildWordIndexDict()


    def CleanAndGetWordList(self, stringList:list)->list:
        sentences = re.sub("[.,!?-]", '', stringList.lower()).split('\n')  # filter '.', ',', '?', '!'
        word_list = list(set(" ".join(sentences).split()))
        # print(word_list)
        return word_list, sentences
    def GetTokenList(self, sentence:str)->list:
        return list(sentence.split())
    def BuildWordIndexDict(self)->dict:
        word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        for i, w in enumerate(self.wordList):
            word_dict[w] = i + 4
            number_dict = {i: w for i, w in enumerate(word_dict)}
            # print(number_dict)
            vocab_size = len(word_dict)
            # print(vocab_size)
        return (word_dict, number_dict, vocab_size)

    def make_batch(self):
       batch = []
       batch_size = 10000
       positive = negative = 0
       max_pred = 15
       maxlen = max([len(i.split()) for i in self.sentences])
       while positive != batch_size/2 or negative != batch_size/2:
           tokens_a_index, tokens_b_index= randrange(len(self.sentences)), randrange(len(self.sentences))

           tokens_a, tokens_b= self.GetTokenList(sentences[tokens_a_index]), self.GetTokenList(sentences[tokens_b_index])

           input_ids = [self.wordDict['[CLS]']] + tokens_a + [self.wordDict['[SEP]']] + tokens_b + [self.wordDict['[SEP]']]
           segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

           # MASK LM
           n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15)))) # 15 % of tokens in one sentence
           cand_maked_pos = [i for i, token in enumerate(input_ids)
                             if token != self.wordDict['[CLS]'] and token != self.wordDict['[SEP]']]
           shuffle(cand_maked_pos)
           masked_tokens, masked_pos = [], []
           for pos in cand_maked_pos[:n_pred]:
               masked_pos.append(pos)
               masked_tokens.append(input_ids[pos])
               if random.random() < 0.8:  # 80%
                   input_ids[pos] = self.wordDict['[MASK]'] # make mask
                   # print("mask: ",  input_ids[pos])
               elif random.random() < 0.5:  # 10%
                   index = randint(0, self.vocabSize - 1) # random index in vocabulary
                   print("replace: ", input_ids[pos], " WITH ", self.numberDict[index], " ",
                         self.wordDict[self.numberDict[index]])
                   input_ids[pos] = self.wordDict[self.numberDict[index]] # replace


           # Zero Paddings
           n_pad = maxlen - len(input_ids)

           input_ids.extend([0] * n_pad)
           segment_ids.extend([0] * n_pad)

           # Zero Padding (100% - 15%) tokens
           if max_pred > n_pred:
               n_pad = max_pred - n_pred
               masked_tokens.extend([0] * n_pad)
               masked_pos.extend([0] * n_pad)

           if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
               batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
               positive += 1
           elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
               batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
               negative += 1
           break
       return batch




if __name__ == "__main__":
    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )
    a = BertPreprocess(text)
    word_list, sentences = a.CleanAndGetWordList(text)
    # print(word_list)
    # print(sentences)
    # print(a.wordDict)
    batchZero = a.make_batch()[0]     # batchZero = [input_ids, segment_ids, masked_tokens, masked_pos, doesFollow]
    Utils.print_list(batchZero)

