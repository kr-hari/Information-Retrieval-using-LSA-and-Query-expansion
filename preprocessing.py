from util import *
import nltk.data
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize

def get_wordnet_pos(treebank_tag):
    """ 
    Function to return Parts of Speech tag using Wordnet
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

class Preprocessing():
    def sentenceSegmentation(self, text):
        """
        Parameters
        ----------
            arg1 : str
                A string (a bunch of sentences)
        Returns
        -------
            list
                A list of strings where each strin is a single sentence
        """
        #sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        segmentedText = nltk.data.load('tokenizers/punkt/english.pickle').tokenize(text.strip())
        return segmentedText

    def tokenization(self, text, args):
        """
        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence
        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """

        

        t = TreebankWordTokenizer()
        tokenizedText = []
        for sentence in text:
            tokenized_sentence = t.tokenize(sentence)
            if args.restrict_tokenization:
                tokenized_sentence = [word for word in tokenized_sentence if word.isalnum()]
            tokenizedText.append(tokenized_sentence)
        return tokenizedText

    def reduction(self, text):
        """
        Stemming/Lemmatization

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of
            stemmed/lemmatized tokens representing a sentence
        """
        x = []
        count = 0
        for j in text:
            xlist = []
            # import pdb
            for word, pos in nltk.pos_tag(text[count]):
                try:
                    xlist.append(WordNetLemmatizer().lemmatize(word,get_wordnet_pos(pos)))
                except:
                    xlist.append(word)
            # pdb.set_trace()
            x.append(xlist)
        reducedText =x 

        return reducedText

        # x = []
        # count = 0
        # for j in text:
        #     xlist = []
        #     for i in text[count]:
        #         xlist.append(WordNetLemmatizer().lemmatize(i))
        #     count += 1
        #     x.append(xlist)
        # reducedText = x
        # return reducedText

    def stopwordremove(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """
        x = []
        stop_words = set(stopwords.words('english'))
        for ele in text:
            xlist = []
            for word in ele:
                if word not in stop_words:
                    xlist.append(word)
            x.append(xlist)
        stopwordRemovedText = x
        return stopwordRemovedText