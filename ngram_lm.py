import math
import random

# PLEASE do not delete or modify the comments that divide the code
# into sections, like the following comment.

################################################################################
# Utility Functions
################################################################################


COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']


def start_pad(c):
    ''' Returns a padding string of length c to append to the front of text
        as a pre-processing step to building n-grams. c = n-1 '''
    return '~' * c


def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-c context and the second is the character '''
    sentence = start_pad(c) + text
    list = []

    for i in range(len(sentence) - c):
        context = sentence[i:i + c]
        char = sentence[i + c]
        list.append((context, char))

    return list


def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model trained on the entire text
        found in the path file '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model


def create_ngram_model_lines(model_class, path, c=2, k=0):
    '''Creates and returns a new n-gram model trained line by line on the
        text found in the path file. '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model


################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, c, k):
        self.c = c
        self.k = k
        self.vocab = set()
        self.ngrams = {}

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return sorted(self.vocab)

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        grams = ngrams(self.c, text)

        for gram in grams:
            self.vocab.add(gram[1])

            if self.ngrams.get(str(gram)):
                self.ngrams[str(gram)] += 1
            else:
                self.ngrams[str(gram)] = 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        count = 0

        for ch in self.get_vocab():
            n = (context, ch)
            if self.ngrams.get(str(n)):
                count += self.ngrams.get(str(n))

        if count == 0:
            return 1 / len(self.vocab)
        else:
            n = (context, char)
            # add-k smoothing
            return (self.ngrams.get(str(n), 0) + self.k) / (count + (self.k * len(self.get_vocab())))

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()

        sum_prob = 0
        for j in sorted(self.vocab):
            sum_prob += self.prob(context, j)

            if r < sum_prob:
                return j

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        text = []
        context = start_pad(self.c)

        for i in range(length):
            char = self.random_char(context)
            text.append(char)
            context = context[1:] + char

        return "".join(text)

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        N = len(text)
        sum_log = 0
        grams = ngrams(self.c, text)

        for gram in grams:
            p = self.prob(gram[0], gram[1])

            if p == 0:
                return float('inf')

            sum_log += math.log2(p)

        return 2 ** (-1 / N * sum_log)


################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        super().__init__(c, k)
        self.weights = [1 / (self.c + 1) for _ in range(self.c + 1)]

    def update(self, text):
        for c in range(self.c + 1):
            grams = ngrams(c, text)

            for gram in grams:
                self.vocab.add(gram[1])

                if self.ngrams.get(str(gram)):
                    self.ngrams[str(gram)] += 1
                else:
                    self.ngrams[str(gram)] = 1

    def get_prob(self, context, char):
        count = 0

        for ch in self.get_vocab():
            n = (context, ch)
            if self.ngrams.get(str(n)):
                count += self.ngrams.get(str(n))

        if count == 0:
            return 1 / len(self.vocab)
        else:
            n = (context, char)
            # add-k smoothing
            return (self.ngrams.get(str(n), 0) + self.k) / (count + (self.k * len(self.get_vocab())))

    def prob(self, context, char):
        """
        P(Wn|Wn-2Wn-1) = lP(Wn) + lP(Wn|Wn-1) + lP(Wn|Wn-2Wn-1)
        """
        total_prob = 0

        for c in range(len(context) + 1):
            weight = self.weights[c]
            p = self.get_prob(context[:c], char)

            total_prob += weight * p
        return total_prob


m = NgramModelWithInterpolation(1, 0)
m.update('abab')

print(m.prob('a', 'a'))
print(m.prob('a', 'b'))

# t = NgramModelWithInterpolation(2, 1)
# t.update('abab')
# t.update('abcd')
#
# print(t.prob('~a', 'b'))
# print(t.prob('ba', 'b'))


################################################################################
# Your N-Gram Model Experimentations
################################################################################

# Add all code you need for testing your language model as you are
# developing it as well as your code for running your experiments
# here.
#
# Hint: it may be useful to encapsulate it into multiple functions so
# that you can easily run any test or experiment at any time.


def assert_equals(v1, v2, message):
    print("Asserting " + str(v1) + " is equal to " + str(v2))
    assert str(v1) == str(v2), message

#
# m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
# print(m.random_text(500))
#
# print("--------")
#
# m = NgramModel(1, 0)
# m.update('abab')
# assert_equals("['a', 'b']", m.get_vocab(), 'vocab is equal')
# m.update('abcd')
# assert_equals("['a', 'b', 'c', 'd']", m.get_vocab(), 'vocab is equal')
# assert_equals("0.0", m.prob('a', 'a'), 'probability is 0')
#
# #
# assert_equals("1.0", m.prob('a', 'b'), "probability is 1.0")
# assert_equals("0.0", m.prob('~', 'c'), "probability is 0.0")
# assert_equals("0.5", m.prob('b', 'c'), "probability is 0.5")
#
# random.seed(1)
# assert_equals("['a', 'd', 'd', 'b', 'b', 'b', 'c', 'd', 'a', 'a', 'd', 'b', 'd', 'a', 'b', 'c', 'a', 'd', 'd', 'a', "
#               "'a', 'c', 'd', 'b', 'a']", [m.random_char('') for i in range(25)], 'first 25 random characters same')
# assert_equals("1.5157165665103982", m.perplexity('abcda'), 'perplexity same')
