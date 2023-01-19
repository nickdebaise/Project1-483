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

    def update_weights(self, new_weights):
        self.weights = new_weights

    def update(self, text):
        for c in range(self.c + 1):
            grams = ngrams(c, text)

            for gram in grams:
                self.vocab.add(gram[1])

                if self.ngrams.get(str(gram)):
                    self.ngrams[str(gram)] += 1
                else:
                    self.ngrams[str(gram)] = 1

    def prob(self, context, char):
        """
        P(Wn|Wn-2Wn-1) = lP(Wn) + lP(Wn|Wn-1) + lP(Wn|Wn-2Wn-1)
        """
        total_prob = 0

        for c in range(len(context) + 1):
            weight = self.weights[c]

            p = super().prob(context[c:], char)

            total_prob += weight * p
        return total_prob


class LanguageModel(NgramModelWithInterpolation):

    def __init__(self, c, k):
        super().__init__(c, k)

    def prob_of_text(self, text):
        total_prob = 0
        grams = ngrams(self.c, text)

        for gram in grams:
            p = self.prob(gram[0], gram[1])

            if p == 0:
                return 0

            total_prob += math.log2(p)

        return 2 ** total_prob


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


#######
# SHAKESPEARE TESTING
#######

# m2 = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
# print(m2.random_text(250))
#
# print(m2.random_text(20))
#
# print("---")
#
# m4 = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
# print(m4.random_text(250))
#
# print(m4.random_text(20))
#
# print("---")
#
# m6 = create_ngram_model(NgramModel, 'shakespeare_input.txt', 6)
# print(m6.random_text(250))
#
# print(m6.random_text(20))
#
# print("---")
#
# m10 = create_ngram_model(NgramModel, 'shakespeare_input.txt', 10)
# print(m10.random_text(250))
#
# print(m10.random_text(20))

#####
#
# Perplexity & Smoothing
#
#####
#
# m = NgramModel(1, 0)
# m.update('abab')
# m.update('abcd')
# print(m.perplexity('abcd'))
# print(m.perplexity('abca'))
# print(m.perplexity('abcda'))
#
# m4_sonnet = create_ngram_model(NgramModel, 'nytimes_article.txt', 6, 0.001)
#
# print("----")
# print(m4_sonnet.perplexity("From fairest creatures we desire increase"))
# print("----")
# famous_quotes = ["Life's but a walking shadow, a poor player that struts and frets his hour upon the stage, "
#                  "and then is heard no more",
#                  "Give every man thy ear, but few thy voice.",
#                  "Our doubts are traitors and make us lose the good we oft might win by fearing to attempt.",
#                  "To be, or not to be",
#                  "The greatest glory in living lies not in never falling, but in rising every time we fall",
#                  "The way to get started is to quit talking and begin doing",
#                  "Your time is limited, so don't waste it living someone else's life. Don't be trapped by dogma – "
#                  "which is living with the results of other people's thinking",
#                  "If you look at what you have in life, you'll always have more. If you look at what you don't have "
#                  "in life, you'll never have enough"]

# avg1 = 0
# avg2 = 0
#
# for i, quote in enumerate(famous_quotes):
#     perp = m4_sonnet.perplexity(quote)
#
#     if i >= 4:
#         avg2 += perp
#     else:
#         avg1 += perp
#     print(perp)
#
# print(avg1 / 4, avg2 / 4)

########
#
# Ngram model testing
#
########

# print("--------")
#
# m1 = NgramModel(1, 0)
# m1.update('abab')
# assert_equals("['a', 'b']", m1.get_vocab(), 'vocab is equal')
# m1.update('abcd')
# assert_equals("['a', 'b', 'c', 'd']", m1.get_vocab(), 'vocab is equal')
# assert_equals("0.0", m1.prob('a', 'a'), 'probability is 0')
#
# #
# assert_equals("1.0", m1.prob('a', 'b'), "probability is 1.0")
# assert_equals("0.0", m1.prob('~', 'c'), "probability is 0.0")
# assert_equals("0.5", m1.prob('b', 'c'), "probability is 0.5")
#
# random.seed(1)
# assert_equals("['a', 'd', 'd', 'b', 'b', 'b', 'c', 'd', 'a', 'a', 'd', 'b', 'd', 'a', 'b', 'c', 'a', 'd', 'd', 'a', "
#               "'a', 'c', 'd', 'b', 'a']", [m1.random_char('') for i in range(25)], 'first 25 random characters same')
# assert_equals("1.5157165665103982", m1.perplexity('abcda'), 'perplexity same')
#
# m1 = NgramModelWithInterpolation(1, 0)
# m1.update('abab')
#
# print(m1.prob('a', 'a'))
# print(m1.prob('a', 'b'))
# #
# t = NgramModelWithInterpolation(2, 1)
# t.update('abab')
# t.update('abcd')
#
# print(t.prob('~a', 'b'))
# print(t.prob('ba', 'b'))
# print(t.prob('~c', 'd'))
# print(t.prob('bc', 'd'))
#
# print("----")
# t.update_weights([0.1, 0.8, 0.1])
#
# print(t.prob('~a', 'b'))
# print(t.prob('ba', 'b'))
# print(t.prob('~c', 'd'))
# print(t.prob('bc', 'd'))
#


#############
#
# LANGUAGE TRAINING
#
############

#
# def train_language_models(training_paths, ngram, k=1):
#     models = []
#
#     for path in training_paths:
#         models.append(create_ngram_model_lines(LanguageModel, path, ngram, k))
#
#     return models
#
#
# def create_language_models():
#     training_paths = ["train/" + code + ".txt" for code in COUNTRY_CODES]
#     validation_paths = ["val/" + code + ".txt" for code in COUNTRY_CODES]
#
#     best_ngram = -1
#     best_correct_percentage = -1
#
#     for x in range(7):
#         num_correct = 0
#         num_total = 0
#         models = train_language_models(training_paths, x)
#
#         for i, path in enumerate(validation_paths):
#             language = COUNTRY_CODES[i]
#
#             with open(validation_paths[i], encoding='utf-8', errors='ignore') as f:
#                 for line in f:
#                     # print("CITY NAME: " + line)
#                     max_prob = -1
#                     curr_model = None
#                     for j, model in enumerate(models):
#                         p = model.prob_of_text(line)
#                         # print(COUNTRY_CODES[j] + " assigned " + str(p) + " to the city")
#                         if p > max_prob or max_prob == -1:
#                             max_prob = p
#                             curr_model = COUNTRY_CODES[j]
#
#                     num_total += 1
#                     # print("Best model was " + curr_model)
#                     # print("Actual model was ", language)
#                     if curr_model == language:
#                         num_correct += 1
#
#             print(num_correct, num_total, num_correct / num_total)
#
#         if best_ngram == -1 or num_correct / num_total > best_correct_percentage:
#             best_ngram = x
#             best_correct_percentage = num_correct / num_total
#
#         print("Finished with " + str(x) + " ngram")
#     print(
#         "The best ngram model was the model with " + str(best_ngram) + " grams. It had a correct percentage of " + str(
#             best_correct_percentage))
#
#
# create_language_models()
