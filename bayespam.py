import argparse
import os
import string
import math
import re

from enum import Enum

class MessageType(Enum):
    REGULAR = 1,
    SPAM = 2

class Counter():

    def __init__(self):
        self.counter_regular = 0
        self.counter_spam = 0

    def increment_counter(self, message_type):
        """
        Increment a word's frequency count by one, depending on whether it occurred in a regular or spam message.

        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            self.counter_regular += 1
        else:
            self.counter_spam += 1

class Bayespam():

    def __init__(self):
        self.regular_list = None
        self.spam_list = None
        self.vocab = {}
        self.n_messages_regular = 0
        self.n_messages_spam = 0
        self.n_messages_total = 0
        self.prob_regular = 0
        self.prob_spam = 0
        self.n_words_regular = 0
        self.n_words_spam = 0
        self.tuning_parameter = 0.0001
        self.cond_likelihood_regular = {} 
        self.cond_likelihood_spam = {} 
        self.test_words = []
        self.logprob_reg_msg_minus_alpha = 0
        self.logprob_spam_msg_minus_alpha = 0

    def list_dirs(self, path):
        """
        Creates a list of both the regular and spam messages in the given file path.

        :param path: File path of the directory containing either the training or test set
        :return: None
        """
        self.regular_list = None
        self.spam_list = None
        # Check if the directory containing the data exists
        if not os.path.exists(path):
            print("Error: directory %s does not exist." % path)
            exit()

        regular_path = os.path.join(path, 'regular')
        spam_path = os.path.join(path, 'spam')

        # Create a list of the absolute file paths for each regular message
        # Throws an error if no directory named 'regular' exists in the data folder
        try:
            self.regular_list = [os.path.join(regular_path, msg) for msg in os.listdir(regular_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'regular'." % path)
            exit()

        # Create a list of the absolute file paths for each spam message
        # Throws an error if no directory named 'spam' exists in the data folder
        try:
            self.spam_list = [os.path.join(spam_path, msg) for msg in os.listdir(spam_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'spam'." % path)
            exit()

    def read_messages(self, message_type):
        """
        Parse all messages in either the 'regular' or 'spam' directory. Each token is stored in the vocabulary,
        together with a frequency count of its occurrences in both message types.
        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            message_list = self.regular_list
        elif message_type == MessageType.SPAM:
            message_list = self.spam_list
        else:
            message_list = []
            print("Error: input parameter message_type should be MessageType.REGULAR or MessageType.SPAM")
            exit()

        for msg in message_list:
            try:
                # Make sure to use latin1 encoding, otherwise it will be unable to read some of the messages
                f = open(msg, 'r', encoding='latin1')

                # Loop through each line in the message
                for line in f:
                    # Split the string on the space character, resulting in a list of tokens
                    split_line = line.split(" ")
                    # Loop through the tokens
                    for idx in range(len(split_line)):
                        token = split_line[idx].lower()  #With the lower() function we convert the string to lowercase
                        token = token.translate(str.maketrans('', '', string.punctuation)) #With these three lines of code we eliminate the punctuation
                        token = token.replace("\n", "") 
                        token = token.replace("\t", "")

                        if len(token) >= 4 and token.isalpha(): #and token.isalpha(): #using this conditional, the words with less than 4 carachters and not alphabetical strings (with numbers or punctuation) strings are not taken into account
                            if token in self.vocab.keys():
                                # If the token is already in the vocab, retrieve its counter
                                counter = self.vocab[token]
                            else:
                                # Else: initialize a new counter
                                counter = Counter()

                            # Increment the token's counter by one and store in the vocab
                            counter.increment_counter(message_type)
                            self.vocab[token] = counter

            except Exception as e:
                print("Error while reading message %s: " % msg, e)
                exit()

    def read_test_message(self, msg, message_type):
        self.test_words = []
        try:
            # Make sure to use latin1 encoding, otherwise it will be unable to read some of the messages
            f = open(msg, 'r', encoding='latin1')

            # Loop through each line in the message
            for line in f:
                # Split the string on the space character, resulting in a list of tokens
                split_line = line.split(" ")
                # Loop through the tokens
                for idx in range(len(split_line)):
                    token = split_line[idx].lower()  #With the lower() function we convert the string to lowercase
                    token = token.translate(str.maketrans('', '', string.punctuation)) #With these three lines of code we eliminate the punctuation
                    token = token.replace("\n", "") 
                    token = token.replace("\t", "")

                    if len(token) >= 4 and token.isalpha(): #and token.isalpha(): #using this conditional, the words with less than 4 carachters and not alphabetical strings (with numbers or punctuation) strings are not taken into account
                        if token not in self.test_words:
                            self.test_words.append(token)

        except Exception as e:
            print("Error while reading message %s: " % msg, e)
            exit()

    def print_vocab(self):
        """
        Print each word in the vocabulary, plus the amount of times it occurs in regular and spam messages.

        :return: None
        """
        for word, counter in self.vocab.items():
            # repr(word) makes sure that special characters such as \t (tab) and \n (newline) are printed.
            print("%s | In regular: %d | In spam: %d" % (repr(word), counter.counter_regular, counter.counter_spam))

    def write_vocab(self, destination_fp, sort_by_freq=False):
        """
        Writes the current vocabulary to a separate .txt file for easier inspection.

        :param destination_fp: Destination file path of the vocabulary file
        :param sort_by_freq: Set to True to sort the vocab by total frequency (descending order)
        :return: None
        """

        if sort_by_freq:
            vocab = sorted(self.vocab.items(), key=lambda x: x[1].counter_regular + x[1].counter_spam, reverse=True)
            vocab = {x[0]: x[1] for x in vocab}
        else:
            vocab = self.vocab

        try:
            f = open(destination_fp, 'w', encoding="latin1")

            for word, counter in vocab.items():
                # repr(word) makes sure that special  characters such as \t (tab) and \n (newline) are printed.
                f.write("%s | In regular: %d | In spam: %d\n" % (repr(word), counter.counter_regular, counter.counter_spam),)
            f.close()
        except Exception as e:
            print("An error occurred while writing the vocab to a file: ", e)

    def number_messages(self):
        """
        Calculates the number of regular and spam messages, but also the total.
        """
        self.n_messages_regular = len(self.regular_list)
        self.n_messages_spam = len(self.spam_list)
        self.n_messages_total = self.n_messages_regular + self.n_messages_spam

    def probability_total(self):
        """
        Computes the probability of having regular or normal messages in comparison with the total 
        """
        self.prob_regular =  self.n_messages_regular / self.n_messages_total
        self.prob_spam =  self.n_messages_spam / self.n_messages_total

    def number_of_words(self):
        """
        Computes the total number of words of each message type: regular and spam
        """
        for counter in self.vocab.values():
            self.n_words_regular += counter.counter_regular
        
        for counter in self.vocab.values():
            self.n_words_spam += counter.counter_spam

    def words_conditional_likelihoods(self):
        """
        Computes the conditional likelihoods for every word in each case (spam and regular)
        """

        for word, counter in self.vocab.items(): # We itarate in the diccionari vocab to get all words and its counters
            if counter.counter_regular == 0: # Check if the regular counter is 0 to avoid 0 probabilities (1 as a tuning parameter)
                self.cond_likelihood_regular[word] = self.tuning_parameter / self.n_words_regular 
            else:
                self.cond_likelihood_regular[word] = counter.counter_regular / self.n_words_regular
            
            if counter.counter_spam == 0: # Check if the spam counter is 0 to avoid 0 probabilities (1 as a tuning parameter)
                self.cond_likelihood_spam[word] = self.tuning_parameter / self.n_words_spam
            else:
                self.cond_likelihood_spam[word] = counter.counter_spam / self.n_words_spam

    def test_classification(self):
        test_classification = {}
        number_incorrects = 0
        number_corrects = 0
        for msg in self.regular_list:
            self.read_test_message(msg, MessageType.REGULAR)
            test_probabilities = self.posteriory_class_log_probabilities()
            if test_probabilities[0] > test_probabilities[1]:
                test_classification[msg] = 'Regular'
                number_corrects += 1
            else:
                test_classification[msg] = 'Spam'
                number_incorrects += 1

        print("In regular messages: ")
        print("Number of corrects: %d" % number_corrects)
        print("Number of incorrects: %d" % number_incorrects)
        accuracy = number_corrects / (number_corrects + number_incorrects)
        print("Accuracy: {:.2f}" .format(accuracy))


        number_incorrects = 0
        number_corrects = 0
        for msg in self.spam_list:
            self.read_test_message(msg, MessageType.SPAM)
            test_probabilities = self.posteriory_class_log_probabilities()
            if test_probabilities[0] > test_probabilities[1]:
                test_classification[msg] = 'Regular'
                number_incorrects += 1
            else:
                test_classification[msg] = 'Spam'
                number_corrects += 1

        print("In spam messages: ")
        print("Number of corrects: %d" % number_corrects)
        print("Number of incorrects: %d" % number_incorrects)
        accuracy = number_corrects / (number_corrects + number_incorrects)
        print("Accuracy: {:.2f}" .format(accuracy))

        for msg, clas in test_classification.items():
            print("%s : | Classification: %s" % (msg, clas))
            

    def posteriory_class_log_probabilities(self):
        """
        In this method, we compute the logProbabilities of the regular or spam messages conditioned by the message recieved,
        to classify later if rather the message is spam or regular
        """
        #list = self.split_message(msg)
        sum1 = 0
        sum2 = 0

        for word in self.test_words: # Firstly, the sum of all the logProb(Wi|regular) and logProb(Wi|spam)
            if word in self.vocab.keys():
                sum1 += math.log(self.cond_likelihood_regular[word])
                sum2 += math.log(self.cond_likelihood_spam[word])
        
        """
        Then we add the P(regular) or P(spam). We define the variable "logprob_reg_msg_minus_alpha" because both logProbabilities
        have the same term alpha, so we will ommit it when comparing
        """
        logprob_reg_msg_minus_alpha = sum1 + self.prob_regular 
        logprob_spam_msg_minus_alpha = sum2 + self.prob_spam

        output = []
        output.append(logprob_reg_msg_minus_alpha)
        output.append(logprob_spam_msg_minus_alpha)

        return output


def main():
    # We require the file paths of the training and test sets as input arguments (in that order)
    # The argparse library helps us cleanly parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str,
                        help='File path of the directory containing the training data')
    parser.add_argument('test_path', type=str,
                        help='File path of the directory containing the test data')
    args = parser.parse_args()

    # Read the file path of the folder containing the training set from the input arguments
    train_path = args.train_path

    # Initialize a Bayespam object
    bayespam = Bayespam()
    # Initialize a list of the regular and spam message locations in the training folder
    bayespam.list_dirs(train_path)

    # Parse the messages in the regular message directory
    bayespam.read_messages(MessageType.REGULAR)
    # Parse the messages in the spam message directory
    bayespam.read_messages(MessageType.SPAM)

    #bayespam.print_vocab()
    bayespam.write_vocab("vocab.txt")

    # Exercise 2.1

    # Count the number of regular, spam and total messages
    bayespam.number_messages()
    
    # Compute the probability of having regular or spam messages in comparison with the total calculated before
    bayespam.probability_total()

    # Exercise 2.2

    # Count the total number of words contained in regular mail
    bayespam.number_of_words()

    # Calculate the conditional likelihoods of every word
    bayespam.words_conditional_likelihoods()

    # Exercise 3.1

    # Calculate the posteriory class probabilities given a new email 'msg'
    test_path = args.test_path
    bayespam.list_dirs(test_path)

    bayespam.test_classification()


    print("N regular messages: ", bayespam.n_messages_regular)
    print("N spam messages: ", bayespam.n_messages_spam)
    print("Probability of regular: ", bayespam.prob_regular)
    print("Probability of spam: ", bayespam.prob_spam)
    print("Number of words in regular mail: ", bayespam.n_words_regular)
    print("Number of words in spam mail: ", bayespam.n_words_spam)

    """
    Now, implement the follow code yourselves:
    1) A priori class probabilities must be computed from the number of regular and spam messages
    2) The vocabulary must be clean: punctuation and digits must be removed, case insensitive
    3) Conditional probabilities must be computed for every word
    4) Zero probabilities must be replaced by a small estimated value
    5) Bayes rule must be applied on new messages, followed by argmax classification
    6) Errors must be computed on the test set (FAR = false accept rate (misses), FRR = false reject rate (false alarms))
    7) Improve the code and the performance (speed, accuracy)
    
    Use the same steps to create a class BigramBayespam which implements a classifier using a vocabulary consisting of bigrams
    """

if __name__ == "__main__":
    main()