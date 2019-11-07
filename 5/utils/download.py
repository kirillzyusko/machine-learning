import os
import tarfile
import email
import re
import nltk
import urlextract
import numpy as np
import scipy.io as sio
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import PorterStemmer
from html import unescape
from email import parser
from email.policy import default
from six.moves import urllib
from collections import Counter

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")


def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()


# fetch_spam_data()

HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]

print(len(ham_filenames))
print(len(spam_filenames))


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return parser.BytesParser(policy=email.policy.default).parse(f)


ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

print(ham_emails[4].get_content().strip())
print(spam_emails[5].get_content().strip())


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()


def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


print(structures_counter(ham_emails).most_common())
print('\n')
print(structures_counter(spam_emails).most_common())

for header, value in spam_emails[0].items():
    print(header, ":", value)


def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' httpaddr ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)


html_spam_emails = [email for email in spam_emails
                    if get_email_structure(email) == "text/html"]

sample_html_spam = html_spam_emails[7]
print("\nSpam email html sample:\n")
print(sample_html_spam.get_content().strip()[:1000], "...")
print("\nEmail content: \n")
print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")


def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


print(email_to_text(sample_html_spam)[:100], "...")


try:
    stemmer = nltk.PorterStemmer()
    for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
        print(word, "=>", stemmer.stem(word))
except ImportError:
    print("Error: stemming requires the NLTK module.")
    stemmer = None

try:
    url_extractor = urlextract.URLExtract()
    print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))
except ImportError:
    print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
                text = re.sub("[$]+", " dollar ", text)
                text = re.sub("[^\s]+@[^\s]+", " emailaddr ", text)
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " httpaddr ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
                special_chars = [
                    "<", "[", "^", ">", "+", "?", "!", "'", ".", ",", ":",
                    "*", "%", "#", "_", "="
                ]
                for char in special_chars:
                    text = text.replace(str(char), "")
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)


vocab = EmailToWordCounterTransformer().fit_transform(spam_emails)
vocab = sum(vocab, Counter())

list = vocab.most_common(1904)
vocab = []
for (k, v) in list:
    vocab.append(k)

vocab = sorted(vocab)


# SAVE DICTIONARY
i = 0
with open('../data/vocab2.txt', 'w') as f:
    for item in vocab:
        try:
            f.write("%s\t%s\n" % (i, item))
            i += 1
        except:
            print('error')

samples = len(ham_filenames) + len(spam_filenames)

vocabList = open('../data/vocab2.txt', "r").read()
vocabList = vocabList.split("\n")
vocabList_d = {}
for ea in vocabList:
    if ea:
        [value, key] = ea.split("\t")
        vocabList_d[key] = value

print(vocabList_d)
print(email_to_text(spam_emails[0]))


def process_email(email_contents):
    """
    Preprocesses the body of an email and returns a list of indices of the words contained in the email.
    """
    # a - Lower case
    email_contents = email_contents.lower()

    # b - remove html/xml tags
    email_contents = re.sub("<[^>]*>", " ", email_contents).split(" ")
    email_contents = filter(len, email_contents)
    email_contents = ' '.join(email_contents)

    # c - Handle URLS
    email_contents = re.sub("[http|https]://[^\s]*", "httpaddr", email_contents)

    # d - Handle Email Addresses
    email_contents = re.sub("[^\s]+@[^\s]+", "emailaddr", email_contents)

    # e - Handle numbers
    email_contents = re.sub("[0-9]+", "number", email_contents)

    # f - Handle $ sign
    email_contents = re.sub("[$]+", "dollar", email_contents)

    # Strip all special characters
    special_chars = [
        "<", "[", "^", ">", "+", "?", "!", "'", ".", ",", ":",
        "*", "%", "#", "_", "="
    ]
    for char in special_chars:
        email_contents = email_contents.replace(str(char), "")
    email_contents = email_contents.replace("\n", " ")

    # Stem the word
    ps = PorterStemmer()
    email_contents = [ps.stem(token) for token in email_contents.split(" ")]
    email_contents = " ".join(email_contents)

    return email_contents


def find_word_indices(processed_email, vocabList_d):
    # Process the email and return word_indices

    word_indices = []

    for char in processed_email.split():
        if len(char) > 1 and char in vocabList_d:
            word_indices.append(int(vocabList_d[char]))

    return word_indices


def email_features(word_indices, vocabList_d):
    """
    Takes in a word_indices vector and  produces a feature vector from the word indices.
    """
    n = len(vocabList_d)

    features = np.zeros((n, 1))

    for i in word_indices:
        features[i] = 1

    return features


def transform_email_to_features(email_contents, vocabList_d):
    # print(email_contents)
    processed_email = process_email(email_contents)
    word_indices = find_word_indices(processed_email, vocabList_d)
    features = email_features(word_indices, vocabList_d)

    return features


# train
X = []
Y = []

print(len(spam_emails))
print(len(ham_emails))

for i in range(400):
    sp = email_to_text(spam_emails[i])
    if sp:
        a = transform_email_to_features(sp, vocabList_d)
        X.append(a.flatten())
        Y.append(1)
for i in range(2000):
    em = email_to_text(ham_emails[i])
    if em:
        X.append(transform_email_to_features(em, vocabList_d).flatten())
        Y.append(0)

sio.savemat('../data/myTrain.mat', {'X': X, 'y': Y})

# test
X = []
Y = []

for i in range(401, 500, 1):
    sp = email_to_text(spam_emails[i])
    if sp:
        a = transform_email_to_features(sp, vocabList_d)
        X.append(a.flatten())
        Y.append(1)
for i in range(2001, 2500, 1):
    em = email_to_text(ham_emails[i])
    if em:
        X.append(transform_email_to_features(em, vocabList_d).flatten())
        Y.append(0)

sio.savemat('../data/myTest.mat', {'Xtest': X, 'ytest': Y})
