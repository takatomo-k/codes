import numpy as np
def wer(hypothesis, reference):
    d = np.zeros((len(reference)+1)*(len(hypothesis)+1), dtype=np.uint8)
    d = d.reshape((len(reference)+1, len(hypothesis)+1))
    for i in range(len(reference)+1):
        for j in range(len(hypothesis)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(reference)+1):
        for j in range(1, len(hypothesis)+1):
            if reference[i-1] == hypothesis[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    ret=d[len(reference)][len(hypothesis)]/len(reference)
    return round(ret*100,1)




#import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

cc = SmoothingFunction()
def bleu(hypothesis,reference):#
    import warnings;warnings.filterwarnings('ignore')
    #import pdb; pdb.set_trace()
    hypothesis=hypothesis.replace(" EOS","").split()
    reference=reference.replace(" EOS","").split()

    ret=sentence_bleu([reference], hypothesis,smoothing_function=cc.method4)*100
    return round(ret,3)
def mse(hypothesis,reference):
    #import pdb; pdb.set_trace()
    return np.mean((hypothesis-reference)*(hypothesis-reference))


#!/usr/bin/python
#
# By Yusuke Oda
#
# coding: utf-8

import sys
import codecs
import math
from collections import defaultdict

# calculate BLEU score between two corpus
def BLEU(ws_hyp, ws_ref, **kwargs):
    # check args
    max_n = 4
    if 'maxn' in kwargs and isinstance(kwargs['maxn'], int) and kwargs['maxn'] > 0:
        max_n = kwargs['maxn']
    dump = False
    if 'dump' in kwargs and kwargs['dump'] is True:
        dump = True

    # calc N-gram precision
    np = 0
    for n in range(1, max_n+1):
        numer = 0
        denom = 0
        for hyp, ref in zip(ws_hyp, ws_ref):
            if len(hyp) < n:
                continue
            possible_ngram = defaultdict(lambda: 0)
            for i in range(len(ref)-(n-1)):
                possible_ngram[tuple(ref[i:i+n])] += 1
            for i in range(len(hyp)-(n-1)):
                key = tuple(hyp[i:i+n])
                #if key in possible_ngram and possible_ngram[key]  target="1"> 0:
                possible_ngram[key] -= 1
                numer += 1
            denom += len(hyp)-(n-1)
        if dump:
            print('P(n=%d) = %f (%d/%d)' % (n, numer/float(denom), numer, denom))
        np += math.log(numer) - math.log(denom)

    # calc brevity penalty
    sumlen_hyp = sum(len(x) for x in ws_hyp)
    sumlen_ref = sum(len(x) for x in ws_ref)
    bp = min(1.0, math.exp(1.0-sumlen_ref/float(sumlen_hyp)))
    if dump:
        print('BP = %f (HYP:%d, REF:%d)' % (bp, sumlen_hyp, sumlen_ref))

    # calc final score
    bleu = bp * math.exp(np/max_n)
    if dump:
        print('BLEU = %f' % bleu)
    return bleu

# calculate BLEU+1 score between two sentences
def BLEUp1(hyp, ref, **kwargs):
    # check args
    max_n = 4
    if 'maxn' in kwargs and isinstance(kwargs['maxn'], int) and kwargs['maxn'] > 0:
        max_n = kwargs['maxn']

    # calc N-gram precision
    np = 0
    for n in range(1, max_n+1):
        numer = 0 if n == 1 else 1
        possible_ngram = defaultdict(lambda: 0)
        for i in range(len(ref)-(n-1)):
            possible_ngram[tuple(ref[i:i+n])] += 1
        for i in range(len(hyp)-(n-1)):
            key = tuple(hyp[i:i+n])
            if key in possible_ngram and possible_ngram[key] > 0:
                possible_ngram[key] -= 1
                numer += 1
        if numer == 0: # no shared unigram
            return 0
        denom = (0 if n == 1 else 1) + max(0, len(hyp)-(n-1))

        np += math.log(numer) - math.log(denom)

    # calc brevity penalty
    bp = min(1.0, math.exp(1.0-len(ref)/float(len(hyp))))

    # calc final score
    bleu = bp * math.exp(np/max_n)
    return bleu*100
