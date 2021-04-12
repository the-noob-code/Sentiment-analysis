from cleanup import *

def train(df):
    print("Training the classifier ...")
    N = df['text'].size

    V = vocab(df)
    N_V = len(V)

    Neg = df[df['label'] == '0']
    Pos = df[df['label'] == '1']
    Neg_size = Neg['text'].size
    Pos_size = Pos['text'].size

    D_POS = wordcount(Pos)
    D_NEG = wordcount(Neg)

    pri_pos_prob = np.log(Pos_size/N)
    pri_neg_prob = np.log(Neg_size/N)

    prob_pos = {}
    prob_neg = {}
    for word in V:
        prob_pos[word] = np.log((D_POS[word]+1)/(Pos_size+N_V))
        prob_neg[word] = np.log((D_NEG[word]+1)/(Neg_size+N_V))
    return (V, pri_pos_prob, pri_neg_prob, prob_pos, prob_neg)

def test_helper(line,V,pri_pos_prob, pri_neg_prob, prob_pos, prob_neg):
    test_pos = pri_pos_prob
    test_neg = pri_neg_prob
    for word in line:
         if word in V:
             test_pos += prob_pos[word]
             test_neg += prob_neg[word]
    if test_pos>test_neg:
        return '1'
    else:
        return '0'

def test(df, V, pri_pos_prob, pri_neg_prob, prob_pos, prob_neg):
    print("Predicting classes for input data ...")
    result = []
    for line in df['text'].values:
        result.append(test_helper(line,V, pri_pos_prob, pri_neg_prob, prob_pos, prob_neg))
    return result
