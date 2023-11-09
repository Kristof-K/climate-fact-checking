Check whether models can learn simple grammar: using pronouns and the correct
conjugation of be

Note that masked prediction cannot be perfect:

Seeing '<MASK> are' there are three equally valid options:
'you are', 'we are', 'they are'
A good model should give all of them high probabilities.

E.g., one model run: 'you are' (0.286), 'we are' (0.311)