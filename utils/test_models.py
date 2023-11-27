import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain


CONTEXT = 'paper'       # or 'talk' to adapt scaling of seaborn plots


def comprehensive_testing(model, text_embedding, text_preprocessor, claims, labels, folder):
    sns.set_context(CONTEXT)

    # tokenize if necessary
    if not isinstance(claims[0], list):
        claims = text_preprocessor.tokenize_raw_sentences(claims)

    token_probabilities = get_probabilities_batched(model, text_embedding, text_preprocessor, claims, batch_size=10)

    n_statements = 10
    i_statements = np.arange(n_statements) * (len(claims) // (n_statements - 1))
    # go through all statements and analyse them
    for i in i_statements:
        while np.all(np.isnan(token_probabilities[i])):
            i += 1
        most_likely_words, probs = get_most_likely_word_and_prob(model, text_embedding, text_preprocessor, claims[i])
        plot_probability_as_line(token_probabilities[i], claims[i], labels[i], '_line_' + str(i), folder)
        plot_probability_as_bars(token_probabilities[i], claims[i], probs, most_likely_words, labels[i],
                                 '_bar_' + str(i), folder)

    investigate_probabilities(token_probabilities, labels, folder)
    investigate_probabilities(token_probabilities, labels, folder, normalize=False)


def get_probabilities(model, text_embedding, text_preprocessor, claims):
    is_fine = np.argwhere(np.array([text_embedding.sample_ok(sent) for sent in claims])).flatten()

    claims_suitable = [claims[i] for i in is_fine]
    if not claims_suitable:     # no claim is suitable
        return [np.repeat(np.nan, len(claim)) for claim in claims]

    masked_statements, masked_words = text_preprocessor.get_masked_word_tokens(claims_suitable)

    x_num = text_embedding.encode_x(masked_statements)
    token_probabilities = model.get_token_probabilities(x_num, masked_words)

    all_probabilities = []
    c = 0
    for i in range(len(claims)):
        if i in is_fine:
            all_probabilities.append(token_probabilities[c:(c + len(claims[i]))])
            c += len(claims[i])
        else:
            all_probabilities.append(np.repeat(np.nan, len(claims[i])))

    return all_probabilities


def get_probabilities_batched(model, text_embedding, text_preprocessor, claims, batch_size=10):
    # model will see batch_size * sentence lengths due to the unfolding of masking!
    n_claims = len(claims)
    claim_batches = [
        [claims[i] for i in range(start, min(start + batch_size, n_claims))] for start in range(0, n_claims, batch_size)
    ]
    t_probs = [
        get_probabilities(model, text_embedding, text_preprocessor, claim_batch) for claim_batch in claim_batches
    ]
    token_probabilities = list(chain.from_iterable(t_probs))  # append all the list next to each other
    return token_probabilities


def get_most_likely_word_and_prob(model, text_embedding, text_preprocessor, claim):
    if not text_embedding.sample_ok(claim):
        return '', -1.0

    masked_statements, masked_words = text_preprocessor.get_masked_word_tokens([claim])
    x_num = text_embedding.encode_x(masked_statements)
    res = [model.get_most_likely_words(x_num[[i]], n_beams=1) for i in range(x_num.shape[0])]
    # unzip words and probabilities
    unzipped = list(zip(*res))
    return list(chain.from_iterable(unzipped[0])), np.concatenate(unzipped[1])     # return words and probabilities


def plot_probability_as_line(probabilities, claim, label, add_name, folder):
    # determine x coordinates based on how long the tokens is, so that we can write the sentence
    # continuously on the x-axis and print the probabilities
    token_lengths = np.array([len(token) for token in claim])
    # add 1 for the spaces
    shifts = np.cumsum(np.insert(token_lengths[:-1] + 1, 0, 0))
    x_vals = shifts + (token_lengths + 1) / 2
    plot_data = pd.DataFrame({'x': x_vals, 'y': probabilities})

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.set_style('whitegrid')
    sns.despine(left=True)
    g = sns.lineplot(x='x', y='y', data=plot_data, ax=ax)
    g.text(1, 0.9 * probabilities.max(), label)
    g.set_xticks(x_vals)
    g.set_xticklabels(labels=claim)
    g.set_xlim([0, np.sum(token_lengths + 1)])
    g.set_xlabel("")
    g.set_ylabel('Probability')
    g.set_title('Token Probabilities')
    fig.savefig(os.path.join(folder, 'statement' + add_name + '.png'))
    fig.clear()
    plt.close()


def plot_probability_as_bars(probabilities, claim, most_likely_p, most_likely_w, label, add_name, folder):
    y_numbers = [str(i) for i in range(len(claim))]
    plot_data = pd.DataFrame({'x': probabilities, 'y': y_numbers, 'x2': most_likely_p})
    col_one = '#359DE3'
    col_two = '#BB2C0B'

    fig, ax1 = plt.subplots(figsize=(4.5, 9))
    sns.set_style('whitegrid')
    sns.barplot(x='x2', y='y', color=col_two, data=plot_data, ax=ax1)
    g = sns.barplot(x='x', y='y', color=col_one, data=plot_data, ax=ax1)
    g.set_ylabel('')
    ax1.tick_params(axis='y', labelcolor=col_one)
    ax1.set_yticklabels(claim)
    # add another y-axis for showing the most likely words
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_yticklabels(most_likely_w)
    ax2.set_ylabel('')
    ax2.tick_params(axis='y', labelcolor=col_two)

    g.set_xlabel('Probability / Score')
    g.set_title(f'Token Scores ({label})')
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'statement' + add_name + '.png'))
    fig.clear()
    plt.close()


def plot_all_prob_curves(plot_data, folder, facet=False, add_name=''):
    labels = plot_data['label'].unique()
    n_facet = 2 if facet else 1
    fig_size = (10, 9) if facet else (6, 4.5)

    fig, ax = plt.subplots(n_facet, n_facet, figsize=fig_size)
    sns.set_style('whitegrid')

    for i in range(n_facet**2):
        if facet:
            my_plot_data = plot_data.loc[plot_data['label'] == labels[i]]
            ax = ax.flatten()
        else:
            my_plot_data = plot_data
            ax = [ax]       # to be able to denote it with index 0

        g = sns.lineplot(x='x', y='y', hue='label', units='group', drawstyle='steps-post', data=my_plot_data,
                         ax=ax[i], estimator=None)
        g.set_xlabel('x')
        g.set_ylabel('y')
        if facet:
            g.set_title(labels[i])
        else:
            g.legend().set_title('Label')

    fig.suptitle('Token Probabilities')
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'probability_plot' + add_name + '.png'))
    fig.clear()
    plt.close()


def investigate_probabilities(all_probabilities, labels, folder, normalize=True):
    # analyze probability distributions
    plot_data = pd.DataFrame()

    for i, probs in enumerate(all_probabilities):
        if np.all(np.isnan(all_probabilities[i])):
            continue
        n = probs.size
        div = n - 1 if normalize else 1.0
        new_data = pd.DataFrame({'x': np.arange(n) / div,
                                 'y': np.sort(probs),
                                 'group': np.repeat(i, n),
                                 'label': np.repeat(labels[i], n)})
        plot_data = pd.concat((plot_data, new_data))

    add_name = '' if normalize else '_unscaled'

    plot_all_prob_curves(plot_data, folder, add_name=add_name)
    plot_all_prob_curves(plot_data, folder, facet=True, add_name=add_name+'_facetted')
