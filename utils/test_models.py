import os
import numpy as np
import pandas as pd
import seaborn as sns


CONTEXT = 'paper'       # or 'talk' to adapt scaling of seaborn plots


def comprehensive_testing(model, text_embedding, text_preprocessor, claims, labels, folder):
    sns.set_context(CONTEXT)

    # tokenize if necessary
    if not isinstance(claims[0], list):
        claims = text_preprocessor.tokenize_raw_sentences(claims)

    token_probabilities = get_probabilities(model, text_embedding, text_preprocessor, claims)
    investigate_statements(token_probabilities, claims, labels, folder)
    investigate_probabilities(token_probabilities, labels, folder)


def get_probabilities(model, text_embedding, text_preprocessor, claims):
    is_fine = np.argwhere(np.array([text_embedding.sample_ok(sent) for sent in claims])).flatten()

    claims_suitable = [claims[i] for i in is_fine]
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


def investigate_statements(all_probabilities, claims, labels, folder):
    n_statements = 10
    i_statements = np.arange(n_statements) * (len(claims) // (n_statements - 1))
    # go through all statements and analyse them
    for i in i_statements:
        # determine x coordinates based on how long the tokens is, so that we can write the sentence
        # continuously on the x-axis and print the probabilities
        token_lengths = np.array([len(token) for token in claims[i]])
        # add 1 for the spaces
        shifts = np.cumsum(np.insert(token_lengths[:-1] + 1, 0, 0))
        x_vals = shifts + (token_lengths + 1) / 2
        plot_data = pd.DataFrame({'x': x_vals, 'y': all_probabilities[i]})

        sns.set_style('whitegrid')
        sns.despine(left=True)
        g = sns.lineplot(x='x', y='y', data=plot_data)
        g.text(1, 0.9 * all_probabilities[i].max(), labels[i])
        g.set_xticks(x_vals)
        g.set_xticklabels(labels=claims[i])
        g.set_xlim([0, np.sum(token_lengths + 1)])
        g.set_xlabel("")
        g.set_ylabel('Probability')
        g.set_title('Token Probabilities')
        g.figure.set_size_inches(6.5, 4.5)
        fig = g.get_figure()
        fig.savefig(os.path.join(folder, 'statement_' + str(i) + '.png'))


def investigate_probabilities(all_probabilities, labels, folder, normalize=True):
    # analyze probability distributions
    plot_data = pd.DataFrame()

    for i, probs in enumerate(all_probabilities):
        n = probs.size
        div = n - 1 if normalize else 1.0
        new_data = pd.DataFrame({'x': np.arange(n) / div,
                                 'y': np.sort(probs),
                                 'group': np.repeat(i, n),
                                 'label': np.repeat(labels[i], n)})
        plot_data = pd.concat((plot_data, new_data))

    sns.set_style('whitegrid')
    g = sns.lineplot(x='x', y='y', hue='label', units='group', drawstyle='steps-post', data=plot_data, estimator=None)
    g.set_axis_labels('x', 'y', labelpad=10)
    g.legend.set_title('Token Probabilities')
    g.legend.set_title('Label')
    g.figure.set_size_inches(6.5, 4.5)
    fig = g.get_figure()
    fig.savefig(os.path.join(folder, 'probability_plot.png'))
