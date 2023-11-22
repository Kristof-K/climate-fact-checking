from datasets import load_dataset
from unidecode import unidecode
import re

SEARCH_WORD = 'climate'
MIN_WORD_COUNT = 7
MIN_WORD_DENSITY = 1.0 / 100.0


def get_climate_articles():
    all_articles = load_dataset("wikipedia", "20220301.en")
    # check for the earliest (? --> lazy matching) of 'See also', 'References', 'References and notes')
    # use re.DOTALL so that '.' also matches newlines
    throw_away = re.compile(r'(.*?)(See also\s*\n|References\s*\n|References and notes\s*\n)', flags=re.DOTALL)

    for i in range(all_articles['train'].shape[0]):
        curr_article = all_articles['train'][i]
        count_climate = curr_article['text'].lower().count(SEARCH_WORD)
        approx_words = len(curr_article['text']) / 7
        #
        if count_climate >= MIN_WORD_COUNT and count_climate / approx_words >= MIN_WORD_DENSITY:
            new_text = unidecode(curr_article['text'])

            crop_text = throw_away.match(new_text)
            if crop_text is None:
                yield new_text
            else:
                yield crop_text.group(1)    # return everything before the matched pattern


def find_climate_articles():
    all_articles = load_dataset("wikipedia", "20220301.en")

    count_total = 0
    for i in range(all_articles['train'].shape[0]):
        curr_article = all_articles['train'][i]
        count_climate = curr_article['text'].lower().count(SEARCH_WORD)
        count_chars = len(curr_article['text'])
        approx_words = count_chars / 7
        #
        if count_climate >= MIN_WORD_COUNT and count_climate / approx_words >= MIN_WORD_DENSITY:
            print(f'[{count_climate} / {approx_words:.2f}]: {all_articles["train"][i]["title"]}')
            count_total += 1
    print(f'Totoal number of articles: {count_total}')


if __name__ == '__main__':
    find_climate_articles()
