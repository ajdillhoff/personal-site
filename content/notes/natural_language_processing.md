+++
title = "Natural Language Processing"
authors = ["Alex Dillhoff"]
date = 2023-04-23T00:00:00-05:00
tags = ["deep learning", "NLP"]
draft = false
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Introduction](#introduction)
- [Text Preprocessing](#text-preprocessing)
- [Tasks](#tasks)
- [Models](#models)
- [Perplexity](#perplexity)

</div>
<!--endtoc-->



## Introduction {#introduction}

-   Text Preprocessing
    -   Character-level tokenization
    -   Word-level tokenization
    -   Subword tokenization
    -   Stopwords
    -   Batching
    -   Padding
-   Unsupervised Pre-Training
    -   Autoregression
    -   BERT loss
-   Tasks
    -   Text Classification
    -   Named Entity Recognition
    -   Question Answering
    -   Summarization
    -   Translation
    -   Text Generation


## Text Preprocessing {#text-preprocessing}

Text preprocessing is an essential step in NLP that involves cleaning and transforming unstructured text data to prepare it for analysis. Some common text preprocessing techniques include:

-   Expanding contractions (e.g., "don't" to "do not") [7]
-   Lowercasing text[7]
-   Removing punctuations[7]
-   Removing words and digits containing digits[7]
-   Removing stopwords (common words that do not carry much meaning) [7]
-   Rephrasing text[7]
-   Stemming and Lemmatization (reducing words to their root forms) [7]


### Common Tokenizers {#common-tokenizers}

Tokenization is the process of breaking a stream of textual data into words, terms, sentences, symbols, or other meaningful elements called tokens. Some common tokenizers used in NLP include:

1.  ****Whitespace Tokenizer****: Splits text based on whitespace characters (e.g., spaces, tabs, and newlines) [2].
2.  ****NLTK Tokenizer****: A popular Python library that provides various tokenization functions, including word and sentence tokenization[1].
3.  ****SpaCy Tokenizer****: Another popular Python library for NLP that offers a fast and efficient tokenizer, which can handle large documents and is customizable[5].
4.  ****WordPiece Tokenizer****: A subword tokenizer used in models like BERT, which breaks text into smaller subword units to handle out-of-vocabulary words more effectively[3].
5.  ****Byte Pair Encoding (BPE) Tokenizer****: A subword tokenizer that iteratively merges the most frequent character pairs in the text, resulting in a vocabulary of subword units[12].
6.  ****SentencePiece Tokenizer****: A library that provides both BPE and unigram-based subword tokenization, which can handle multiple languages and does not rely on whitespace for tokenization[6].

These tokenizers differ in the way they split text into tokens and handle language-specific considerations, such as handling out-of-vocabulary words, dealing with punctuation, and managing whitespace characters. The choice of tokenizer depends on the specific NLP task and the characteristics of the text data being processed.

Citations:
[1] <https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/>
[2] <https://neptune.ai/blog/tokenization-in-nlp>
[3] <https://towardsdatascience.com/comparing-transformer-tokenizers-686307856955>
[4] <https://www.analyticsvidhya.com/blog/2021/09/essential-text-pre-processing-techniques-for-nlp/>
[5] <https://www.analyticsvidhya.com/blog/2019/07/how-get-started-nlp-6-unique-ways-perform-tokenization/>
[6] <https://www.reddit.com/r/MachineLearning/comments/rprmq3/d_sentencepiece_wordpiece_bpe_which_tokenizer_is/>
[7] <https://www.analyticsvidhya.com/blog/2021/06/must-known-techniques-for-text-preprocessing-in-nlp/>
[8] <https://towardsdatascience.com/top-5-word-tokenizers-that-every-nlp-data-scientist-should-know-45cc31f8e8b9>
[9] <https://www.projectpro.io/recipes/explain-difference-between-word-tokenizer>
[10] <https://www.telusinternational.com/insights/ai-data/article/what-is-text-mining>
[11] <https://towardsdatascience.com/tokenization-for-natural-language-processing-a179a891bad4>
[12] <https://towardsdatascience.com/a-comprehensive-guide-to-subword-tokenisers-4bbd3bad9a7c>
[13] <https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8>
[14] <https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/>
[15] <https://docs.tamr.com/new/docs/tokenizers-and-similarity-functions>
[16] <https://pitt.libguides.com/textmining/preprocessing>
[17] <https://medium.com/@ajay_khanna/tokenization-techniques-in-natural-language-processing-67bb22088c75>
[18] <https://datascience.stackexchange.com/questions/75304/bpe-vs-wordpiece-tokenization-when-to-use-which>
[19] <https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing>
[20] <https://www.tokenex.com/blog/ab-what-is-nlp-natural-language-processing-tokenization/>
[21] <https://hungsblog.de/en/technology/learnings/difference-between-the-tokenizer-and-the-pretrainedtokenizer-class/>
[22] <https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html>
[23] <https://medium.com/nlplanet/two-minutes-nlp-a-taxonomy-of-tokenization-methods-60e330aacad3>
[24] <https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/>
[25] <https://medium.com/@utkarsh.kant/tokenization-a-complete-guide-3f2dd56c0682>
[26] <https://stackoverflow.com/questions/380455/looking-for-a-clear-definition-of-what-a-tokenizer-parser-and-lexers-are>
[27] <https://blog.floydhub.com/tokenization-nlp/>
[28] <https://medium.com/product-ai/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908>
[29] <https://pub.towardsai.net/in-depth-tokenization-methods-of-14-nlp-libraries-with-python-example-297ecdd14c1>
[30] <https://datascience.stackexchange.com/questions/88680/what-is-the-difference-between-countvectorizer-and-tokenizer-or-are-they-the>
[31] <https://www.freecodecamp.org/news/train-algorithms-from-scratch-with-hugging-face/>
[32] <https://exchange.scale.com/public/blogs/preprocessing-techniques-in-nlp-a-guide>
[33] <https://huggingface.co/docs/transformers/tokenizer_summary>
[34] <https://blog.octanove.org/guide-to-subword-tokenization/>
[35] <https://www.enjoyalgorithms.com/blog/text-data-pre-processing-techniques-in-ml/>
[36] <https://www.geeksforgeeks.org/nlp-how-tokenizing-text-sentence-words-works/>


## Tasks {#tasks}


### Text Classification {#text-classification}

Commonly seen in the form of sentiment analysis, where the objective is to classify whether some input text is positive or negative. Document classification, in which documents are identified by their content, is also useful.


### Named Entity Recognition {#named-entity-recognition}

Extract important nouns from a body of text.


### Question Answering {#question-answering}


### Summarization {#summarization}


### Translation {#translation}


### Text Generation {#text-generation}

Generate text from a prompt. This could be in the form of a simple question or some initial dialog. This is also seen in tools like GitHub Co-Pilot to generate code based on contextual code in the same project.


## Models {#models}

Discuss GPT2


## Perplexity {#perplexity}

A measure of confidence of a language model. A naive model may predict a word by randomly selecting any of the \\(N\\) words in its vocabulary. As the model is optimized and the distribution of possible sequences is acquired, the perplexity decreases.
