# COLX-585 projects

### General guidelines about using other people's code
Feel free to use other people's code (assuming appropriate license), but you ``M U S T``:
* Credit other people, and provide documentation/references/links to their repos
* Understand how the code works
* Have some other type of contribution(s). In other words, it is not sufficient for your project to just run someone else's code on the data and submit your results. Your other contribution(s) could be doing analysis of your results, comparisons of different methods, critique of the methods, deriving useful insights and interpretations of project outcome, etc. Please read milestone 1 *very carefully* for more information.
* If you are not sure or have questions, please feel free to the instructor and/or TAs during their office hours. 

---
# COVID-19 PROJECTS
This is a cluster of projects around the theme of COVID-19. You could pick one of these, or perhaps they will inspire a new project idea you may like to propose for a project. For the following, we will provide a somewhat detailed description for the first two projects, as a way to walk you through what could be done. The same lines of thought, procedures, etc. could be extended to the rest of the projects within the COVID-19 Projects Cluster. 

## COVID-19 Sentiment Analysis
* Sentiment Analysis of COVID-19 social media data within (1) a certain location and/or (2) about certain topics.
The ``location``, for example, can involve North American, European, and Asian countries or cities. The ``topics``, for example, could include health, family, politics, school, etc.
* For this project, you could collect Twitter data from one or more countries or cities. Some people use hashtags to do that, or put a bounding box around a certain city or country for a given period of time. 
* The project could involve: (1) building a deep learning sentiment analysis system using some of the data (if you decide to annotate some of it) or using already labeled and available sentiment analysis data and (2) running the system on the collected Twitter data to perform some analysis. 
* The project could provide some insights. For example, what are people's sentiments toward politicians within certain locations? How do people feel about basic services, supply (e.g., grocery), health care system? There will be many such questions the project can explore.

* Notes: 
- If you need some data to get started, please ask and we will make available some general, English language Twitter dataset from the past 2-3 months (~ 1 million tweets, with no specific location), but you may want to augment the data or collect your own.
- If you need a labeled sentiment analysis dataset to train on, please ask and we will provide some English Twitter data.
- It will probably make your work interesting if you apply on more than one language (2 may be enough, but it's your decision)

## COVID-19 Racism & Hate Speech
* In Vancouver there has been a substantial increase in hate crimes, particularly anti-Asian crimes, that have occured since the start of the COVID-19 pandemic, with police accounts of anti-Asian crimes rising from 12 in 2019 to 98 in 2020 [(CBC)](https://www.cbc.ca/news/canada/british-columbia/bc-hate-crimes-vancouver-police-report-premier-1.5919015). It goes without saying that there is a lot of racism and hate speech associated with COVID-19. This type of toxic language can trigger harmful real-world events, and so this project can seek to detect when such language is used on social media. Since last year much work has been done to understand this toxic rhetoric on social media, and projects such as [CLAWS'S COVID-HATE](http://claws.cc.gatech.edu/covid) ([see also the paper](https://arxiv.org/pdf/2005.12423.pdf) and [papers which cite it](https://scholar.google.ca/scholar?cites=16591943361821068954&as_sdt=2005&sciodt=0,5&hl=en)) have assembled corpora of such posts, as well as performed a significant amount of research on the topic. In choosing this project, you should aim to ask and answer a question not originally addressed by the CLAWS project, but which might be answerable through their *COVID-HATE* dataset (a dataset of annotated tweets related to COVID-19, either hateful, anti-hate, or neutral). 
Potential approaches:
*  Profile the geographic extent of anti-Asian hate speech in Canada during the available time period.
* Collect more recent data and apply a similar classification technique to evaluate current levels of anti-Asian hate speech, as well as counter-speech.
* Investigate a relationship between anti-Asian hate speech and noteable events, for instance lockdowns.

## COVID-19 Misinformation 
* Coronavirus disease 2019 (COVID-19) has widely impacted the lives of people around the world. Meanwhile, misinformation related to COVID-19 is spread rapidly on social media. Such misinformation has misled individuals, disturbed society, and caused negative health impacts. Hence, it is necessary to develop a system to automatically identify misinformation on social media. 
* [WNUT-2020 Task2](https://competitions.codalab.org/competitions/25845) ([Nguyen et al. (2020)](https://www.aclweb.org/anthology/2020.wnut-1.41/)) is a shared task to identify of informative COVID-19 English Tweets. 
* Nguyen et al. provide total of 10K annotated tweets developed. Each tweet is annotated with a label from set {``informative``, ``uninformative``}.
* The dataset is split into 70% Train, 10% Dev, and 20% Test. You can download the dataset from the [Github](https://github.com/VinAIResearch/COVID19Tweet)



---

# Several Projects


## Crash Course in ASR with Wav2Vec2
Automatic Speech Recognition is now in the realm of every-day technology with Google Home, Siri, and Alexa able to recognize fairly sophisticated requests that would have seemed implausible 20 years ago. One recent advance, Facebook AI Research’s Wav2Vec2, aims to make the process of building ASR models simpler by partially training with a self-supervised training process learning directly from raw-audio rather than transcribed audio. This allows for higher quality models without needing as much data. To make matters even better, the model is now supported by Huggingface’s Transformers library, with access to pre-trained models to quickly speed up the task of building fine-tuned models. Potential Ideas:
* Build a model for an unsupported language by fine-tuning the Multilingual Pre-trained model (See this starter code: [XLSR-Wav2Vec2](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb)
* Evaluate performance of one of the trained models on specific dialects, for instance evaluate the English Wav2Vec2 models on different [dialects of English](https://www.openslr.org/83/) 

## Generated text detection: Project 1
* Text generative models (TGMs) excel in producing text that matches the style of human language reasonably well. Such TGMs can be misused by adversaries, e.g., by automatically generating fake news and fake product reviews that can look authentic and fool humans. Detectors that can distinguish text generated by TGM from human written text play a vital role in mitigating such misuse of TGMs.
* Distinguish human written text (webtext) from GPT-2 generated text (xl-1542M model) 
* [Dataset](https://github.com/openai/gpt-2-output-dataset) (250K/250K train, 5K/5K valid, 5K/5K test)
* [Baseline classifier](https://github.com/openai/gpt-2-output-dataset/blob/master/baseline.py)
* [Analysis of baseline classifier](https://github.com/openai/gpt-2-output-dataset/blob/master/detection.md)
* Some ideas worth exploring:
  * Use advanced contextualized word representation model like [BERT,T5](https://huggingface.co/transformers/)
  * Use simple bag of word embedding models like [fastText](https://fasttext.cc/)
  * Use other neural classification models from Supervised Learning II
* References
  * [GPT-2 original](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  * [GPT-2 followup](https://arxiv.org/pdf/1908.09203.pdf)

## Generated text detection: Project 2
* This project has similar motivation as Project 1. Additionally, for some security applications, it might be crucial to identify the “author” of text given a piece of text.
* Given a tweet, classify if the tweet is written by Human or RNN or GPT-2 or Others (MC)
* [Dataset](https://www.kaggle.com/mtesconi/twitter-deep-fake-text) 
* [Baseline classifier](https://github.com/openai/gpt-2-output-dataset/blob/master/baseline.py)
* Some ideas worth exploring:
  * Use advanced contextualized word representation model like [BERT,T5](https://huggingface.co/transformers/)
  * Use simple bag of word embedding models like [fastText](https://fasttext.cc/)
  * Use other neural classification models from Supervised Learning II
* References
  * [TweepFake: about Detecting Deepfake Tweets](https://arxiv.org/abs/2008.00036)  
  * [GPT-2 original](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  * [GPT-2 followup](https://arxiv.org/pdf/1908.09203.pdf)
  * [Reverse Engineering Configurations](https://www.aclweb.org/anthology/2020.acl-main.25/)
  * [Authorship Attribution](https://www.aclweb.org/anthology/2020.emnlp-main.673/)

## Machine translation for noisy Reddit text
* Noisy or non-standard input text can cause disastrous mistranslations in most modern Machine Translation (MT) systems, and there has been growing research interest in creating noise-robust MT systems.
* Build a neural machine translation model that translates noisy Reddit post (with typos, grammar errors, code switching and more) from French language to English language
* [Dataset](http://www.statmt.org/wmt19/robustness.html) (19161 train, 886 valid, 1022 test)
* Some ideas worth exploring:
  * Data augmentation using synthetic noise (take natural data and corrupt it); natural noise (Wikipedia errors?)
  * Backtranslation
  * Sub-word models like BPE, WordPiece
  * Use advanced multilingual contextualized word representation model like [mBART,mT5](https://huggingface.co/transformers/)
* References
  * [Dataset paper](https://www.aclweb.org/anthology/D18-1050.pdf)
  * [Related shared task](http://www.statmt.org/wmt19/robustness.html)
  * [Findings of the First Shared Task on Machine Translation Robustness](https://www.aclweb.org/anthology/W19-5303/)
  * [Improving Robustness of Machine Translation with Synthetic Noise](https://www.aclweb.org/anthology/N19-1190.pdf)

## Formality Style Transfer
* [Grammarly Yahoo Answers Formality Corpus (GYAFC)](https://arxiv.org/pdf/1803.06535.pdf) is a dataset for **English formality style transfer**. It contains a total of 110,000 informal-formal sentence pairs. The goal of this task is to generate a formal paraphrase given an informal sentence. [Rao and Tetreault (2018)](https://arxiv.org/pdf/1803.06535.pdf) use the [Yahoo Answers L6 corpus](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l) to extract informal sentences and manually rewrite the informal sentences into formal ones. For example:
  * **Informal sentence:**  Wow , I am very dumb in my observation skills ......
  * **Formal paraphrase:** I do not have good observation skills.

* GYAFC dataset contain two main domains, Entertainment \& Music (E\&M) and Family \& Relationships (F\&R). 
* Rao and Tetreault (2018) randomly divide the informal sentences from E\&M and F\&R into Train, Dev and Test, separately. You can find the data distribution in the [paper](https://arxiv.org/pdf/1803.06535.pdf). To provide higher evaluation sets, the authors collect four formal references for each sentence of Dev and Test sets with restrict annotation rules.
* We already got permission from the authors to share the dataset within the class. Please contact with instructor or TAs to access the dataset if you are interested in this project. 

## Code-switched machine translation
* Code-mixing, the interleaving of two or more languages within a sentence or discourse is ubiquitous in multilingual societies.
* Build a neural machine translation model that translates English language to Hinglish (Hindi and English codeswitched) language
* [Dataset](https://ritual.uh.edu/lince/datasets)
* Some ideas worth exploring:
  * Backtranslation
  * Sub-word models like BPE, WordPiece
  * Use advanced multilingual contextualized word representation model like [mBART,mT5](https://huggingface.co/transformers/)
* References
  * [Related shared task](https://code-switching.github.io/2021#shared-task)
  * [Generate synthetic CS](https://www.researchgate.net/publication/347233848_A_Semi-supervised_Approach_to_Generate_the_Code-Mixed_Text_using_Pre-trained_Encoder_and_Transfer_Learning)

## Jigsaw Multilingual Toxic Comment Classification
* Identify toxic comments from benign or non-toxic comment
* [Data](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/data)
* Some ideas worth exploring:
  * Use advanced contextualized word representation model like [BERT](https://huggingface.co/transformers/)
  * Use simple bag of word embedding models like [fastText](https://fasttext.cc/)
  * Use other neural classification models from Supervised Learning II

## Compare and contrast several NLU/NLG architectures (e.g. BERT, GPT-2) via probing tasks and write the analysis
* A plethora of new NLP models have been proposed, many of which are thought to be opaque compared to their feature-rich counterparts. This has led researchers to analyze, interpret, and evaluate neural networks in novel and more fine-grained ways.
* Use existing or define new probing task to identify the linguistic features encoded by NLU/NLG models
* [Existing probing task](https://github.com/facebookresearch/SentEval/tree/master/data/probing)
* [NLU/NLG models](https://github.com/huggingface/transformers)
* Some ideas worth exploring
  * Analyzing state-of-the-art NLU model like BERT, XLNet, T5 and NLG model like GPT-2, CTRL
  * Contrast the performance with an untrained baseline model from the same family
  * Probe all the layers for each probing task and visualize using heatmap
* References
  * [What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties](https://www.aclweb.org/anthology/P18-1198.pdf)
  * [Linguistic Knowledge and Transferability of Contextual Representations](https://arxiv.org/abs/1903.08855)
  * [Survey](https://transacl.org/ojs/index.php/tacl/article/view/1570)
