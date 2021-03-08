
# COLX 531 Neural Machine Translation

An introduction to machine translation, with a focus on neural methods. The course provides a brief history of MT, provides a brief coverage of statistical word-level MT, then delves into more recent neural machine translation (NMT) methods such as sequence-to-sequence models, attention, backtranslation, and multilingual modeling.

Upon completion of this course students will be able to:


* *become* aware of the history of MT and recent advances in NMT
* *identify* the core principles of training and designing NMT models
* *identify* main methods for evaluating MT models 
* *become* aware of the main datasets, tools, competitions in MT
* *apply* deep learning methods to MT problems in critical, creative, and novel ways

---
# Class Meetings

This course occurs during Block 5 in the 2020/21 school year.

Lectures: Tuesdays and Thursdays, 10:30-12:00 PT, [@Zoom](https://ubc.zoom.us/j/62467194367?pwd=VEgveVZBNENncXo1R0lhUG03RHBUUT09) (pwd: 4242)   

Labs: Thursdays, 2-4 PT, Zoom

---

# Assessments

This is an *assignment-based course*. You'll be evaluated as follows:

| Assesment | Weight   | Due Date |  Location          | 
|------   | ------- |--------------------------| ----- |
| Lab Assignment 1	| 25%	| Feb 27, 11:59pm	| Submit to Github |
| Lab Assignment 2	| 25%	| Mar 6, 11:59pm	| Submit to Github |
| Lab Assignment 3	| 25%	| Mar 13, 11:59pm	| Submit to Github |
| Lab Assignment 4	| 25%	| Mar 20, 11:59pm	| Submit to Github |


---
## Teaching Team

| Position           | Name    | Slack Handle | GHE Handle |
| :----------------: | :-----: | :----------: | :--------: |
| Main Instructor | Muhammad Abdul Mageed |    `@Muhammad Mageed`       | `@amuham01`        |
| Teaching Assistant | Peter Sullivan |    `@prsull`       | `@prsull`        |
| Lab Instructor | Ganesh Jawahar | `@ganeshjw` | `@ganeshjw` |

---
## Office Hours: 

| Name           | Time    | Location |
| :----------------: | :----------: | :--------: |
| Muhammad Abdul Mageed |    Wed. 12:00-14:00 PT       | [@Zoom](https://ubc.zoom.us/j/62467194367?pwd=VEgveVZBNENncXo1R0lhUG03RHBUUT09)        |
| Peter Sullivan |   Fri. 11:00-12:00 PT | @Zoom (weekly link on slack)        |
| Ganesh Jawahar | Thurs. 18:00-19:00 PT | [@Zoom](https://ubc.zoom.us/j/67010634869?pwd=RFVZVWxpU3lsZGtmMU9vWXUwWHBBdz09) |

---
# Weekly Content

| Lecture | Topic   | Readings                 | Lecture |  Tutorials |  Assignment |
|------   | ------- |--------------------------| -------- | -------- | -------- |
| 1   |  MT Intro & History | - PK CH01 ; - [Wikipedia Article](https://en.wikipedia.org/wiki/History_of_machine_translation)    |  [[slides](lectures/Lecture-1_MT_2021.pdf)]; [[video](https://ubc.zoom.us/rec/share/cyf7bAWgV-egorNHuCEw0ysEcIXnRDKheYiS_qvUZjCrMV6T0OD2QFQgZd_Bwjxl.AmxrTFVqUCwrGAO-)]; ```pwd:``` A9Fd49m&  | ```GIZA++ & MUSE``` [[Notebook](tutorials/week1/GIZA_tutorial.ipynb)]; [[video](https://youtu.be/dWRjV_BIaqk)] | [T1 and Q1](https://youtu.be/Ttq-wX-Z-bw) [Q2](https://youtu.be/h08C16U-mJM) [Q3](https://youtu.be/ELK6Jh438vg) [Q4](https://youtu.be/ZL8UULnoX4k) [Q5](https://youtu.be/lCgVKfjmWRY) |
| 2   |  Word-Level Translation  | -  PK CH04; - [Word Translation without Parallel Data (MUSE)](https://arxiv.org/pdf/1710.04087.pdf); - Optional: [Context-Aware Cross-Lingual Mapping](https://www.aclweb.org/anthology/N19-1391.pdf)   | [[slides](lectures/Lecture-2_MT_2021.pdf)];  [[extra_slides](lectures/word2vec_2021.pdf)];[[video](https://ubc.zoom.us/rec/share/rzzvKbEPyC2vXl4mG_vSwP9nYwWoNObl7jMM0GA3TnrSzEqcMwE-HdtnciurWUxt.YsC1Uh7IdkOi5F1z)]; ```pwd:``` e1b1^&B. | | |
| 3   |  Seq2Seq Models & Attention I | - [Sequence to sequence learning with neural networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf); [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850); Optional: [Neural machine translation by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473); - [Grammar as a foreign language](https://papers.nips.cc/paper/5635-grammar-as-a-foreign-language.pdf) | [[slides](lectures/Lectures-3-4_MT_2021.pdf)]; [[video](https://ubc.zoom.us/rec/share/tj_SGjESdz97KUXMmtu5OrKt2LOGykb-a1-iB0M7rkP8ep1di8y1Dk7EQnxYymDZ.USVnMXlWKotOB-gB)]; ```pwd:``` Y!bjrT5C | ```Seq2Seq + Evaluation``` [[Notebook](tutorials/week2/seq2seq_tutorial.ipynb)]; [[video](https://youtu.be/qP4uOuUGoh8)] | [T2](https://youtu.be/kIEeGogkpDI) [Q1 and Q2](https://youtu.be/A-FUb1iov-8) [Q3 and Q4](https://youtu.be/IuAIRL0xaNc)  |
| 4   |  Seq2Seq Models & Attention II | Same as last session | [[attn_slide](lectures/additive_attn.png)] [[video](https://ubc.zoom.us/rec/share/-NVQtUqCPb-Q2ZFSlQTDoDQg4wCqcXISz6IkCtqGSGabUdHsSnqa1BR3mRTx_tUM.5JjRI0cf_sML29GW)]; ```pwd:``` J^.^f7?d | | | 
| 5   |   Eval |  - [BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf); - Optional: [METEOR: An automatic metric for MT evaluation with improved correlation with human judgments](https://www.aclweb.org/anthology/W05-0909.pdf) | [[slides](lectures/Lecture_5_MT-2021.pdf)]; [[video](xx)]; ```pwd:``` |  ```Seq2Seq with Attention``` [[Notebook](tutorials/week3/attention_tutorial.ipynb)]; [[video](https://youtu.be/52yXJs4akKc)] ; ```OpenNMT``` [[video](https://youtu.be/wg3U700SSt8)]| [[T3](https://youtu.be/yxRlXSyeZvw)]  [[Q1](https://youtu.be/D7The-Zj7uE)] [[Q2](https://youtu.be/-rW4lM7eKR8)] [[Q3](https://youtu.be/vAFoYW9RvWA)] |
| 6   |   Transformers I  | [Attention is all you need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)   | Recordings: [Transformer 1](https://www.youtube.com/watch?v=IoXR8z-nfYI)  [[video](xx)]; ```pwd:``` | | |
| 7   |   Transformers II  |     | [[video](xx)]; ```pwd:```; [Transformer 2](https://www.youtube.com/watch?v=4Z5gkkCptHI) | ```Transformer``` [[Notebook](tutorials/week4/transformer_tutorial.ipynb)]; [[video](https://youtu.be/bByGuEwWJug)] | | 
| 8   |  Back-Translation & Multilingual NMT | -  [Improving Neural Machine Translation Models with Monolingual Data](https://arxiv.org/pdf/1511.06709.pdf); - [Google's neural machine translation system: Bridging the gap between human and machine translation](https://arxiv.org/pdf/1609.08144.pdf); - Optional: [Investigating Multilingual NMT Representations at Scale](https://arxiv.org/pdf/1909.02197.pdf); - [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00288)   | [[video](xx)]; ```pwd:``` | | | 

# Resources
1. [Philipp Koehn's (PK): Statistical Machine Translation Book](https://www.amazon.com/Statistical-Machine-Translation-Philipp-Koehn-ebook/dp/B00AKE1W9O/ref=sr_1_1?keywords=machine+translation&qid=1581614635&sr=8-1) 
2. [Jurafsky & Martin (JM): Speech and language processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf).

# Additional Resources/Reviews/Papers
* [Neural Machine Translation: A Review](https://arxiv.org/pdf/1912.02047.pdf)
* [Neural Machine Translation](https://arxiv.org/pdf/1709.07809.pdf) (Draft book chapter)
* [From Feature To Paradigm: Deep Learning In Machine Translation](https://jair.org/index.php/jair/article/view/11198/26411) (Survey)
* [Cross-lingual Alignment vs Joint Training: A Comparative Study and A Simple Unified Framework](https://arxiv.org/pdf/1910.04708.pdf)

## Language Models:  
* [JM_CH03](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
* [Brown et al 1992](https://www.aclweb.org/anthology/J92-1002.pdf) 

## Style Transfer: 
* [Multiple-Attribute Text Style Transfer](https://arxiv.org/pdf/1811.00552.pdf)
* [Dear sir or madam, may i introduce the gyafc dataset: Corpus, benchmarks and metrics for formality style transfer](https://arxiv.org/pdf/1803.06535.pdf)
* [Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation Comprehension](https://papers.nips.cc/paper/9284-controllable-unsupervised-text-attribute-transfer-via-editing-entangled-latent-representation.pdf)

---
### Where to find good papers:
* [[ACL Anthology](https://www.aclweb.org/anthology/)]
* [[NeurIPS 2020](https://papers.nips.cc/paper/2020)]; [[NeurIPS 2019](https://nips.cc/Conferences/2019/AcceptedPapersInitial)]; [[NeurIPS 2018](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)]
* [[ICLR 2020](https://iclr.cc/virtual_2020/papers.html?filter=keywords)]; [[ICLR 2019](https://iclr.cc/Conferences/2019/Schedule?type=Poster)]; [[ICLR 2018](https://dblp.org/db/conf/iclr/iclr2018)]
* [[AAAI Digital Library](https://www.aaai.org/Library/conferences-library.php)]

# Policies

Please see the general [MDS policies](https://ubc-mds.github.io/policies/).




