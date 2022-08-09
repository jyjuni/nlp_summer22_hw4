# Lexical Substitution Task
Repo for HW4 - NLP Summer 2022 @ Columbia University

## Cloud Setup
- [Cloud setup instructions](cloud_setup.md): Broad outline of steps to redeem credits, setup VM, and install dependencies through notebook.
- [Cloud setup tutorial on Youtube](https://www.youtube.com/watch?v=Zj0DxBioBq8)
- [HW4 Notebook](Assingment4.ipynb): Notebook to install dependencies and run boilerplate.

## Introduction

In this assignment you will work on a lexical substitution task, using, WordNet, pre-trained Word2Vec embeddings, and BERT. This task was first proposed as a shared task at SemEval 2007 Task 10 (Links to an external site.). In this task, the goal is to find lexical substitutes for individual target words in context. For example, given the following sentence:

"Anyway , my pants are getting tighter every day ." 

the goal is to propose an alternative word for tight, such that the meaning of the sentence is preserved. Such a substitute could be constricting, small or uncomfortable.

In the sentence

"If your money is tight don't cut corners ." 

the substitute small would not fit, and instead possible substitutes include scarce, sparse, limitited, constricted. You will implement a number of basic approaches to this problem and compare their performance.

## How to
**run**:
```python
python lexsub_main.py lexsub_trial.xml  > smurf.predict
```
**evaluate**:
```python
perl score.pl smurf.predict gold.trial
```
Feel free to open an issue in this repository if there's something that's missing or can be improved. Pull requests are accepted as well if you think you can improve anything :)
