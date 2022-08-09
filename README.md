# Repo for HW4 - NLP Summer 2022 @ Columbia University

## Cloud Setup
- [Cloud setup instructions](cloud_setup.md): Broad outline of steps to redeem credits, setup VM, and install dependencies through notebook.
- [Cloud setup tutorial on Youtube](https://www.youtube.com/watch?v=Zj0DxBioBq8)
- [HW4 Notebook](Assingment4.ipynb): Notebook to install dependencies and run boilerplate.


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
