# Project 2: Classification
## Buy on the site or note

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [scipy](https://www.scipy.org/)

### Instructions

This is a simple example of classification explained in an Alura video in the Machine Learning course, introduction to classification.

This is a project to identify whether a person will buy on your site or not.

The algorithm is working well, while the hit_rate_base result is 50%, the hit_rate result is 65%, showing that this model works better than the traditional form.

### Code

Code is provided in the file `classify_matching.py`.

In a terminal or command window, navigate to the top-level project directory `pigs_or_dogs/` (that contains this README) and run one of the following commands:

```python classify_acess.py```

### Data

Data is provided in the file 'search.csv'.

### Need help improvement 

In the terminal, when I run the file it displays the following message:

classify_matching.py:29: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
  train_data = X[:size_of_training]
classify_matching.py:30: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
  train_marcations = Y[:size_of_training]
classify_matching.py:32: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
  test_data = X[-size_of_test:]
classify_matching.py:33: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
  test_marcations = Y[-size_of_test:]

