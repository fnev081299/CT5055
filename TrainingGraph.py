# Imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# graphing purposes ----------------------------------------------------------------------------------------------------
# Plots graph of the training
def plot_graph(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        color="black",
        x_jitter=.5,
        line_kws={'color': 'blue'}
    )

    ax.set(xlabel='Episodes/Games',
           ylabel='Scoring')

    plt.title('Snake AI Training')
    plt.savefig("Training.png")
    plt.show()
