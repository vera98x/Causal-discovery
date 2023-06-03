from pygam import LinearGAM, s, f, PoissonGAM
from pygam.datasets import wage
import matplotlib.pyplot as plt

X, y = wage(return_X_y=True)

## model
gam = PoissonGAM(s(0) + s(1) + f(2))
gam.gridsearch(X, y)


## plotting
plt.figure()
fig, axs = plt.subplots(1,3)

titles = ['year', 'age', 'education']
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    if i == 0:
        ax.set_ylim(-30,30)
    ax.set_title(titles[i])
gam.summary()
plt.show()
