import matplotlib.pyplot as plt
import Animator as thk_animator

num_epochs=100
animator = thk_animator.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
animator.add(1, [0.9,0.7,0.3])
animator.add(10, [0.6,0.8,0.5])
animator.add(15, [0.5,0.7,0.7])
animator.add(20, [0.3,0.8,0.9])

plt.show()
