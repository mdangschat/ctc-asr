import os
import matplotlib
from matplotlib import rc


def pyplot_display(func):
    """Provides decorator for `matplotlib.pyplot` plots.
    It only uses `show()` display or PyCharm remote has been found.
    Else the plot is being saved to /tmp/<func name>.png.

    Note:
        Wrapped methods need the imported pyplot argument as their first argument.
        Wrapped methods need to return the `fig = plt.figure(...)` argument after completion.

    Args:
        func (function): The plot function.

    Returns:
        function: The wrapped function.
    """
    def wrapper(*args, **kwargs):
        rc('font', **{'family': 'serif',
                      'serif': ['DejaVu Sans'],
                      'size': 13
                      })
        rc('text', usetex=True)

        # Setup plot output based on if a display is available or not.
        display = 'DISPLAY' in os.environ or \
                  all(var in os.environ for var in ['PYCHARM_HOSTED', 'PYCHARM_MATPLOTLIB_PORT'])
        if display:
            from matplotlib import pyplot as plt
        else:
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt

        fig = func(plt, *args, **kwargs)  # Call wrapped function.

        # Display or save the plot.
        if display:
            plt.show()
            # print('plt.show()')
        else:
            path = '/tmp/{}.png'.format(func.__name__)
            fig.savefig(path)
            print('Plot saved to: {}'.format(path))

    return wrapper
