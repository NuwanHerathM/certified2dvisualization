import os
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import statistics
import numpy as np

class Visualization:

    @staticmethod
    def merge(intervals):
        """Return the union of the intervals."""

        iterator = iter(intervals)
        res = []
        last = (-1, -1)
        while True:
            try:
                item = next(iterator)
            except StopIteration:
                break  # Iterator exhausted: stop the loop
            else:
                if last[1] == item[0]:
                    last = (last[0], item[1])
                else:
                    res.append(item)
                    last = item
        return res

    @staticmethod
    def show(intervals, poly, default_file, use_clen, use_idct, use_dsc, use_cs, hide, save, freq, time_distibution, n, grid, idct2d):
        if (not hide or save):
            # Drawing of the polynomial
            fig1 = plt.figure(dpi=600)
            base = os.path.basename(poly)
            eval_method = "clenshaw" * use_clen + "idct" * use_idct + "horner" * (1 - max(use_clen, use_idct))
            isol_method = "interval" * (1 - max(use_dsc, use_cs)) + "dsc" * use_dsc + "gm" * use_cs
            fig1.canvas.set_window_title(f"{os.path.splitext(base)[0]}: n={n - 1}, " + eval_method + ", " + isol_method)

            ax1 = fig1.add_subplot(111, aspect='equal')
            ax1.tick_params(axis='both', which='minor', labelsize=10)

            segments = []
            colors = []
            for i in range(n):
                for e in Visualization.merge(intervals[i]):
                    segments.append([(grid.xs[i], grid.ys[e[0]]), (grid.xs[i], grid.ys[e[1]])])
                    colors.append((not e[2], 0, 0, 1))

            lc = mc.LineCollection(segments, colors=colors, linewidths=0.1)
            ax1.add_collection(lc)
            plt.xlim(grid.x_min, grid.x_max)
            plt.ylim(grid.y_min, grid.y_max)

            if (poly == default_file):
                # draw a circle if the default file is used
                circle = plt.Circle((0, 0), 2, color='r', fill=False)
                ax1.add_artist(circle)

            if save:
                filename = os.path.splitext(os.path.basename(poly))[0]
                plt.savefig(f"../output/{filename}_{n-1}_{eval_method}_{isol_method}.png", bbox_inches='tight')
                plt.savefig(f"../output/{filename}_{n-1}_{eval_method}_{isol_method}.pdf", bbox_inches='tight', dpi=1200)
            
            # Frequency analysis
            if freq:
                (distr, res) = time_distibution
                x = np.linspace(distr.min(), distr.max(), res.frequency.size)
                fig2 = plt.figure(figsize=(5, 4))

                ax2 = fig2.add_subplot(1, 1, 1)

                ax2.bar(x, res.frequency, width=res.binsize)

                fig2.canvas.set_window_title('Relative frequency histogram')
                vert_mean = plt.axvline(x=distr.mean(), figure=fig2, color='k')
                vert_median = plt.axvline(x=statistics.median(distr), figure=fig2, color='r')
                plt.legend([vert_mean, vert_median], ['mean', 'median'])

                ax2.set_xlim([x.min(), x.max()])
                plt.xlabel('time (s)')

            if not hide:
                plt.show()
