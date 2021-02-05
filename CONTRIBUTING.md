# How to add jupyter notebooks:
- Make sure all the cells of the jupyter notebook are [blacked](https://github.com/psf/black). You can use [`nb_black`](https://github.com/dnanhkhoa/nb_black) for this.
- All plots should be made using [`SciencePlots`](https://github.com/garrettj403/SciencePlots). First use the following function to estimate the figure size:
```python
def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.
    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'full':
        width_pt = 513.11743
    elif width == 'half':
        width_pt = 242.26653
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
```
 and then use the plot command as follows:
 
 ```python
 
with plt.style.context(['science', 'grid']):
    plt.figure(figsize=set_size(width='half')) # for half page width figures use full for full page width figures
    plt.plot(...) # your code
    plt.savefig("name.pdf",bbox_inches='tight')
 
 ```
