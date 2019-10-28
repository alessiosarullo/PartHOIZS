import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from lib.bbox_utils import rescale_masks_to_img


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_mat(mat, xticklabels, yticklabels, x_inds=None, y_inds=None, alternate_labels=True,
             axes=None, vrange=None, cbar=True, bin_colours=False, grid=False, plot=True, title=None, log=False,
             neg_color=None, zero_color=None, cmap='jet'):
    lfsize = 8
    if axes is None:
        plt.figure(figsize=(16, 9))
        ax = plt.gca()
    else:
        ax = axes

    mat = mat.copy()
    if vrange is None:
        mat_max = mat[~np.isinf(mat) & ~np.isnan(mat)].max()
        mat_min = mat[~np.isinf(mat) & ~np.isnan(mat)].min()
        if 0 <= mat_min and mat_max <= 1:
            vrange = (0, 1)
        else:
            vrange = (mat_min, mat_max)

    num_colors = 5 if bin_colours else 256
    cmap = plt.get_cmap(cmap, lut=num_colors)
    if neg_color:
        cmap.set_under(np.array(neg_color))
        vrange = (0, vrange[1])
    if zero_color:
        cmap.set_over(np.array(zero_color))
        mat[mat == 0] = 10 * (vrange[1] + 1)

    if log:
        vrange = (max(vrange[0], 1), vrange[1])
        if neg_color:
            mat[mat < 0] = 1e-5
        mat_ax = ax.matshow(mat, cmap=cmap, norm=LogNorm(vmin=vrange[0], vmax=vrange[1]))
    else:
        mat_ax = ax.matshow(mat, cmap=cmap, vmin=vrange[0], vmax=vrange[1])

    if cbar:
        plt.colorbar(mat_ax, ax=ax,
                     # fraction=0.04,
                     pad=0.06,
                     )

    y_tick_labels = [l.replace('_', ' ') for l in yticklabels]
    y_ticks = np.arange(len(y_tick_labels))
    y_inds = y_inds if y_inds is not None else range(len(y_tick_labels))

    maj_ticks = y_ticks[::2]
    maj_tick_labels = ['%s %d' % (lbl, i) for i, lbl in zip(y_inds, y_tick_labels)][::2]
    ax.set_yticks(maj_ticks)
    ax.set_yticklabels(maj_tick_labels)
    ax.tick_params(axis='y', which='major', left=True, labelleft=True, right=True, labelright=False, labelsize=lfsize)

    min_ticks = y_ticks[1::2]
    ax.set_yticks(min_ticks, minor=True)
    if alternate_labels:
        min_tick_labels = ['%d %s' % (i, lbl) for i, lbl in zip(y_inds, y_tick_labels)][1::2]
    else:
        min_tick_labels = ['%s %d' % (lbl, i) for i, lbl in zip(y_inds, y_tick_labels)][1::2]
    ax.set_yticklabels(min_tick_labels, minor=True)
    ax.tick_params(axis='y', which='minor', left=True, labelleft=not alternate_labels, right=True, labelright=alternate_labels, labelsize=lfsize)

    x_tick_labels = [l.replace('_', ' ').strip() for l in xticklabels]
    x_ticks = np.arange(len(x_tick_labels))
    x_inds = x_inds if x_inds is not None else range(len(x_tick_labels))

    maj_ticks = x_ticks[::2]
    maj_tick_labels = ['%d %s' % (i, lbl) for i, lbl in zip(x_inds, x_tick_labels)][::2]
    ax.set_xticks(maj_ticks)
    ax.set_xticklabels(maj_tick_labels, rotation=45, ha='left', rotation_mode='anchor')
    ax.tick_params(axis='x', which='major', top=True, labeltop=True, bottom=True, labelbottom=False, labelsize=lfsize)

    min_ticks = x_ticks[1::2]
    ax.set_xticks(min_ticks, minor=True)
    if alternate_labels:
        min_tick_labels = ['%s %d' % (lbl, i) for i, lbl in zip(x_inds, x_tick_labels)][1::2]
        ax.set_xticklabels(min_tick_labels, minor=True, rotation=45, ha='right', rotation_mode='anchor')
    else:
        min_tick_labels = ['%d %s' % (i, lbl) for i, lbl in zip(x_inds, x_tick_labels)][1::2]
        ax.set_xticklabels(min_tick_labels, minor=True, rotation=45, ha='left', rotation_mode='anchor')
    ax.tick_params(axis='x', which='minor', top=True, labeltop=not alternate_labels, bottom=True, labelbottom=alternate_labels, labelsize=lfsize)

    if title is not None:
        ax.set_title(title)

    if grid:
        ax.grid(which='major', color='k', linestyle='-', linewidth=1)

    plt.tight_layout()
    if plot:
        plt.show()
    return ax


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def postprocess_for_visualisation(boxes_ext, masks, union_boxes, img_infos):
    assert img_infos.shape[0] == 1
    img_infos = img_infos[0].cpu().numpy()
    im_h, im_w = img_infos[:2].astype(np.int)
    im_scale = img_infos[2]
    boxes_ext = boxes_ext.cpu().numpy()
    masks = masks.cpu().numpy()

    box_classes = np.argmax(boxes_ext[:, 5:], axis=1)
    boxes = boxes_ext[:, 1:5] / im_scale
    boxes_with_scores = np.concatenate((boxes, boxes_ext[np.arange(boxes_ext.shape[0]), 5 + box_classes][:, None]), axis=1)
    masks = rescale_masks_to_img(masks, boxes, im_h, im_w)

    union_boxes = union_boxes / im_scale
    return boxes_with_scores, box_classes, masks, union_boxes
