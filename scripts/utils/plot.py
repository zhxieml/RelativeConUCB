import matplotlib
import matplotlib.pyplot as plt

COLORS = {
    "LinUCB": "#1E90FF",
    "ConUCB": "#FFA500",
    "pos+doublebestrelated2": "#8D96A6",
    "pos&neg+doublebestrelated2": "#008000",
    "difference+bestthendiffrelated2": "#DC143C",
    "difference+bestdiffrelated2": "#A47BE3"
}

PATTERNS_NOSHARE = {
    "LinUCB": (COLORS["LinUCB"], "solid", "LinUCB"),
    "ConUCB": (COLORS["ConUCB"], "solid", "ConUCB"),
    "RelativeConUCB_doublebestrelated2_pos": (COLORS["pos+doublebestrelated2"], "solid", "RelativeConUCB-Pos"),
    "RelativeConUCB_doublebestrelated2_pos&neg": (COLORS["pos&neg+doublebestrelated2"], "solid", "RelativeConUCB-Pos\&Neg"),
    "RelativeConUCB_bestdiffrelated2_difference": (COLORS["difference+bestdiffrelated2"], "solid", "RelativeConUCB-Difference"),
    "RelativeConUCB_bestthendiffrelated2_difference": (COLORS["difference+bestthendiffrelated2"], "solid", "RelativeConUCB-Difference (fast)"),
}

PATTERNS_SHAREATTRIBUTE = {
    "ConUCB_share-attribute": (COLORS["ConUCB"], "solid", "ConUCB"),
    "RelativeConUCB_doublebestrelated2_pos_share-attribute": (COLORS["pos+doublebestrelated2"], "solid", "RelativeConUCB-Pos"),
    "RelativeConUCB_doublebestrelated2_pos&neg_share-attribute": (COLORS["pos&neg+doublebestrelated2"], "solid", "RelativeConUCB-Pos\&Neg"),
    "RelativeConUCB_bestdiffrelated2_difference_share-attribute": (COLORS["difference+bestdiffrelated2"], "solid", "RelativeConUCB-Difference"),
    "RelativeConUCB_bestthendiffrelated2_difference_share-attribute": (COLORS["difference+bestthendiffrelated2"], "solid", "RelativeConUCB-Difference (fast)"),
}

LABELS = [pattern[2] for pattern in PATTERNS_NOSHARE.values()]

PATTERNS = dict(list(PATTERNS_NOSHARE.items()) + list(PATTERNS_SHAREATTRIBUTE.items()))

def plot_result_uncertainty(mean, color, label, linestyle, error=None, error_alpha=0.3):
    x = range(0, len(mean), 20)
    p, = plt.plot(x, mean[x], color=color, label=label, linestyle=linestyle)

    if error is not None:
        lowew, higher = mean - error, mean + error
        plt.fill_between(x, lowew[x], higher[x], alpha=error_alpha, facecolor=color)

    return p

def set_plot_rc():
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

def apply_font(ax, font_size):
    ticks = ax.get_xticklabels() + ax.get_yticklabels()

    for t in ticks:
        t.set_fontname("Times New Roman")
        t.set_fontsize(font_size)

    txt = ax.get_xlabel()
    txt_obj = ax.set_xlabel(txt)
    txt_obj.set_fontname("Times New Roman")
    txt_obj.set_fontsize(font_size)

    txt = ax.get_ylabel()
    txt_obj = ax.set_ylabel(txt)
    txt_obj.set_fontname("Times New Roman")
    txt_obj.set_fontsize(font_size)

    txt = ax.get_title()
    txt_obj = ax.set_title(txt)
    txt_obj.set_fontname("Times New Roman")
    txt_obj.set_fontsize(font_size)

    txt = ax.get_title()
    txt_obj = ax.set_title(txt)
    txt_obj.set_fontname("Times New Roman")
    txt_obj.set_fontsize(font_size)