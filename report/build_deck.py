"""Builds the oral deck report/oral_deck.pptx (about 10 minutes, 3 speakers).

Run after notebook 02:  uv run --with python-pptx python report/build_deck.py
Numbers come from the executed notebooks; speaker notes hold the full script.
"""

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
FIG = HERE / "figures"

NAVY = RGBColor(0x14, 0x2B, 0x45)
INK = RGBColor(0x1E, 0x24, 0x2C)
GREY = RGBColor(0x5E, 0x67, 0x73)
PAPER = RGBColor(0xF6, 0xF3, 0xEC)
CARD = RGBColor(0xFF, 0xFF, 0xFF)
GOLD = RGBColor(0xE0, 0x9A, 0x2B)
TEAL = RGBColor(0x2E, 0x7D, 0x8C)
MUTED = RGBColor(0xC9, 0xCF, 0xD6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
TOTAL = 13


def bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def text(slide, s, x, y, w, h, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT,
         italic=False, font=BODY, anchor=MSO_ANCHOR.TOP, spacing=6):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        r = p.add_run()
        r.text = line
        f = r.font
        f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
        f.color.rgb = color
    return tb


def rich(slide, runs, x, y, w, h, size=16, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """One paragraph per list item; each item is a list of (text, bold, color)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(10)
        for t, b, c in para:
            r = p.add_run()
            r.text = t
            r.font.size, r.font.bold, r.font.name = Pt(size), b, BODY
            r.font.color.rgb = c
    return tb


def shape(slide, kind, x, y, w, h, fill, line=None):
    sh = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
    sh.shadow.inherit = False
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = 0.06
    return sh


def badge(slide, n, x, y, d=0.55, fill=GOLD, color=NAVY):
    """The deck's motif: a numbered gold disc."""
    c = shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    r.font.size, r.font.bold, r.font.name = Pt(16 if d < 0.7 else 22), True, HEAD
    r.font.color.rgb = color
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def frame(slide, n, kicker, title, speaker):
    bg(slide, PAPER)
    text(slide, kicker.upper(), 0.7, 0.42, 9, 0.3, 12, True, TEAL)
    text(slide, title, 0.7, 0.72, 11.9, 1.0, 30, True, NAVY, font=HEAD)
    text(slide, f"{speaker}  ·  {n}/{TOTAL}", 9.6, 7.0, 3.1, 0.3, 10, color=GREY, align=PP_ALIGN.RIGHT)
    text(slide, "CART from scratch · Adult census", 0.7, 7.0, 6, 0.3, 10, color=GREY)


def stat(slide, value, label, x, y, w, color=NAVY, size=44):
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, 1.75, CARD)
    text(slide, value, x + 0.25, y + 0.2, w - 0.5, 0.85, size, True, color, font=HEAD)
    text(slide, label, x + 0.25, y + 1.08, w - 0.5, 0.6, 13, color=GREY)


def card(slide, x, y, w, h, head, body, n=None, accent=TEAL):
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, CARD)
    off = 0.3
    if n is not None:
        badge(slide, n, x + 0.25, y + 0.25, 0.5)
        off = 0.95
    text(slide, head, x + off, y + 0.3, w - off - 0.25, 0.45, 17, True, accent)
    text(slide, body, x + 0.3, y + 0.95, w - 0.6, h - 1.1, 14, color=INK)


def table(slide, rows, x, y, col_w, size=14, row_h=0.48, highlight=None):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                               Inches(sum(col_w)), Inches(row_h * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xFD, 0xF3, 0xE1) if i == highlight else CARD)
            c.margin_left = c.margin_right = Inches(0.1)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.text = str(v)
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            f = p.runs[0].font
            f.size, f.name, f.bold = Pt(size), BODY, i == 0 or i == highlight
            f.color.rgb = WHITE if i == 0 else INK
    return t


def style_chart(chart, legend=False):
    chart.font.name, chart.font.size = BODY, Pt(12)
    chart.font.color.rgb = GREY
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xE4, 0xE1, 0xDA)
    va.format.line.fill.background()
    chart.category_axis.format.line.color.rgb = MUTED
    chart.category_axis.tick_labels.font.size = Pt(12)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s.strip()


# 1 · Title ------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
badge(s, "?", 0.8, 0.9, 0.9)
text(s, "Is education the main driver\nof a high income?", 0.8, 2.1, 11.5, 2.0, 46, True, WHITE, font=HEAD, spacing=0)
text(s, "A CART classification tree, built from scratch in NumPy, on the 1994 US census", 0.8, 4.35, 11, 0.6, 20, color=GOLD)
text(s, "Lucas · Manuel · Amine", 0.8, 5.6, 8, 0.4, 18, True, WHITE)
text(s, "Mathematics & Statistics for Machine Learning — A. Diallo · Albert School MSc 1 · 13 November 2026",
     0.8, 6.05, 11.5, 0.4, 13, color=MUTED)
notes(s, """
LUCAS (15 s)
Good afternoon. We are Lucas, Manuel and Amine. Our question is simple: is education really what makes people earn more than 50,000 dollars a year? To answer it, we built a decision tree from scratch and tested it against scikit-learn.
""")

# 2 · Question ---------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
frame(s, 2, "The question", "We started with a belief we wanted to test", "Lucas")
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.7, 1.95, 7.4, 3.1, NAVY)
text(s, "“Is education the main driver of earning more than $50K a year, or do other factors such as "
        "marital status, age and hours worked matter more?”", 1.1, 2.3, 6.6, 2.5, 22, color=WHITE, italic=True, font=HEAD)
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 8.5, 1.95, 4.15, 3.1, CARD)
text(s, "HYPOTHESIS", 8.85, 2.25, 3.5, 0.3, 12, True, TEAL)
text(s, "Education is factor n°1", 8.85, 2.65, 3.5, 1.0, 24, True, NAVY, font=HEAD)
text(s, "The project confirms or rejects it — the answer is on slide 10.", 8.85, 3.85, 3.5, 1.0, 14, color=GREY)
rich(s, [[("Why a decision tree?  ", True, NAVY),
          ("it handles numbers and categories without scaling, its rules are readable, and it can be checked "
           "exactly against scikit-learn — so the answer comes from a model we fully understand.", False, INK)]],
     0.7, 5.45, 12, 1.2, 16)
notes(s, """
LUCAS (45 s)
Our starting hypothesis was that education is the number-one factor. That is the common belief, and it is testable.
Why a decision tree, CART? Three reasons. First, it handles numeric and categorical variables without any scaling. Second, its rules and importances are readable, so it can actually answer a "which factor" question. Third, it is short enough to implement ourselves and to verify exactly against scikit-learn, which the brief requires.
""")

# 3 · Data -------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
frame(s, 3, "The data", "Adult: 48,842 adults from the 1994 US census", "Manuel")
stat(s, "48,842", "people, 14 variables (Kaggle / UCI)", 0.7, 1.95, 3.85)
stat(s, "23.9 %", "earn more than $50K: imbalanced", 4.75, 1.95, 3.85, GOLD)
stat(s, "≈ 6 %", "missing, only in 3 categorical variables", 8.8, 1.95, 3.85, TEAL)
card(s, 0.7, 4.0, 5.95, 2.75, "What we removed",
     "fnlwgt — a census sampling weight, not a trait of the person.\n"
     "education — the same information as education-num; keeping both would split education's importance in two.")
card(s, 6.85, 4.0, 5.8, 2.75, "Missing values are a signal",
     "Only 9 % of people with a missing occupation earn >50K (vs 24 %).\n"
     "So we keep them as a category “Unknown” instead of dropping or imputing them.")
notes(s, """
MANUEL (45 s)
The data is the Adult dataset from Kaggle, originally from the 1994 US census: about 49,000 people, 14 variables, and a binary target, above or below 50K.
Two things matter. First, only 24 percent earn more than 50K: the classes are imbalanced, which will drive our choice of metrics.
Second, missing values: they only affect workclass, occupation and country, and they are informative — people with a missing occupation earn much less. So instead of deleting them we keep "Unknown" as a category.
We removed fnlwgt, a sampling weight, and education as text, because education-num holds the same information.
""")

# 4 · Model ------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
frame(s, 4, "The model", "A tree asks yes/no questions that make groups pure", "Lucas")
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.7, 1.95, 5.6, 4.75, CARD)
text(s, "GINI IMPURITY", 1.05, 2.2, 5, 0.3, 12, True, TEAL)
text(s, "G = 1 − Σ pₖ²", 1.05, 2.6, 5, 0.8, 34, True, NAVY, font=HEAD)
text(s, "Chance that two random people in the node have different labels: 0 if pure, 0.5 for a 50/50 mix.",
     1.05, 3.45, 4.95, 0.9, 14, color=GREY)
text(s, "SCORE OF A SPLIT", 1.05, 4.45, 5, 0.3, 12, True, TEAL)
text(s, "G_split = (n_L/n)·G_L + (n_R/n)·G_R", 1.05, 4.85, 5, 0.6, 18, True, NAVY, font=HEAD)
text(s, "Weighted by size: isolating one pure point is not a good split.", 1.05, 5.55, 4.95, 0.8, 14, color=GREY)
text(s, "Course exercise 3B.1 — 10 clients: 6 repaid, 4 defaulted", 6.7, 1.95, 6, 0.4, 14, True, NAVY)
table(s, [["Question", "Yes", "No", "G_split"],
          ["root (all 10)", "—", "—", "0.48"],
          ["A · income > 30?", "5 R · 0 D", "1 R · 4 D", "0.16"],
          ["B · age > 40?", "3 R · 1 D", "3 R · 3 D", "0.45"]],
      6.7, 2.5, [2.3, 1.3, 1.3, 1.05], 14, highlight=2)
rich(s, [[("Our code finds A on its own ", True, NAVY),
          ("(threshold 33 = midpoint of 28 and 38): impurity 0.48 → 0.16.", False, INK)],
         [("Greedy and recursive: ", True, NAVY),
          ("best question now, then repeat in each group until a stopping rule (depth, leaf size, purity).", False, INK)]],
     6.7, 4.75, 5.95, 2.0, 15)
notes(s, """
LUCAS (1 min)
A tree asks questions of the form "is x below a threshold?". To choose the question, we measure how mixed a group is with the Gini impurity: one minus the sum of squared class shares. It is the chance that two people drawn at random in the node have different labels — zero for a pure node, 0.5 for a 50/50 node.
A split is scored by the weighted impurity of its two children. We weight by size, otherwise isolating a single point would look perfect.
We checked this on exercise 3B.1 of the course: the root has Gini 0.48, "income above 30" gives 0.16, "age above 40" gives 0.45 — and our code, testing every threshold, finds the income question by itself.
The tree is built greedily: the best question now, no look-ahead, then we repeat in each child until a stopping rule.
""")

# 5 · Split search -----------------------------------------------------------
s = prs.slides.add_slide(BLANK)
frame(s, 5, "Optimisation", "Finding the best split: from O(n²) to O(n log n)", "Lucas")
card(s, 0.7, 1.95, 5.5, 2.15, "Which thresholds?",
     "Only midpoints between two consecutive distinct values: every threshold in between gives the same partition.", 1)
card(s, 0.7, 4.3, 5.5, 2.45, "Vectorised search",
     "Sort a feature once. Moving to the next threshold moves one point to the left, so cumulative sums of the "
     "labels give the counts of every threshold in one NumPy line.", 2)
cd = CategoryChartData()
cd.categories = ["250", "500", "1,000", "2,000", "4,000"]
cd.add_series("Speed-up vs naive loops", (36, 60, 94, 159, 261))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.6), Inches(1.95), Inches(6.05), Inches(4.8), cd)
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "Vectorised search is up to 261× faster (n rows)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
pl = ch.plots[0]
pl.gap_width = 60
pl.series[0].format.fill.solid()
pl.series[0].format.fill.fore_color.rgb = TEAL
pl.has_data_labels = True
pl.data_labels.number_format = '0"×"'
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size, pl.data_labels.font.bold = Pt(13), True
pl.data_labels.font.color.rgb = NAVY
notes(s, """
LUCAS (1 min)
The heart of training is the split search. Between two consecutive values, every threshold gives the same partition, so we only test midpoints — like scikit-learn.
A naive search splits the data for each candidate: that is quadratic, n squared per feature. Our vectorised version sorts the feature once; going from one threshold to the next moves exactly one point from right to left, so the class counts for all thresholds are just cumulative sums. Everything is one NumPy expression: n log n.
On continuous data, it is 36 times faster at 250 rows and 260 times faster at 4,000 — the gap grows with n, as the complexity predicts.
""")

# 6 · Verification -----------------------------------------------------------
s = prs.slides.add_slide(BLANK)
frame(s, 6, "Verification", "Our tree makes the same decisions as scikit-learn", "Lucas")
stat(s, "100 %", "same predictions, depths 1 to 4", 0.7, 1.95, 3.85, TEAL)
stat(s, "99.98 %", "same predictions on the test set (depth 8)", 4.75, 1.95, 3.85, NAVY)
stat(s, "6 / 6", "automated tests pass (pytest)", 8.8, 1.95, 3.85, GOLD)
card(s, 0.7, 4.0, 5.95, 2.75, "What the tests check",
     "Known Gini values · naive = vectorised · same first split · unlimited tree learns the train set · "
     "same predictions and importances · same leaves after pruning.")
card(s, 6.85, 4.0, 5.8, 2.75, "Why not exactly 100 %?",
     "Ties: two one-hot columns with exactly the same Gini. scikit-learn visits features in random order, "
     "we visit them in column order. Both choices are equally optimal.")
notes(s, """
LUCAS (45 s)
How do we know our tree is correct? We compared it with scikit-learn's DecisionTreeClassifier with the same hyperparameters. Up to depth 4 the predictions are 100 percent identical; on the final test set, 99.98 percent — two rows out of almost ten thousand.
Those two rows are ties: two one-hot columns that give exactly the same Gini. scikit-learn breaks ties in a random feature order, we use column order. Both are optimal.
Six automated tests check all of this, and they all pass. I hand over to Manuel for the experiments.
""")

# 7 · Protocol ---------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
frame(s, 7, "Protocol", "A fair comparison, with no data leakage", "Manuel")
steps = [("Split", "60 / 20 / 20\nstratified, seed 42"), ("Encode", "one-hot fitted\non train only"),
         ("Tune", "depth, leaf size\non validation"), ("Test", "used once,\nboth models")]
for i, (h, b) in enumerate(steps):
    x = 0.7 + i * 3.1
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.95, 2.75, 1.95, CARD)
    badge(s, i + 1, x + 0.25, 2.2, 0.5)
    text(s, h, x + 0.95, 2.25, 1.7, 0.4, 18, True, NAVY, font=HEAD)
    text(s, b, x + 0.25, 2.9, 2.3, 0.9, 14, color=INK)
    if i < 3:
        shape(s, MSO_SHAPE.RIGHT_ARROW, x + 2.8, 2.75, 0.25, 0.35, GOLD)
text(s, "Why not accuracy alone?", 0.7, 4.3, 6, 0.4, 18, True, NAVY, font=HEAD)
table(s, [["Model (validation)", "Accuracy", "F1"],
          ["Always predict “≤50K”", "76 %", "0"]], 0.7, 4.85, [3.4, 1.3, 1.0], 15)
text(s, "A model that never finds a high earner already scores 76 % accuracy.", 0.7, 5.95, 5.8, 0.7, 14, color=GREY)
rich(s, [[("F1", True, TEAL), (" (main): balance of precision and recall on the >50K class", False, INK)],
         [("Precision / recall", True, TEAL), (": which error dominates", False, INK)],
         [("ROC AUC", True, TEAL), (": quality of the ranking, threshold-free", False, INK)],
         [("No scaling", True, TEAL), (": a tree only compares a variable to a threshold", False, INK)]],
     7.2, 4.35, 5.45, 2.4, 15)
notes(s, """
MANUEL (1 min)
Our protocol: a stratified 60/20/20 split with seed 42, so each set keeps 24 percent of high earners. We split first and encode afterwards, with one-hot columns fixed on the training set only — so no information from validation or test leaks into training. No scaling: a tree compares a variable to a threshold, rescaling does not change the partition.
Hyperparameters are chosen on validation, and the test set is used exactly once, at the end, for both models.
On metrics: a model that always says "below 50K" gets 76 percent accuracy and finds nobody. So our main metric is F1 on the high-earner class, with precision, recall and ROC AUC.
""")

# 8 · Depth ------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
frame(s, 8, "Tuning", "Depth 8: the point where the tree starts to overfit", "Manuel")
depths = [str(d) for d in range(1, 19)]
train = (0.0, .573, .620, .620, .640, .650, .655, .679, .679, .687, .698, .710, .724, .735, .756, .766, .778, .806)
val = (0.0, .570, .621, .620, .637, .650, .653, .677, .669, .670, .668, .666, .668, .665, .666, .661, .655, .669)
cd = CategoryChartData()
cd.categories = depths
cd.add_series("Train F1", train)
cd.add_series("Validation F1", val)
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.7), Inches(1.9), Inches(7.9), Inches(4.9), cd)
ch = gf.chart
style_chart(ch, legend=True)
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0.5, 0.85
ch.value_axis.tick_labels.number_format = "0.00"
ch.value_axis.tick_labels.number_format_is_linked = False
ch.category_axis.has_title = True
ch.category_axis.axis_title.text_frame.text = "max_depth"
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
for ser, col in zip(ch.plots[0].series, (MUTED, TEAL)):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(2.75)
    ser.smooth = False
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = col
    ser.marker.format.line.color.rgb = col
pt = ch.plots[0].series[1].points[7]
pt.data_label.has_text_frame = True
pt.data_label.text_frame.text = "best: 0.677"
pt.data_label.position = XL_LABEL_POSITION.ABOVE
pt.data_label.font.bold, pt.data_label.font.size = True, Pt(13)
pt.data_label.font.color.rgb = NAVY
card(s, 8.9, 1.95, 3.75, 1.5, "Too shallow", "Bias: both curves low.")
card(s, 8.9, 3.6, 3.75, 1.5, "Too deep", "Variance: train ↑, val ↓.", accent=GOLD)
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 8.9, 5.25, 3.75, 1.5, NAVY)
text(s, "max_depth = 8\nmin_samples_leaf = 5", 9.2, 5.45, 3.2, 1.0, 18, True, WHITE, font=HEAD)
notes(s, """
MANUEL (1 min)
Here is the train and validation F1 as a function of depth. Below depth 5, both curves are low: the tree is too simple, high bias. Validation F1 peaks at depth 8, 0.677. After that, the training score keeps climbing towards 0.8 while validation stalls and drops: the extra leaves learn noise — overfitting, high variance.
At depth 1, F1 is zero: the "married" split leaves a majority of low earners on both sides, so the tree never predicts high income — but it already ranks people well, AUC 0.76.
We also tuned the minimum leaf size: 5 is slightly better. And scikit-learn's curve lies exactly on top of ours at every depth.
""")

# 9 · Test results -----------------------------------------------------------
s = prs.slides.add_slide(BLANK)
frame(s, 9, "Final comparison · test set", "Same results as scikit-learn, 20× slower", "Manuel")
table(s, [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "Fit time"],
          ["CART from scratch", "0.857", "0.751", "0.602", "0.669", "0.906", "1.0 s"],
          ["scikit-learn", "0.857", "0.751", "0.602", "0.669", "0.906", "0.04 s"],
          ["Always “≤50K”", "0.761", "0", "0", "0", "0.500", "—"]],
      0.7, 1.95, [3.4, 1.45, 1.45, 1.3, 1.2, 1.5, 1.6], 15, 0.55, highlight=1)
card(s, 0.7, 4.5, 3.85, 2.25, "Correct",
     "Identical metrics and confusion matrices; 99.98 % identical predictions.", 1)
card(s, 4.75, 4.5, 3.85, 2.25, "Cautious",
     "Precision 0.75 > recall 0.60: when it says >50K it is right, but it misses 40 % of high earners.", 2)
card(s, 8.8, 4.5, 3.85, 2.25, "Slower",
     "Same O(n log n) growth; scikit-learn is compiled Cython and never copies the data.", 3, accent=GOLD)
notes(s, """
MANUEL (1 min)
The final test, used once. Both trees get exactly the same numbers: 85.7 percent accuracy, precision 0.75, recall 0.60, F1 0.669, AUC 0.906 — far above the baseline.
The tree is cautious: when it predicts a high income it is usually right, but it misses 40 percent of the high earners. That is the usual effect of a minority class with a 0.5 threshold — Amine will come back to it.
The only difference is speed: we take one second, scikit-learn 0.04. Both grow linearly with the data; the gap is a constant factor — scikit-learn is compiled Cython and works in place, we re-sort and copy arrays in Python at every node. Amine now answers the question.
""")

# 10 · Answer ----------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
frame(s, 10, "The answer", "Hypothesis rejected: family situation comes first", "Amine")
cd = CategoryChartData()
order = [("hours-per-week", .032), ("age", .040), ("capital-loss", .070), ("capital-gain", .192),
         ("education", .215), ("marital status", .427)]
cd.categories = [k for k, _ in order]
cd.add_series("Importance", [v for _, v in order])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.7), Inches(1.9), Inches(7.2), Inches(4.9), cd)
ch = gf.chart
style_chart(ch)
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.category_axis.tick_labels.font.size = Pt(14)
pl = ch.plots[0]
pl.gap_width = 45
pl.vary_by_categories = False
ser = pl.series[0]
for i, (k, _) in enumerate(order):
    p = ser.points[i]
    p.format.fill.solid()
    p.format.fill.fore_color.rgb = GOLD if k == "marital status" else (NAVY if k == "education" else MUTED)
pl.has_data_labels = True
pl.data_labels.number_format = "0.00"
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size, pl.data_labels.font.bold = Pt(14), True
pl.data_labels.font.color.rgb = INK
text(s, "Importance by original variable (depth 8, one-hot columns summed)", 0.7, 6.65, 7.2, 0.3, 11, color=GREY)
rich(s, [[("1st question of the tree: ", True, NAVY), ("“married?”", True, GOLD)],
         [("Marital status ", True, NAVY), ("weighs twice education (0.43 vs 0.22).", False, INK)],
         [("Education is clearly 2nd, ", True, NAVY), ("ahead of capital income.", False, INK)],
         [("Age and hours ", True, NAVY), ("add little once these are known.", False, INK)],
         [("Same ranking ", True, NAVY), ("on 5 random splits of the data.", False, INK)]],
     8.3, 2.1, 4.35, 4.6, 17)
notes(s, """
AMINE (1 min 15)
So, is education the main driver? No. The very first question the tree asks is "is the person married?". When we sum the importances of each original variable, marital status weighs 0.43 — twice education, at 0.22. Education is clearly second, ahead of capital gains. Age and hours worked add little once marital status and education are known.
And this ranking is robust: on five different random splits, the first split is always "married" and the importances barely move. Our hypothesis is rejected.
""")

# 11 · Ablation and causality ------------------------------------------------
s = prs.slides.add_slide(BLANK)
frame(s, 11, "Robustness & meaning", "Ablation confirms it — but it is not causation", "Amine")
table(s, [["Removed variable(s)", "Validation F1 loss", "AUC loss"],
          ["family (marital + relationship)", "−0.122", "−0.047"],
          ["education", "−0.058", "−0.013"],
          ["hours per week", "−0.048", "−0.007"],
          ["capital gain + loss", "−0.042", "−0.031"],
          ["age", "≈ 0", "−0.005"]],
      0.7, 1.95, [3.6, 2.1, 1.5], 15, 0.52, highlight=1)
text(s, "Refit without the variable, same hyperparameters, validation set.", 0.7, 5.2, 7.2, 0.4, 12, color=GREY)
text(s, "relationship and marital status hold the same information (Husband / Wife ⇔ married): "
        "we remove them together.", 0.7, 5.65, 7.2, 1.0, 14, color=INK)
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 8.3, 1.95, 4.35, 4.8, NAVY)
text(s, "PREDICTION ≠ CAUSATION", 8.65, 2.25, 3.7, 0.3, 12, True, GOLD)
text(s, "Marriage does not make people rich.", 8.65, 2.65, 3.7, 1.0, 21, True, WHITE, font=HEAD)
text(s, "In 1994, “married” is a proxy for age and career stage, for sex (most married earners are “Husband”) "
        "and for household arrangements.\n\ncapital-gain is a consequence of wealth more than a cause.",
     8.65, 3.75, 3.7, 2.9, 14, color=WHITE)
notes(s, """
AMINE (1 min)
Importances are computed on the training set and favour some variables, so we added a second piece of evidence: ablation. We remove a variable, refit, and measure what we lose on validation. Removing the family situation costs the most by far: 0.12 of F1. Education comes second.
One detail: relationship and marital status contain the same information — "husband" or "wife" means married. Remove one, the tree uses the other; so we remove them together.
But careful: this is prediction, not causation. Marriage does not make people rich. In this 1994 data, being married is a proxy for age, career stage and sex. Our conclusion is: among these census variables, family situation is the best predictor of a high income, and education the second.
""")

# 12 · Limits ----------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
frame(s, 12, "Limits", "When does the model fail?", "Amine")
card(s, 0.7, 1.95, 5.9, 2.3, "Misses high earners",
     "Recall 0.60 at threshold 0.5. A threshold of 0.34 (chosen on validation) gives recall 0.81 and F1 0.685.", 1)
card(s, 6.75, 1.95, 5.9, 2.3, "Less accurate for women",
     "Recall 0.52 vs 0.62 for men; predicts 7.5 % of high earners for a true 11 %. It learns the 1994 pay gap.", 2)
card(s, 0.7, 4.45, 5.9, 2.3, "One tree is unstable",
     "Top splits are stable, deep leaves are not (validation F1 0.638–0.659 over 5 splits). Axis-parallel, greedy cuts.", 3)
card(s, 6.75, 4.45, 5.9, 2.3, "Ensembles do better",
     "Same splits, validation AUC: our tree 0.910 · random forest 0.920 · gradient boosting 0.931.", 4, accent=GOLD)
notes(s, """
AMINE (1 min)
Where does the model fail? First, it misses high earners: recall 0.60. Lowering the threshold to 0.34, chosen on validation, raises recall to 0.81 — the right threshold depends on the cost of each error.
Second, it is less accurate for women: recall 0.52 against 0.62. Only 11 percent of women earn above 50K in this data, and the model predicts even fewer: it learns the 1994 pay gap. Using it for real decisions would need a fairness analysis.
Third, a single tree is unstable in its lower levels, and its cuts are axis-parallel and greedy.
That is why ensembles do better: a random forest averages trees to reduce variance, boosting corrects errors step by step — AUC 0.92 and 0.93 against 0.91. Both use the split search we implemented as their building block.
""")

# 13 · Conclusion ------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
text(s, "TAKEAWAYS", 0.8, 0.7, 6, 0.3, 13, True, GOLD)
text(s, "What we learned", 0.8, 1.05, 11, 0.8, 36, True, WHITE, font=HEAD)
items = [("We built CART from scratch", "Gini, vectorised search, greedy growth, pruning — matches scikit-learn on 99.98 % of test rows."),
         ("Measured fairly", "Stratified split, no leakage, test used once; F1 0.669 and AUC 0.906 vs a 76 %-accuracy baseline."),
         ("Education is not n°1", "Family situation predicts a >50K income best, education second — an association, not a cause."),
         ("Next step", "A random forest built on our own tree, and a threshold tuned to the cost of errors.")]
for i, (h, b) in enumerate(items):
    y = 2.15 + i * 1.18
    badge(s, i + 1, 0.8, y, 0.6)
    text(s, h, 1.7, y - 0.02, 10.5, 0.45, 20, True, WHITE, font=HEAD)
    text(s, b, 1.7, y + 0.45, 10.8, 0.5, 15, color=MUTED)
text(s, "Thank you — questions?", 0.8, 6.85, 8, 0.4, 16, True, GOLD)
notes(s, """
LUCAS (30 s)
To conclude: we built a CART tree from scratch and showed it makes the same decisions as scikit-learn. We evaluated it with a clean protocol and the right metrics. And we answered our question: education is not the main driver of a high income in this data — family situation comes first, education second, as an association and not a cause. The natural next step is a random forest built on our own tree. Thank you, we are happy to take your questions.
""")

out = HERE / "oral_deck.pptx"
prs.save(out)
print(f"{out} written, {len(prs.slides)} slides")
