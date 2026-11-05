"""Builds report/report.pdf from the figures written by notebook 02.

Run after the two notebooks:  uv run --with reportlab python report/build_report.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).parent
FIG = HERE / "figures"

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=13.5, spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11)
caption = ParagraphStyle("caption", parent=small, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=10)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4)
h1.keepWithNext = h2.keepWithNext = 1
formula = ParagraphStyle("formula", parent=body, alignment=TA_CENTER, fontName="Times-Italic", fontSize=11)
code = ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10, leftIndent=12,
                      backColor=colors.HexColor("#f4f4f4"), borderPadding=4, spaceAfter=8)

P = lambda t, s=body: Paragraph(t, s)


def table(rows, widths, header=True):
    t = Table(rows, colWidths=widths, hAlign="CENTER")
    style = [("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def figure(name, width, text):
    img = Image(str(FIG / name))
    img.drawHeight = img.drawHeight * width / img.drawWidth
    img.drawWidth = width
    return KeepTogether([img, P(text, caption)])


def cell(t):
    return Paragraph(t, small)


story = [
    P("<b>Is education the main driver of a high income?</b><br/>"
      "A CART classification tree from scratch on the Adult census data",
      ParagraphStyle("title", parent=ss["Title"], fontSize=17, leading=22)),
    P("Lucas Petrelluzzi, Manuel, Amine — Mathematics &amp; Statistics for Machine Learning (A. Diallo), "
      "Albert School, MSc 1 — November 2026", ParagraphStyle("au", parent=body, alignment=TA_CENTER)),
    P("Code: <font name='Courier'>github.com/Petrel971/ML-Project-Bootcamp</font> — data: "
      "<font name='Courier'>kaggle.com/datasets/uciml/adult-census-income</font>",
      ParagraphStyle("au2", parent=small, alignment=TA_CENTER, spaceAfter=10)),

    P("<b>Abstract.</b> We implement a CART classification tree from scratch in NumPy (Gini impurity, "
      "vectorised exhaustive split search, greedy recursive growth, cost-complexity pruning, impurity-based "
      "importances) and use it to ask which census variables best predict an income above $50K. On a stratified "
      "test set our tree matches <font name='Courier'>sklearn.tree.DecisionTreeClassifier</font> exactly in "
      "accuracy (85.7 %), F1 (0.669) and agrees on 99.98 % of predictions, while training about 20× slower. "
      "Contrary to our hypothesis, education is not the first driver: being married comes first (importance 0.43, "
      "largest ablation loss), education second (0.22). We discuss why this is association, not causation, and "
      "where a single tree fails (minority-class recall, women, variance).", body),

    P("1. Problem and research question", h1),
    P("The Adult dataset (UCI, extracted from the 1994 US Census) describes 48,842 adults with 14 attributes and "
      "a binary label: does the person earn more than $50K a year? It is a classic benchmark with a moderate class "
      "imbalance (23.9 % of >50K) and a mix of numeric and categorical variables. A common belief is that "
      "<i>education</i> is what drives income. We test it:"),
    P("<b>Research question.</b> <i>Is education the main driver of earning more than $50K a year, or do other "
      "factors such as marital status, age and hours worked matter more?</i> Starting hypothesis: education is "
      "the number-one factor."),
    P("A decision tree suits the question: it handles mixed variables without scaling, and its rules and "
      "importances are readable. The brief also requires implementing the algorithm ourselves and comparing it "
      "with a library on the same splits; CART is short enough to be written and <i>verified</i> exactly against "
      "scikit-learn."),

    P("2. Data", h1),
    table([["Variable type", "Variables"],
           [cell("Numeric (5 kept)"), cell("age, education-num (1–16, diploma level), capital-gain, capital-loss, hours-per-week")],
           [cell("Categorical (8)"), cell("workclass, marital-status, occupation, relationship, race, sex, native-country")],
           [cell("Dropped (2)"), cell("<b>fnlwgt</b>: census sampling weight, not a property of the person. <b>education</b>: "
                                      "same information as education-num; keeping both would split the importance of education in two.")]],
          [3.5 * cm, 12.5 * cm]),
    Spacer(1, 6),
    P("Missing values only affect three categorical variables: workclass (5.7 %), occupation (5.8 %) and "
      "native-country (1.8 %). They are <i>informative</i>: only 9.4 % of the rows with a missing occupation earn "
      ">50K versus 23.9 % overall. We therefore keep them as a category \"Unknown\" rather than dropping or imputing "
      "them. Taken one at a time, the variables most associated with income (Cramér's V) are relationship (0.45) "
      "and marital-status (0.45), then education (0.36), occupation (0.35), age (0.32) and hours (0.27)."),
    figure("exploration.png", 16 * cm, "Figure 1 — Share of >50K by variable (dashed line: overall 24 %)."),

    P("3. Method: CART from scratch", h1),
    P("3.1 Model", h2),
    P("A classification tree partitions the feature space into rectangles R<sub>1</sub>, …, R<sub>M</sub> "
      "(the leaves) with rules x<sub>j</sub> ≤ s at internal nodes. In leaf R<sub>m</sub> the class proportions "
      "are p<sub>mk</sub> = (1/n<sub>m</sub>) Σ<sub>i∈R<sub>m</sub></sub> 1(y<sub>i</sub> = k); the tree predicts "
      "argmax<sub>k</sub> p<sub>mk</sub> and returns p<sub>mk</sub> as probability."),
    P("3.2 Objective", h2),
    P("Node impurity is measured by the Gini index (entropy is also implemented):"),
    P("G = 1 − Σ<sub>k</sub> p<sub>k</sub><super>2</super>,   0 for a pure node, 0.5 at most with two classes", formula),
    P("A split sending n<sub>L</sub> points left and n<sub>R</sub> right is scored by the size-weighted impurity of "
      "its children; the best split minimises it, i.e. maximises the gain G<sub>parent</sub> − G<sub>split</sub>:"),
    P("G<sub>split</sub> = (n<sub>L</sub>/n) G<sub>L</sub> + (n<sub>R</sub>/n) G<sub>R</sub>", formula),
    P("G is the chance that two points drawn at random in the node have different labels. Weighting by size "
      "prevents a split that isolates a single pure point from looking excellent. At the root of Adult "
      "G = 1 − (0.761² + 0.239²) = 0.364. On exercise 3B.1 of the lecture (10 clients, root G = 0.48), "
      "\"income &gt; 30?\" gives G<sub>split</sub> = 0.16 and \"age &gt; 40?\" 0.45; our search over all "
      "thresholds finds the income split (threshold 33, same partition)."),
    P("3.3 Optimisation: split search", h2),
    P("Between two consecutive distinct values a &lt; b of a feature, all thresholds give the same partition, so "
      "we test the midpoints (a + b)/2, as scikit-learn does. A naive search splits and scores the data for "
      "every candidate: O(d·n²). Our <b>vectorised</b> search sorts each feature once; moving to the next threshold "
      "moves one point from right to left, so the left class counts of all thresholds are the cumulative sums of "
      "the one-hot labels (right = total − left). All scores of a feature come from one NumPy expression, "
      "O(d·n log n). On continuous data it is 36× faster than the naive version at n = 250 and 260× at n = 4,000."),
    P("3.4 Construction and stopping", h2),
    P("As in Lecture 3B, the tree is grown greedily — best question now, no look-ahead — and recursively (finding the optimal tree is NP-hard). A node becomes a leaf when "
      "max_depth is reached, it has fewer than min_samples_split or 2·min_samples_leaf points, it is pure, no "
      "valid threshold exists, or the weighted decrease (N<sub>t</sub>/N)(G<sub>t</sub> − G<sub>split</sub>) is "
      "below min_impurity_decrease. Optional cost-complexity pruning repeatedly collapses the node with the "
      "smallest g(t) = (R(t) − R(T<sub>t</sub>)) / (|T<sub>t</sub>| − 1) while g(t) ≤ α."),
    Preformatted("""fit(X, y):  root = build(X, y, depth=0)
build(X, y, depth):
    node = leaf with class proportions of y
    if stopping rule: return node
    (j, s, score) = best_split(X, y)          # vectorised, O(d n log n)
    if no split or weighted decrease too small: return node
    node.rule = (x_j <= s)
    node.left  = build(X[x_j <= s], y[...], depth + 1)
    node.right = build(X[x_j >  s], y[...], depth + 1)
    return node""", code),
    P("3.5 Prediction and importance", h2),
    P("An observation goes down the tree until a leaf (we route all rows at once with boolean masks). The "
      "importance of feature j is Σ over nodes splitting on j of (N<sub>t</sub>/N)(G<sub>t</sub> − "
      "(N<sub>L</sub>/N<sub>t</sub>)G<sub>L</sub> − (N<sub>R</sub>/N<sub>t</sub>)G<sub>R</sub>), normalised to 1. "
      "For one-hot encoded variables we sum the importances of their columns."),
    P("3.6 Verification", h2),
    P("Exercise 3B.1 reproduced exactly, six automated tests (<font name='Courier'>uv run pytest</font>) and the checkpoints of notebook 01 check: known "
      "Gini values; naive = vectorised search; the same first split as scikit-learn; a perfect fit of the training "
      "set when the depth is unlimited; identical predictions and importances for depths 1–4; the same number of "
      "leaves after pruning. On Adult, predictions agree with scikit-learn at 100 % for depths 1–4 and 99.99 % at "
      "depths 5–6; the rare differences are <b>ties</b> (two one-hot columns with exactly the same Gini), broken by "
      "column order for us and by a random feature order in scikit-learn."),

    P("4. Experimental protocol", h1),
    table([["Step", "Choice", "Reason"],
           [cell("Split"), cell("Stratified 60/20/20, seed 42"), cell("Train to learn, validation to tune, test used <b>once</b>; 23.9 % >50K in each set")],
           [cell("Encoding"), cell("One-hot, columns fixed on train"), cell("Split first, then encode: no information from validation/test (no leakage)")],
           [cell("Scaling"), cell("None"), cell("A tree compares a variable to a threshold: scale-invariant")],
           [cell("Tuning"), cell("max_depth 1–20, min_samples_leaf 1–100"), cell("Selected by validation F1, confirmed by 5-fold CV on train")],
           [cell("Metrics"), cell("F1, precision, recall, ROC AUC, accuracy"), cell("A constant \"≤50K\" model already gets 76 % accuracy with F1 = 0")],
           [cell("Comparison"), cell("Same splits, same hyperparameters"), cell("Ours vs DecisionTreeClassifier; also fit time and agreement rate")]],
          [2.3 * cm, 5.2 * cm, 8.5 * cm]),
    Spacer(1, 6),
    P("Train/validation/test sizes: 29,304 / 9,769 / 9,769 rows, 91 columns after encoding."),

    P("5. Results", h1),
    P("5.1 Choosing the complexity", h2),
    figure("depth_curve.png", 12 * cm, "Figure 2 — Train and validation F1 by depth; our tree and scikit-learn overlap."),
    P("Validation F1 rises until depth 8 (0.677) and then stalls while train F1 keeps rising towards 0.8: deeper "
      "trees overfit (variance); shallower ones underfit (bias). Depth 1 has F1 = 0 although its AUC is 0.76: "
      "both children of the \"married\" split still have a ≤50K majority. min_samples_leaf = 5 slightly improves "
      "validation F1 (0.679) and AUC (0.910). <b>Chosen: max_depth = 8, min_samples_leaf = 5.</b> A 5-fold "
      "cross-validation on the training set confirms it: mean F1 reaches 0.658 ± 0.015 at depth 8, and deeper trees "
      "stay within one standard deviation (best 0.662 at depth 14); by the one-standard-error rule the simplest, depth 8, is kept."),
    P("5.2 Comparison on the test set", h2),
    table([["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "Fit time"],
           ["CART from scratch", "0.857", "0.751", "0.602", "0.669", "0.906", "≈ 1.0 s"],
           ["scikit-learn", "0.857", "0.751", "0.602", "0.669", "0.906", "≈ 0.04 s"],
           ["Always ≤50K", "0.761", "0", "0", "0", "0.500", "—"]],
          [3.6 * cm, 1.8 * cm, 1.8 * cm, 1.6 * cm, 1.4 * cm, 1.8 * cm, 1.9 * cm]),
    Spacer(1, 6),
    P("Both trees give the same prediction on 99.98 % of test rows (2 rows differ, caused by a tie) and the same "
      "metrics; the confusion matrices are identical (Figure 3). Our training is about 20× slower at every sample "
      "size from 2,000 to 29,304 rows; both grow roughly linearly in n. The gap is a constant factor: scikit-learn's "
      "splitter is compiled Cython working in place on sorted indices, while our Python recursion re-sorts features "
      "and copies sub-arrays at each node."),
    figure("confusion.png", 11 * cm, "Figure 3 — Test confusion matrices (threshold 0.5)."),

    P("6. Answer to the research question", h1),
    figure("importance.png", 10 * cm, "Figure 4 — Importance by original variable (depth-8 tree, one-hot columns summed)."),
    table([["Evidence", "Family situation", "Education", "Others"],
           [cell("Association alone (Cramér's V)"), cell("<b>0.45</b>"), cell("0.36"), cell("occupation 0.35, age 0.32, hours 0.27")],
           [cell("First split"), cell("<b>married (civ-spouse)?</b>"), cell("—"), cell("—")],
           [cell("Importance (depth 8)"), cell("<b>0.43</b>"), cell("0.22"), cell("capital-gain 0.19, capital-loss 0.07, age 0.04, hours 0.03")],
           [cell("Ablation: validation AUC loss"), cell("<b>0.047</b>"), cell("0.013"), cell("capital 0.031, hours 0.007, age 0.005")],
           [cell("Ablation: validation F1 loss"), cell("<b>0.122</b>"), cell("0.058"), cell("hours 0.048, capital 0.042, age ≈ 0")],
           [cell("5 random splits"), cell("first split 5/5"), cell("2nd each time"), cell("—")]],
          [4.2 * cm, 3.3 * cm, 2.4 * cm, 6.1 * cm]),
    Spacer(1, 6),
    P("<b>The hypothesis is rejected.</b> The family situation — being married and living with a spouse — is the "
      "strongest predictor by every measure: first split of the tree, twice the importance of education, and the "
      "largest loss when it is removed. Education is clearly the <b>second</b> factor, ahead of capital income; age "
      "and hours worked add little once marital status and education are known. The readable depth-3 tree tells "
      "the same story: unmarried people without capital gains have a 2–15 % chance of >50K depending on education; "
      "married people with at least an associate degree (education-num ≥ 12) reach 65 %."),
    P("Two subtleties. (1) relationship and marital-status carry the same information (Husband/Wife ⇔ married): "
      "the tree uses one and ignores the other, so removing either alone costs little; removing both costs the most. "
      "(2) Age has low importance not because it is irrelevant (V = 0.32) but because it is absorbed by marital "
      "status: importance measures what a variable adds <i>given the others</i>."),
    P("<b>Association, not causation.</b> Marriage does not cause a high income. In this 1994 sample, marital "
      "status is a proxy for age and career stage, for sex (married earners are mostly \"Husband\"), and possibly "
      "for household arrangements; capital-gain is closer to a consequence of wealth than a cause. Our conclusion is "
      "predictive: among these census variables, family situation is the best single predictor of a >50K income, "
      "and education the second."),

    P("7. Discussion: limits and when the model fails", h1),
    P("All analyses below use the validation set or cross-validation; the test set is not used after section 5.2. "
      "<b>Minority class.</b> At the default threshold the tree misses 40 % of high earners (test recall 0.60). A "
      "threshold of 0.36, chosen by 5-fold cross-validation on the training set, raises validation recall from 0.61 to "
      "0.82 and F1 from 0.679 to 0.694, at the cost of precision (0.76 → 0.60) — the right threshold depends on the cost of each error."),
    figure("threshold.png", 9.5 * cm, "Figure 5 — Precision, recall and F1 by decision threshold (out-of-fold, 5-fold CV on train)."),
    P("<b>Errors by sex (validation).</b> Recall is 0.54 for women against 0.63 for men (F1 0.65 vs 0.68). Only 11 % of women "
      "earn >50K, and the model predicts 7.4 %: it learns and slightly amplifies the 1994 income gap. "
      "Any use of such a model for decisions about people would need a fairness analysis."),
    P("<b>Variance.</b> Over 5 random splits, the first split and the top importances are unchanged (marital "
      "0.41–0.42, education 0.22–0.23), so our answer is robust; validation F1 varies from 0.638 to 0.659 and the "
      "deep, small leaves change. <b>Structure.</b> Axis-aligned cuts approximate smooth effects by staircases; "
      "greedy growth misses interactions invisible at the first step; impurity importance is computed on the "
      "training set and favours variables with many thresholds, which is why we added ablation. <b>Cost.</b> "
      "Our implementation is about 20× slower than scikit-learn for identical results."),
    P("<b>Improvements.</b> Ensembles reduce the weaknesses of one tree. For discussion we ran scikit-learn's "
      "random forest (300 trees: validation F1 0.681, AUC 0.920, lower variance) and gradient boosting (F1 0.719, AUC "
      "0.931, lower bias) on the same splits, against 0.679 / 0.910 for our tree. Both reuse the split search we "
      "implemented as their building block; a natural extension is to build a random forest on our CART."),

    P("8. Conclusion", h1),
    P("We implemented CART from scratch and verified it against scikit-learn: same splits, same predictions on "
      "99.98 % of test rows, same metrics (accuracy 85.7 %, F1 0.669, AUC 0.906), with a 20× slower training. Used "
      "to answer our question, it shows that <b>education is not the main driver of a high income in the 1994 "
      "census: family situation comes first, education second</b>, a predictive association that reflects age, "
      "sex and household structure rather than a causal effect."),

    P("Reproducibility", h1),
    P("<font name='Courier'>uv sync</font> installs the exact versions of <font name='Courier'>uv.lock</font> "
      "(Python ≥ 3.11, NumPy, pandas, scikit-learn, Matplotlib, Seaborn). <font name='Courier'>uv run pytest</font> "
      "runs the tests; then run <font name='Courier'>notebooks/01_cart_from_scratch.ipynb</font> and "
      "<font name='Courier'>notebooks/02_experiments.ipynb</font> top to bottom (the data is downloaded from OpenML "
      "on first run and cached in <font name='Courier'>data/</font>, not redistributed). Seed 42 everywhere; "
      "the model code is in <font name='Courier'>src/cart.py</font> (NumPy only), scikit-learn is used only for "
      "splitting, metrics and the comparison."),
    P("References", h1),
    P("[1] Breiman, Friedman, Olshen, Stone (1984). <i>Classification and Regression Trees</i>. Wadsworth.<br/>"
      "[2] Becker, Kohavi (1996). <i>Adult</i> [dataset]. UCI Machine Learning Repository, doi:10.24432/C5XW20.<br/>"
      "[3] Hastie, Tibshirani, Friedman (2009). <i>The Elements of Statistical Learning</i>, ch. 9.2 and 15. Springer.<br/>"
      "[4] Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. <i>JMLR</i> 12, 2825–2830.<br/>"
      "[5] A. Diallo (2026). <i>Lecture 3B — Beyond Lines: Trees, SVMs and k-means</i>, Part I, and <i>Lab 1</i>. Mathematics &amp; Statistics for ML, Albert School.", small),
]


# Keep a heading on the same page as the figure that follows it.
merged = []
for f in story:
    if isinstance(f, KeepTogether) and merged and isinstance(merged[-1], Paragraph) and merged[-1].style in (h1, h2):
        heads = [merged.pop()]
        if merged and isinstance(merged[-1], Paragraph) and merged[-1].style is h1:
            heads.insert(0, merged.pop())
        f = KeepTogether(heads + f._content)
    merged.append(f)
story = merged


def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.drawString(2 * cm, 1.2 * cm, "CART from scratch on Adult — Lucas, Manuel, Amine")


SimpleDocTemplate(str(HERE / "report.pdf"), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                  topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="CART from scratch on Adult",
                  author="Lucas, Manuel, Amine").build(story, onFirstPage=footer, onLaterPages=footer)
print("report/report.pdf written")
