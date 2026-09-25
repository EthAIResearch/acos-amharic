#!/usr/bin/env python3
"""
Generate Publication & Executive Quality Presentation PDF for Amharic ACOS Research
Covers the entire research trajectory from 6-stage pipeline to ByT5-Base Epoch 9.
"""
import os
import sys
import re

# Ensure local user site-packages are accessible for reportlab
sys.path.insert(0, '/home/codeknight/.local/lib/python3.14/site-packages')

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Ethiopic font if available
ethiopic_font_regular = '/usr/share/fonts/google-droid-sans-fonts/DroidSansEthiopic-Regular.ttf'
ethiopic_font_bold = '/usr/share/fonts/google-droid-sans-fonts/DroidSansEthiopic-Bold.ttf'

has_ethiopic_font = False
if os.path.exists(ethiopic_font_regular) and os.path.exists(ethiopic_font_bold):
    pdfmetrics.registerFont(TTFont('Ethiopic', ethiopic_font_regular))
    pdfmetrics.registerFont(TTFont('EthiopicBold', ethiopic_font_bold))
    has_ethiopic_font = True

GEEZ_REGEX = re.compile(r'([\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F][\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F\s]*[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]|[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F])')

def fmt(text):
    """Automatically wrap any Ge'ez script words in <font name="Ethiopic">."""
    if has_ethiopic_font and isinstance(text, str):
        return GEEZ_REGEX.sub(r'<font name="Ethiopic">\1</font>', text)
    return text

# Color Palette
PRIMARY_DARK = colors.HexColor('#0F172A')   # Slate 900
BRAND_BLUE   = colors.HexColor('#2563EB')   # Blue 600
BRAND_LIGHT  = colors.HexColor('#EFF6FF')   # Blue 50
TEXT_DARK    = colors.HexColor('#1E293B')   # Slate 800
TEXT_MUTED   = colors.HexColor('#64748B')   # Slate 500
BORDER_COLOR = colors.HexColor('#CBD5E1')   # Slate 300
ACCENT_GREEN = colors.HexColor('#059669')   # Emerald 600
ACCENT_LIGHT_GREEN = colors.HexColor('#ECFDF5') # Emerald 50
ACCENT_AMBER = colors.HexColor('#D97706')   # Amber 600
ACCENT_LIGHT_AMBER = colors.HexColor('#FFFBEB') # Amber 50
ACCENT_RED   = colors.HexColor('#DC2626')   # Red 600
ACCENT_LIGHT_RED = colors.HexColor('#FEF2F2') # Red 50
CARD_BG      = colors.HexColor('#F8FAFC')   # Slate 50

PAGE_WIDTH, PAGE_HEIGHT = landscape(letter)  # 792 x 612 pt
MARGIN_H = 36  # Left & Right
MARGIN_V = 36  # Top & Bottom
USABLE_WIDTH = PAGE_WIDTH - 2 * MARGIN_H  # 720 pt
USABLE_HEIGHT = PAGE_HEIGHT - 2 * MARGIN_V # 540 pt


class PresentationCanvas(canvas.Canvas):
    """Custom canvas that tracks pages and draws slide frame headers & footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_slide_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_slide_decorations(self, total_pages):
        self.saveState()
        page_num = self._pageNumber

        # Don't draw regular slide headers on the title cover slide (page 1)
        if page_num > 1:
            # Top accent bar
            self.setFillColor(PRIMARY_DARK)
            self.rect(0, PAGE_HEIGHT - 4, PAGE_WIDTH, 4, fill=1, stroke=0)
            self.setFillColor(BRAND_BLUE)
            self.rect(MARGIN_H, PAGE_HEIGHT - 4, 140, 4, fill=1, stroke=0)

            # Header subtle divider
            self.setStrokeColor(colors.HexColor('#E2E8F0'))
            self.setLineWidth(0.6)
            self.line(MARGIN_H, PAGE_HEIGHT - 44, PAGE_WIDTH - MARGIN_H, PAGE_HEIGHT - 44)

            # Header institute branding right
            self.setFont('Helvetica-Bold', 7.5)
            self.setFillColor(BRAND_BLUE)
            self.drawRightString(PAGE_WIDTH - MARGIN_H, PAGE_HEIGHT - 28, 'EthAIResearch / INSA')
            self.setFont('Helvetica', 7)
            self.setFillColor(TEXT_MUTED)
            self.drawRightString(PAGE_WIDTH - MARGIN_H, PAGE_HEIGHT - 38, 'Amharic ACOS Quadruple Research')

            # Footer
            self.setStrokeColor(colors.HexColor('#E2E8F0'))
            self.setLineWidth(0.6)
            self.line(MARGIN_H, 28, PAGE_WIDTH - MARGIN_H, 28)

            self.setFont('Helvetica', 7.5)
            self.setFillColor(TEXT_MUTED)
            self.drawString(MARGIN_H, 16, 'Amharic ACOS Quadruple Extraction: Comprehensive Research Progress & Architecture Report')
            self.drawRightString(PAGE_WIDTH - MARGIN_H, 16, f'Slide {page_num} of {total_pages}')
        else:
            # Cover slide decorative border
            self.setFillColor(PRIMARY_DARK)
            self.rect(0, PAGE_HEIGHT - 8, PAGE_WIDTH, 8, fill=1, stroke=0)
            self.setFillColor(BRAND_BLUE)
            self.rect(MARGIN_H, PAGE_HEIGHT - 8, 220, 8, fill=1, stroke=0)

            self.setFillColor(PRIMARY_DARK)
            self.rect(0, 0, PAGE_WIDTH, 6, fill=1, stroke=0)
            self.setFillColor(ACCENT_GREEN)
            self.rect(PAGE_WIDTH - MARGIN_H - 160, 0, 160, 6, fill=1, stroke=0)

        self.restoreState()


def create_presentation_pdf(output_filename="docs/Amharic_ACOS_Research_Presentation.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=landscape(letter),
        leftMargin=MARGIN_H,
        rightMargin=MARGIN_H,
        topMargin=48,
        bottomMargin=32
    )

    base_styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY_DARK
    )
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=17,
        textColor=TEXT_MUTED
    )
    slide_cat_style = ParagraphStyle(
        'SlideCategory',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=BRAND_BLUE
    )
    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=PRIMARY_DARK
    )
    slide_sub_style = ParagraphStyle(
        'SlideSubtitle',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=TEXT_MUTED
    )
    body_style = ParagraphStyle(
        'SlideBody',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=TEXT_DARK
    )
    body_bold = ParagraphStyle(
        'SlideBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    bullet_style = ParagraphStyle(
        'SlideBullet',
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-7
    )
    math_style = ParagraphStyle(
        'MathBlock',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1E3A8A')
    )
    card_title_style = ParagraphStyle(
        'CardTitle',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=PRIMARY_DARK
    )

    def P(text, style):
        return Paragraph(fmt(text), style)

    def header_block(category_tag, title_text, subtitle_text):
        """Standardized Slide Header block."""
        return [
            P(category_tag.upper(), slide_cat_style),
            Spacer(1, 2),
            P(title_text, slide_title_style),
            Spacer(1, 1),
            P(subtitle_text, slide_sub_style),
            Spacer(1, 10)
        ]

    def card_box(title, content_flowables, bg_color=CARD_BG, border_color=BORDER_COLOR, width=USABLE_WIDTH):
        """Constructs a clean bordered card containing text/flowables."""
        table_data = []
        if title:
            table_data.append([P(f"<b>{title}</b>", card_title_style)])
        table_data.append([content_flowables])
        t = Table(table_data, colWidths=[width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg_color),
            ('BOX', (0,0), (-1,-1), 0.8, border_color),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        return t

    story = []

    # =========================================================================
    # SLIDE 1: Title Slide (Executive Cover)
    # =========================================================================
    story.append(Spacer(1, 20))
    story.append(P("Azariyas Mekonen — Ethiopian Artificial Intelligence Institute (INSA / EthAIResearch)", ParagraphStyle(
        'InstBadge', fontName='Helvetica-Bold', fontSize=9, leading=11, textColor=BRAND_BLUE
    )))
    story.append(Spacer(1, 10))
    story.append(P("Towards End-to-End Amharic Aspect-Category-Opinion-Sentiment (ACOS) Quadruple Extraction", title_style))
    story.append(Spacer(1, 8))
    story.append(P("From 6-Stage Extractive Pipelines to Byte-Level Generative Foundation Models", subtitle_style))
    story.append(Spacer(1, 20))

    # Executive Highlights Card
    exec_content = [
        P("<b>Executive Summary & Scientific Milestones:</b>", body_bold),
        Spacer(1, 4),
        P("• <b>Rigorous Systematic Evolution:</b> Documenting the engineering and mathematical journey across <b>6 distinct model generations</b> for low-resource Amharic sentiment quadruples.", bullet_style),
        P("• <b>Root-Cause Diagnostic Discoveries:</b> Identified and mathematically formalized the <b>Error Compounding Collapse</b> of modular pipelines (F1 ≈ ∏ F1_i → 14.2%) and the <b>BIO Token Fragmentation Ceiling</b> of sequential taggers (33.04% recall cap).", bullet_style),
        P("• <b>Extractive Span-ASTE Optimization:</b> Engineered Deep Biaffine Bilinear Attention (Run 2) and vectorized O(1) prefix-sum span mean-pooling (Run 3), raising upstream Aspect detection to a record <b>55.20% F1</b>.", bullet_style),
        P("• <b>Historical Breakthrough with Generative ByT5-Base:</b> Solved the fundamental extractive limitation (0.00% implicit extraction) by transitioning to byte-level autoregression, achieving <b>19.28% Implicit Quad F1</b> and <b>182 True Positive full quadruples</b> at Epoch 9.", bullet_style),
    ]
    story.append(card_box("", exec_content, bg_color=BRAND_LIGHT, border_color=colors.HexColor('#BFDBFE'), width=USABLE_WIDTH))

    story.append(Spacer(1, 25))
    meta_table = Table([
        [
            P("<b>Author:</b><br/>Azariyas Mekonen", body_style),
            P("<b>Target Domain:</b><br/>Amharic E-Commerce & Hospitality Reviews", body_style),
            P("<b>Current State:</b><br/>ByT5-Base Generative (Final Test Evaluation)", body_style),
            P("<b>Date & Version:</b><br/>September 2026 · Technical Report v3.0", body_style)
        ]
    ], colWidths=[180, 180, 180, 180])
    meta_table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 2: Task Definition & The 3 Amharic Linguistic Realities
    # =========================================================================
    story.extend(header_block(
        "Foundational Problem Formulation",
        "The Amharic ACOS Quadruple Extraction Challenge",
        "Formulating fine-grained sentiment as four-dimensional semantic tuples under low-resource Semitic morphology."
    ))

    task_desc = [
        P("<b>Formal Quadruple Objective:</b>", body_bold),
        P("Given an Amharic review sentence X = [x₁, x₂, ..., xₙ], extract the full set of semantic quadruples:<br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;<b>Q = {(a, c, s, o) | a ∈ A ∪ {NULL}, c ∈ C, s ∈ S, o ∈ O ∪ {NULL}}</b><br/>"
          "where <b>a</b> is the aspect term, <b>c</b> is one of 29 predefined categories, <b>s ∈ {POS, NEG, NEU}</b> is sentiment polarity, and <b>o</b> is the opinion expression.", body_style)
    ]
    story.append(card_box("", task_desc, bg_color=CARD_BG, border_color=BORDER_COLOR))
    story.append(Spacer(1, 10))

    # 3 Linguistic Realities Columns
    col1 = [
        P("<b>1. Extreme Morphological Fusion</b>", ParagraphStyle('H1', parent=body_bold, textColor=ACCENT_RED)),
        Spacer(1, 3),
        P("Amharic is a morphologically rich, highly agglutinative Semitic language. Pronouns, prepositions, and negation clitics fuse into single tokens:<br/>"
          "• <i>'አልተመቸኝም'</i> ('al-te-meche-gn-m') decomposes into: <i>al-</i> (negation) + <i>temeche</i> (root) + <i>-gn</i> (1st pers. obj) + <i>-m</i> (conjunction).<br/>"
          "• Standard tokenizers split tokens arbitrarily, shattering opinion modifiers.", body_style)
    ]
    col2 = [
        P("<b>2. High Implicit Density (42.7%)</b>", ParagraphStyle('H2', parent=body_bold, textColor=ACCENT_AMBER)),
        Spacer(1, 3),
        P("Nearly half of all annotated quadruples lack an explicit aspect surface token or explicit opinion:<br/>"
          "• <b>Implicit Aspect (a = NULL):</b> <i>'በጣም ውድ ነው'</i> ('Very expensive') → Target is the implicit price/service.<br/>"
          "• <b>Implicit Opinion (o = NULL):</b> Fact-stating reviews expressing strong sentiment without sentiment adjectives.<br/>"
          "• <b>Extractive impossibility:</b> Extractive models cannot extract tokens that do not exist.", body_style)
    ]
    col3 = [
        P("<b>3. Open Vocabulary & Zero-Shot</b>", ParagraphStyle('H3', parent=body_bold, textColor=BRAND_BLUE)),
        Spacer(1, 3),
        P("Informal Amharic social reviews exhibit massive lexical diversity:<br/>"
          "• <b>66.2% of opinion spans</b> in the test set never appeared in the training corpus.<br/>"
          "• Phonetic spelling variations and slang (e.g. <i>'ዋው', 'ጀመረኝ'</i>) lead to high subword fragmentation under fixed vocabularies.<br/>"
          "• Requires robust morphological abstraction or byte-level modeling.", body_style)
    ]

    t_realities = Table([[col1, col2, col3]], colWidths=[235, 235, 235])
    t_realities.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CARD_BG),
        ('BOX', (0,0), (0,0), 0.8, colors.HexColor('#FCA5A5')),
        ('BOX', (1,0), (1,0), 0.8, colors.HexColor('#FCD34D')),
        ('BOX', (2,0), (2,0), 0.8, colors.HexColor('#93C5FD')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_realities)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 3: Phase 1 — The 6-Stage Modular Pipeline Architecture
    # =========================================================================
    story.extend(header_block(
        "Phase 1: Baseline Architecture",
        "The 6-Stage Modular Extractive Pipeline",
        "Initial decomposition of the 4D extraction task into sequential sub-tasks using pretrained Afro-XLM-R."
    ))

    p1_summary = P(
        "To tackle the combinatorial complexity of extracting (a, c, s, o) simultaneously, Phase 1 engineered an isolated, modular 6-stage pipeline. "
        "Each stage was developed and tuned as an independent neural classifier relying on the output of upstream stages.",
        body_style
    )
    story.append(p1_summary)
    story.append(Spacer(1, 8))

    # Pipeline stages table
    pipeline_stages = [
        ["Stage", "Sub-Task Function", "Neural Architecture & Representation", "Isolated Stage Metric"],
        ["Stage 1", "Span Extraction\n(ATE & OTE)", "Afro-XLM-R + Linear Token Classifier (BIO tagging for Aspect & Opinion)", "ATE F1: 50.6%\nOTE F1: 46.2%"],
        ["Stage 2", "Candidate Span Pairing", "Cartesian Product (N_t × N_o) + Syntactic / Positional Distance Filter", "Pair Candidate Recall: 68.0%"],
        ["Stage 3", "Aspect Category Detection", "Afro-XLM-R Contextual Pair Encoder + 29-Class Softmax Classifier", "Category F1: 71.0%"],
        ["Stage 4", "Sentiment Polarity", "Pair Context Representation + 3-Class Softmax (POS, NEG, NEU)", "Sentiment Macro F1: 88.0%"],
        ["Stage 5", "Implicit Span Recovery", "Heuristic Null-Aspect Assignment based on sentence-level dominant categories", "Implicit Recovery F1: 18.5%"],
        ["Stage 6", "Quad Assembly & Filter", "Constraint-Satisfaction Aggregator & Quadruple Confidence Scorer", "Full Quad Exact F1: 14.2%"]
    ]

    t_pipe = Table([[P(f"<b>{c}</b>" if r==0 else c.replace('\n', '<br/>'), body_style) for c in row] for r, row in enumerate(pipeline_stages)],
                   colWidths=[65, 155, 360, 140])
    t_pipe.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_pipe)
    story.append(Spacer(1, 8))

    why_modular = [
        P("<b>Architectural Rationale & Motivation:</b> Modularity allowed fine-grained error inspection and specialized loss functions per stage. "
          "Afro-XLM-R provided cross-lingual contextual embeddings for African languages. However, the system hid a catastrophic systemic flaw.", body_style)
    ]
    story.append(card_box("", why_modular, bg_color=BRAND_LIGHT, border_color=colors.HexColor('#BFDBFE')))
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 4: Phase 1 Autopsy: The Error Compounding Collapse
    # =========================================================================
    story.extend(header_block(
        "Phase 1 Autopsy & Diagnostic",
        "The Error Compounding Law in Cascaded Pipelines",
        "Mathematical proof of how independent stage successes guarantee end-to-end failure in information extraction."
    ))

    col_left_text = [
        P("<b>The Cascading Bottleneck Formula:</b>", body_bold),
        Spacer(1, 2),
        P("In any strictly sequential pipeline without bidirectional feedback, end-to-end quadruple recall is upper-bounded by the product of individual stage recalls:", body_style),
        Spacer(1, 4),
        P("<b>Recall<sub>e2e</sub> ≤ Recall<sub>ATE</sub> × Recall<sub>Pair</sub> × Recall<sub>Cat</sub> × Recall<sub>Sent</sub></b>", math_style),
        Spacer(1, 6),
        P("<b>The Zero-Error-Recovery Theorem:</b>", body_bold),
        P("• If Stage 1 misses an aspect term (e.g. <i>Recall = 50.6%</i>), that true quadruple is <b>irrevocably lost</b>. Stages 2, 3, and 4 can never evaluate it.<br/>"
          "• Stage 2 pairing retains only 68% of surviving candidates: <i>0.506 × 0.680 = 34.4%</i>.<br/>"
          "• Category classification retains 71%: <i>0.344 × 0.710 = 24.4%</i>.<br/>"
          "• Sentiment classification retains 88%: <i>0.244 × 0.880 = 21.5%</i>.<br/>"
          "• Assembly and precision penalties drive final exact F1 to <b>14.2%</b>.", body_style),
        Spacer(1, 6),
        P("<b>Critical Takeaway:</b> High isolated stage accuracy creates a false sense of security. Cascaded extractive pipelines are fundamentally unsuitable for fine-grained 4D tuple extraction.", ParagraphStyle('Takeaway', parent=body_style, textColor=ACCENT_RED))
    ]

    card_left = card_box("", col_left_text, bg_color=CARD_BG, border_color=BORDER_COLOR, width=340)

    fig1_path = "docs/figures/fig1_pipeline_cascade.png"
    if os.path.exists(fig1_path):
        img_fig1 = Image(fig1_path, width=355, height=192)
    else:
        img_fig1 = P("Figure 1 not found", body_style)

    t_slide4 = Table([[card_left, img_fig1]], colWidths=[355, 365])
    t_slide4.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_slide4)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 5: Phase 2 — Joint Dual-Channel Synchronized Labeling (SDRN)
    # =========================================================================
    story.extend(header_block(
        "Phase 2: Transition to Joint Modeling",
        "Synchronous Dual-Channel Recurrent Networks (SDRN)",
        "Coupling Aspect and Opinion extraction via cross-channel interaction matrices and linear-chain CRFs."
    ))

    p2_intro = P(
        "To break the cascading pipeline bottleneck, Phase 2 transitioned to <b>Joint Modeling</b> using the Synchronous Dual-channel Recurrent Network (SDRN). "
        "Instead of extracting aspects and opinions in isolation, SDRN deploys dual BiLSTM encoders with interactive cross-channel synchronization.",
        body_style
    )
    story.append(p2_intro)
    story.append(Spacer(1, 8))

    sdrn_math = [
        P("<b>Architectural & Mathematical Mechanics of SDRN:</b>", body_bold),
        Spacer(1, 3),
        P("<b>1. Dual Recurrent Encoders:</b> Input word embeddings are routed into twin BiLSTMs yielding target hidden states <b>H<sup>t</sup> = [h₁<sup>t</sup>, ..., hₙ<sup>t</sup>]</b> and opinion hidden states <b>H<sup>o</sup> = [h₁<sup>o</sup>, ..., hₙ<sup>o</sup>]</b>.", body_style),
        Spacer(1, 3),
        P("<b>2. Cross-Channel Synchronization Matrix:</b> Computes pairwise mutual semantic affinity between token <i>i</i> and token <i>j</i>:<br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;<b>M<sub>ij</sub> = tanh(W<sub>t</sub> h<sub>i</sub><sup>t</sup> + W<sub>o</sub> h<sub>j</sub><sup>o</sup> + b) ∈ R<sup>d</sup></b><br/>"
          "Interactive attention weights are derived: <b>α<sub>ij</sub> = softmax(v<sup>T</sup> M<sub>ij</sub>)</b>, updating target states with opinion context.", math_style),
        Spacer(1, 3),
        P("<b>3. Linear-Chain CRF Decoders:</b> Sequential CRF inference layers enforce valid label transitions over BIO tags:<br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;<b>s(y, x) = ∑<sub>i=1</sub><sup>n</sup> A<sub>y<sub>i-1</sub>, y<sub>i</sub></sub> + ∑<sub>i=1</sub><sup>n</sup> P<sub>i, y<sub>i</sub></sub></b>, resolved via Viterbi decoding.", math_style)
    ]
    story.append(card_box("", sdrn_math, bg_color=CARD_BG, border_color=BORDER_COLOR))
    story.append(Spacer(1, 8))

    sdrn_eval = Table([
        [
            P("<b>What SDRN Solved:</b><br/>• Replaced 6 separate models with a unified multi-task network.<br/>• Allowed aspect representations to condition opinion representations.<br/>• AOPE Pair F1 rose to <b>34.85%</b> (more than double the pipeline).", body_style),
            P("<b>The Hidden Structural Flaw:</b><br/>• Relied entirely on token-level sequence labeling (BIO tags).<br/>• Rigid markovian transition matrix penalized complex Amharic morphology.<br/>• Imposed an invisible ceiling on pair candidate recall.", ParagraphStyle('Err', parent=body_style, textColor=ACCENT_RED))
        ]
    ], colWidths=[355, 355])
    sdrn_eval.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), ACCENT_LIGHT_GREEN),
        ('BACKGROUND', (1,0), (1,0), ACCENT_LIGHT_RED),
        ('BOX', (0,0), (0,0), 0.8, ACCENT_GREEN),
        ('BOX', (1,0), (1,0), 0.8, ACCENT_RED),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(sdrn_eval)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 6: Phase 2 Autopsy: Linear-Chain CRF & BIO Fragmentation Ceiling
    # =========================================================================
    story.extend(header_block(
        "Phase 2 Autopsy & Diagnostic",
        "The BIO Token Fragmentation Ceiling (33.04% Recall Cap)",
        "Why token-level sequence labeling collapses on morphologically rich, multi-word Semitic expressions."
    ))

    sdrn_autopsy_text = [
        P("<b>The 3 Fatal Failure Modes of Token-Level BIO Tagging:</b>", body_bold),
        Spacer(1, 3),
        P("• <b>Subword Boundary Jitter:</b> Amharic agglutinative clitics force tokenizers to split words into 2-4 subwords. A single misclassified subword (e.g. <i>'O'</i> instead of <i>'I-OPN'</i>) fractures the span into invalid fragments.", body_style),
        P("• <b>Devastating Candidate Ceiling:</b> Quantitative inspection of the SDRN candidate generation pool revealed that only <b>33.04% of ground truth aspect-opinion pairs</b> were successfully co-extracted by the CRF. <b>66.96% of pairs were discarded before relation pairing ever began!</b>", ParagraphStyle('Ceil', parent=body_style, textColor=ACCENT_RED)),
        P("• <b>Non-Contiguous & Multi-Word Failures:</b> Amharic opinion expressions average 2.4 tokens (e.g. <i>'በጣም ደስ የሚል'</i>). The CRF frequently predicted disjoint tags: <i>[B-OPN, O, I-OPN]</i>.", body_style),
        Spacer(1, 4),
        P("<b>Verdict:</b> Token-level BIO tagging is fundamentally incompatible with Amharic syntax. The entire paradigm had to shift to <b>Direct Span Enumeration</b>.", body_bold)
    ]
    card_sdrn_autopsy = card_box("", sdrn_autopsy_text, bg_color=CARD_BG, border_color=BORDER_COLOR, width=340)

    fig2_path = "docs/figures/fig2_ceiling_recall.png"
    if os.path.exists(fig2_path):
        img_fig2 = Image(fig2_path, width=355, height=196)
    else:
        img_fig2 = P("Figure 2 not found", body_style)

    t_slide6 = Table([[card_sdrn_autopsy, img_fig2]], colWidths=[355, 365])
    t_slide6.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_slide6)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 7: Phase 3 — The Span-ASTE Paradigm Shift (Span Enumeration)
    # =========================================================================
    story.extend(header_block(
        "Phase 3: The Span Paradigm Shift",
        "Span-ASTE: Direct Span-Level Enumeration & Pruning",
        "Eliminating BIO tagging entirely by treating arbitrary word n-grams as atomic semantic candidates."
    ))

    p3_intro = P(
        "To overcome the 33.04% candidate recall ceiling, Phase 3 introduced <b>Span-ASTE</b> (Xu et al., 2021) adapted for Amharic. "
        "Instead of predicting tags per token, the model explicitly enumerates all continuous spans up to length L_max = 8 and models relations directly between span pairs.",
        body_style
    )
    story.append(p3_intro)
    story.append(Spacer(1, 8))

    aste_steps = [
        ["Phase", "Algorithmic Formulation", "Mathematical Definition & Complexity"],
        ["1. Span Enumeration", "Enumerate all candidate spans up to maximum width L_max = 8",
         "S = {(s, e) | 1 ≤ s ≤ e ≤ n, e - s + 1 ≤ L_max}<br/>Total candidate spans: |S| = O(n · L_max)"],
        ["2. Boundary Representation", "Represent each span using start token, end token, and width embedding",
         "h_span = [x_start; x_end; f_width] ∈ R^(2d + d_w)<br/>x_i: Contextual token embedding from Afro-XLM-R (768-d)"],
        ["3. Top-k Span Pruning", "Rank spans by span classification logits and retain only top-k candidates",
         "k = ⌊λ · n⌋ (where λ = 1.5 per sentence)<br/>Reduces candidate space from O(n²) to manageable O(n)"],
        ["4. Pairwise Relation Scoring", "Form Cartesian product of pruned aspect and opinion spans for relation classification",
         "R(s_t, s_o) = Softmax(MLP([h_s_t; h_s_o; f_dist]))<br/>Classes: {INVALID, POS, NEG, NEU}"]
    ]

    t_aste = Table([[P(f"<b>{c}</b>" if r==0 else c, body_style) for c in row] for r, row in enumerate(aste_steps)],
                   colWidths=[100, 260, 360])
    t_aste.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_aste)
    story.append(Spacer(1, 8))

    impact_card = [
        P("<b>Breakthrough Impact of Span Enumeration:</b> Candidate ceiling recall immediately skyrocketed from <b>33.04% (SDRN) to 81.10% (Span-ASTE)</b>! "
          "The model was finally capable of considering the vast majority of true aspect-opinion pairs in Amharic reviews.", ParagraphStyle('Impact', parent=body_style, textColor=ACCENT_GREEN))
    ]
    story.append(card_box("", impact_card, bg_color=ACCENT_LIGHT_GREEN, border_color=ACCENT_GREEN))
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 8: Span-ASTE Run 1: Diagnostics & Threshold Calibration
    # =========================================================================
    story.extend(header_block(
        "Span-ASTE Run 1: Diagnostic & Calibration",
        "Uncovering the Extreme Class Imbalance & Threshold Sweeps",
        "Diagnosing why high candidate recall (81.1%) yielded low pair recall (22.7%) under default decision boundaries."
    ))

    r1_col1 = [
        P("<b>Run 1 Baseline Results (Default Threshold τ = 0.50):</b>", body_bold),
        Spacer(1, 3),
        P("• Aspect Detection (ATE): <b>P: 51.52%, R: 49.69%, F1: 50.59%</b><br/>"
          "• Opinion Detection (OTE): <b>P: 47.90%, R: 39.81%, F1: 43.48%</b><br/>"
          "• AOPE Pair Extraction: <b>P: 47.93%, R: 22.68%, F1: 30.80%</b><br/>"
          "• ASTE Triplet Extraction: <b>P: 42.15%, R: 19.95%, F1: 27.09%</b>", body_style),
        Spacer(1, 4),
        P("<b>The Diagnostic Mystery:</b> Why was Pair Recall (22.68%) less than half of Precision (47.93%) despite an 81.1% candidate pool?", ParagraphStyle('Q', parent=body_style, textColor=ACCENT_AMBER)),
        Spacer(1, 3),
        P("<b>Root Cause: 96.8% Class Skew:</b> In the pruned span pair matrix, over 96.8% of span pairs have no sentiment relation (`INVALID`). "
          "Cross-entropy training trained the model to be excessively conservative, outputting probabilities between 0.25 and 0.45 for true relations.", body_style)
    ]
    card_r1_left = card_box("", r1_col1, bg_color=CARD_BG, border_color=BORDER_COLOR, width=340)

    # Right: Threshold Sweep Table
    sweep_rows = [
        ["Relation Threshold (τ)", "Valid Pairs Captured", "Pair Precision", "Pair Recall", "Pair F1"],
        ["τ = 0.50 (Default)", "290", "47.93%", "22.68%", "30.80%"],
        ["τ = 0.40", "362", "42.10%", "26.50%", "32.51%"],
        ["τ = 0.30 (Optimal τ*)", "422", "37.50%", "32.06%", "34.54% (+3.74)"],
        ["τ = 0.20", "501", "31.20%", "36.20%", "33.51%"],
        ["τ = 0.10", "680", "22.10%", "41.50%", "28.84%"]
    ]
    t_sweep = Table([[P(f"<b>{c}</b>" if r==0 else (f"<b>{c}</b>" if 'Optimal' in c else c), body_style) for c in row] for r, row in enumerate(sweep_rows)],
                    colWidths=[110, 55, 55, 55, 70])
    t_sweep.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('BACKGROUND', (0,3), (-1,3), ACCENT_LIGHT_GREEN),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))

    card_r1_right = card_box("Empirical Threshold Calibration Discovery", [
        t_sweep,
        Spacer(1, 4),
        P("<b>Finding:</b> Setting τ* = 0.30 calibrated the model to the severe class imbalance, immediately unlocking <b>+132 additional valid pairs</b> without retraining!", body_style)
    ], bg_color=CARD_BG, border_color=BORDER_COLOR, width=365)

    t_slide8 = Table([[card_r1_left, card_r1_right]], colWidths=[355, 365])
    t_slide8.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_slide8)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 9: Phase 4 — Deep Biaffine Attention & Span Synchronization (Run 2)
    # =========================================================================
    story.extend(header_block(
        "Phase 4: Advanced Relation Modeling",
        "Deep Biaffine Bilinear Attention & Span Synchronization (Run 2)",
        "Replacing naive concatenation MLPs with multiplicative bilinear scoring tensors (Dozat 2017, Nguyen 2018)."
    ))

    biaffine_math = [
        P("<b>Why Bilinear Tensors Over Concat MLPs:</b>", body_bold),
        Spacer(1, 2),
        P("Concat MLPs compute additive features: W₁ h_t + W₂ h_o. They fail to model subtle semantic affinities between aspect and opinion heads. "
          "Deep Biaffine attention computes direct multiplicative cross-space interactions:", body_style),
        Spacer(1, 3),
        P("<b>1. Bilinear Tensor Scoring:</b><br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;<b>S<sub>ij, c</sub><sup>bilin</sup> = (h<sub>i</sub><sup>t</sup>)<sup>T</sup> U<sub>c</sub> h<sub>j</sub><sup>o</sup></b><br/>"
          "where U<sub>c</sub> ∈ R<sup>d × d</sup> is a class-specific tensor parameter for class c.", math_style),
        Spacer(1, 3),
        P("<b>2. Affine Context Synchronization:</b><br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;<b>S<sub>ij, c</sub><sup>aff</sup> = W<sub>c</sub> [h<sub>i</sub><sup>t</sup>; h<sub>j</sub><sup>o</sup>; h̃<sub>i</sub><sup>t</sup>; h̃<sub>j</sub><sup>o</sup>] + b<sub>c</sub></b><br/>"
          "where h̃<sub>i</sub><sup>t</sup> = ∑<sub>j</sub> A<sub>ij</sub> h<sub>j</sub><sup>o</sup> injects bidirectional cross-span attention context.", math_style),
        Spacer(1, 3),
        P("<b>3. Total Scoring:</b> <b>S<sub>ij</sub> = S<sub>ij</sub><sup>bilin</sup> + S<sub>ij</sub><sup>aff</sup></b>, evaluated over 4 relation classes.", math_style)
    ]
    card_biaf_left = card_box("", biaffine_math, bg_color=CARD_BG, border_color=BORDER_COLOR, width=340)

    fig3_path = "docs/figures/fig3_biaffine_tensor.png"
    if os.path.exists(fig3_path):
        img_fig3 = Image(fig3_path, width=355, height=172)
    else:
        img_fig3 = P("Figure 3 not found", body_style)

    t_slide9 = Table([[card_biaf_left, img_fig3]], colWidths=[355, 365])
    t_slide9.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_slide9)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 10: Run 2 Breakthrough & Boundary Truncation Bottleneck
    # =========================================================================
    story.extend(header_block(
        "Run 2 Results & Diagnostic Autopsy",
        "Biaffine Breakthrough & The Boundary Truncation Bottleneck",
        "Surpassing Run 1 across all metrics, but uncovering the multi-word span boundary representation flaw."
    ))

    r2_comp = [
        ["Metric Category", "Run 1 (MLP Concat)", "Run 2 (Deep Biaffine)", "Absolute Delta (Run 2 vs 1)"],
        ["Aspect Detection (ATE F1)", "50.59%", "53.76%", "+3.17% (Significant gain)"],
        ["Opinion Detection (OTE F1)", "43.48%", "45.02%", "+1.54%"],
        ["AOPE Pair F1 (Default τ=0.50)", "32.06%", "34.02%", "+1.96% (+183 pairs captured)"],
        ["ASTE Triplet F1 (Default τ=0.50)", "27.09%", "29.37%", "+2.28% (Triplets up to 29.4%)"],
        ["Valid Pairs Captured (at τ*=0.30)", "422 pairs", "605 pairs", "+183 additional pairs (+43.4%)"]
    ]
    t_r2 = Table([[P(f"<b>{c}</b>" if r==0 else c, body_style) for c in row] for r, row in enumerate(r2_comp)],
                 colWidths=[180, 150, 160, 210])
    t_r2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_r2)
    story.append(Spacer(1, 8))

    r2_flaw = [
        P("<b>The Boundary Truncation Bottleneck:</b>", ParagraphStyle('HFlaw', parent=body_bold, textColor=ACCENT_RED)),
        Spacer(1, 2),
        P("Rigorous qualitative inspection of Run 2 prediction errors revealed a glaring algorithmic weakness in the span representation formula:<br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;<b>h<sub>span</sub> = [x<sub>start</sub>; x<sub>end</sub>; f<sub>width</sub>]</b><br/>"
          "• <b>Complete Loss of Internal Context:</b> For multi-word Amharic opinions (e.g. <i>'በጣም ጥሩ አይደለም'</i> = 'very good not'), "
          "only the first token (<i>'በጣም'</i>) and last token (<i>'አይደለም'</i>) were encoded. The critical sentiment carrier (<i>'ጥሩ'</i>) was completely ignored!<br/>"
          "• Resulted in severe boundary mismatch: the model predicted <i>'ጥሩ'</i> as positive, missing the negation <i>'አይደለም'</i>.<br/>"
          "• <b>Conclusion:</b> Spans must aggregate representations across <i>all</i> intermediate tokens without exploding inference compute.", body_style)
    ]
    story.append(card_box("", r2_flaw, bg_color=CARD_BG, border_color=colors.HexColor('#FCA5A5')))
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 11: Phase 5: Path A — O(1) Prefix-Sum Span Mean Pooling (Run 3)
    # =========================================================================
    story.extend(header_block(
        "Phase 5: Architectural Innovation (Path A)",
        "Vectorized O(1) Prefix-Sum Span Mean-Pooling & Cost-Sensitive Loss",
        "Capturing complete internal span semantics with zero asymptotic overhead, coupled with Focal loss re-weighting."
    ))

    pool_math = [
        P("<b>Vectorized O(1) Cumulative Prefix-Sum:</b>", body_bold),
        Spacer(1, 2),
        P("To capture every token in a multi-word span without O(L · n²) re-computation, we construct cumulative prefix sums of encoder representations:", body_style),
        Spacer(1, 3),
        P("<b>P[t] = ∑<sub>k=1</sub><sup>t</sup> x<sub>k</sub> &nbsp;⟹&nbsp; h<sub>mean</sub> = (P[e] - P[s]) / (e - s + 1)</b>", math_style),
        P("<b>h<sub>span</sub> = [x<sub>start</sub>; x<sub>end</sub>; h<sub>mean</sub>; f<sub>width</sub>] ∈ R<sup>2324</sup></b>", math_style),
        Spacer(1, 3),
        P("<b>Cost-Sensitive Class Re-weighting:</b>", body_bold),
        P("To directly counter the 96.8% invalid class skew at training time without heuristic thresholds, we integrated Focal Loss re-weighting:<br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;<b>L<sub>focal</sub> = -α<sub>t</sub> (1 - p<sub>t</sub>)<sup>γ</sup> log(p<sub>t</sub>)</b><br/>"
          "with class penalty weights: <b>w = [INVALID: 0.25, POS: 2.5, NEG: 2.5, NEU: 3.0]</b>.", body_style)
    ]
    card_pool_left = card_box("", pool_math, bg_color=CARD_BG, border_color=BORDER_COLOR, width=340)

    fig4_path = "docs/figures/fig4_prefix_meanpool.png"
    if os.path.exists(fig4_path):
        img_fig4 = Image(fig4_path, width=355, height=160)
    else:
        img_fig4 = P("Figure 4 not found", body_style)

    t_slide11 = Table([[card_pool_left, img_fig4]], colWidths=[355, 365])
    t_slide11.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_slide11)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 12: Run 3 Results & The Inherent Extractive Ceiling
    # =========================================================================
    story.extend(header_block(
        "Run 3 Results & Paradigm Limit",
        "Upstream State-of-the-Art & The Inherent Extractive Ceiling",
        "Mean-pooling achieved project-best upstream detection, revealing the ultimate theoretical boundary of extractive modeling."
    ))

    r3_metrics = [
        ["Sub-Task / Metric", "Run 1 (Baseline)", "Run 2 (Biaffine)", "Run 3 (MeanPool+Focal)", "Scientific Significance"],
        ["Aspect Term Extraction (ATE F1)", "50.59%", "53.76%", "<b>55.20%</b>", "<b>Project SOTA (+4.61% over baseline)</b>"],
        ["Opinion Term Recall (OTE Recall)", "39.81%", "41.60%", "<b>46.70%</b>", "<b>+6.89% boost in multi-word opinions</b>"],
        ["Candidate Ceiling Recall", "81.10%", "80.46%", "<b>81.91%</b>", "Near-optimal upper-bound retention"],
        ["Calibrated Pair Recall (τ=0.30)", "32.06%", "34.02%", "<b>45.78%</b>", "Captured 640 true positive pairs"],
        ["Implicit Quadruple Extraction F1", "0.00%", "0.00%", "0.00%", "<b>Catastrophic Zero Recall on 42.7% of data</b>"]
    ]
    t_r3 = Table([[P(c, body_style) for c in row] for row in r3_metrics],
                 colWidths=[175, 100, 105, 120, 200])
    t_r3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('BACKGROUND', (0,5), (-1,5), ACCENT_LIGHT_RED),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_r3)
    story.append(Spacer(1, 8))

    ceiling_card = [
        P("<b>The Extractive Wall: Why Extractive Architectures Cannot Solve ACOS:</b>", ParagraphStyle('HWall', parent=body_bold, textColor=ACCENT_RED)),
        Spacer(1, 2),
        P("By definition, an extractive span model selects continuous token slices s = [x_i, ..., x_j] from the input string X.<br/>"
          "• In Amharic ACOS, <b>42.7% of all quads have implicit targets (a = NULL) or implicit opinions (o = NULL)</b>.<br/>"
          "• Because NULL has no token coordinate in X, extractive models assign <b>probability 0.00</b> to every implicit tuple.<br/>"
          "• No matter how sophisticated the biaffine tensor or prefix-sum pooling becomes, the extractive model is <b>mathematically capped at 57.3% total quad recall</b>.<br/>"
          "• <b>Strategic Pivot:</b> The entire extractive paradigm reached its theoretical limit. The only viable path forward was <b>Sequence-to-Sequence Generation</b>.", body_style)
    ]
    story.append(card_box("", ceiling_card, bg_color=CARD_BG, border_color=colors.HexColor('#F87171')))
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 13: Phase 6: Generative Shift — ByT5-Base for End-to-End ACOS
    # =========================================================================
    story.extend(header_block(
        "Phase 6: Generative Foundation Model",
        "The Paradigm Shift to Generative ByT5-Base",
        "Formulating quadruple extraction as autoregressive byte-level sequence generation."
    ))

    p6_intro = P(
        "Phase 6 executed a fundamental architectural shift: discarding extractive span pipelines in favor of a unified <b>Sequence-to-Sequence Generative Foundation Model</b> using <b>ByT5-Base</b> (Xue et al., 2022). "
        "The model directly maps an Amharic input string into a structured, linearized string of sentiment quadruples.",
        body_style
    )
    story.append(p6_intro)
    story.append(Spacer(1, 8))

    gen_form = [
        P("<b>Autoregressive Target Linearization Formulation:</b>", body_bold),
        Spacer(1, 3),
        P("<b>P(Y | X) = ∏<sub>m=1</sub><sup>M</sup> P(y<sub>m</sub> | y<sub>&lt;m</sub>, X)</b> &nbsp;&nbsp;where Y is linearized as:<br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;<code>[ [Aspect] | [Category] | [Sentiment] | [Opinion] ] ; [ [Aspect] | [Category] | ... ]</code>", math_style),
        Spacer(1, 3),
        P("<b>Example:</b><br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;Input: <i>'ምግቡ በጣም ጥሩ አይደለም ግን አገልግሎቱ ፈጣን ነው'</i><br/>"
          "&nbsp;&nbsp;&nbsp;&nbsp;Output: <code>[ ምግቡ | FOOD#QUALITY | NEGATIVE | በጣም ጥሩ አይደለም ] ; [ አገልግሎቱ | SERVICE#GENERAL | POSITIVE | ፈጣን ]</code>", math_style)
    ]
    story.append(card_box("", gen_form, bg_color=CARD_BG, border_color=BORDER_COLOR))
    story.append(Spacer(1, 8))

    b1 = [
        P("<b>1. Raw UTF-8 Byte Processing</b>", body_bold),
        Spacer(1, 2),
        P("Operates directly on 256 byte IDs. No SentencePiece or WordPiece tokenizer. Preserves Ge'ez morphological clitics at the byte level without arbitrary splitting.", body_style)
    ]
    b2 = [
        P("<b>2. Strict 0% OOV Rate</b>", body_bold),
        Spacer(1, 2),
        P("Every possible Amharic word, unseen slang, misspelling, and punctuation mark maps to valid UTF-8 bytes. Zero <code>&lt;unk&gt;</code> tokens, solving the 66% unseen opinion challenge.", body_style)
    ]
    b3 = [
        P("<b>3. Native Implicit NULL Generation</b>", ParagraphStyle('B3', parent=body_bold, textColor=ACCENT_GREEN)),
        Spacer(1, 2),
        P("The autoregressive decoder freely emits literal `NULL` strings for implicit aspects or opinions without requiring ad-hoc pipeline stages. Seamless implicit quad handling.", body_style)
    ]
    t_byt5 = Table([[b1, b2, b3]], colWidths=[235, 235, 235])
    t_byt5.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CARD_BG),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (2,0), (2,0), ACCENT_LIGHT_GREEN),
        ('BOX', (2,0), (2,0), 0.8, ACCENT_GREEN),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_byt5)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 14: Engineering ByT5: Resolving FP16 NaN Collapse
    # =========================================================================
    story.extend(header_block(
        "ByT5 Engineering & Optimization",
        "Overcoming the FP16 NaN Collapse & Scaling Validation",
        "Hardware-level stabilization: BFloat16 dynamic range, Adafactor memory reduction, and fast greedy evaluation."
    ))

    col_fail = [
        P("<b>The Epoch 1 Training Failure:</b>", ParagraphStyle('HFail', parent=body_bold, textColor=ACCENT_RED)),
        Spacer(1, 3),
        P("During the initial ByT5-Base training run on GPU:<br/>"
          "• ByT5 byte sequences are ~3-4× longer than subword sequences (512 max bytes).<br/>"
          "• In standard FP16 mixed precision, byte embedding gradients caused <b>catastrophic arithmetic underflow and exponent overflow</b> (> 65,504).<br/>"
          "• Training loss exploded to <b>NaN within step 350</b> of Epoch 1.<br/>"
          "• In addition, validation generation took <b>45 minutes per epoch</b> using beam search, making multi-epoch training infeasible.", body_style)
    ]
    card_fail = card_box("", col_fail, bg_color=CARD_BG, border_color=colors.HexColor('#FCA5A5'), width=340)

    col_fixes = [
        P("<b>The 3 Critical Engineering Breakthroughs:</b>", ParagraphStyle('HFix', parent=body_bold, textColor=ACCENT_GREEN)),
        Spacer(1, 3),
        P("<b>1. Precision Upgrade: BFloat16 (BF16):</b><br/>"
          "Replaced FP16 with BF16. BF16 allocates 8 exponent bits (same as FP32), expanding dynamic range from 10<sup>±4.8</sup> to 10<sup>±38</sup>. Completely eliminated gradient overflow and NaN crashes.", body_style),
        Spacer(1, 2),
        P("<b>2. Adafactor Optimizer (Factorized Second Moments):</b><br/>"
          "AdamW maintains two 32-bit states per parameter (16 bytes/param). Adafactor factorizes second moments (V ≈ R · C / mean(R)), saving <b>60% optimizer VRAM</b> and enabling batch size 8 + gradient accumulation 4.", body_style),
        Spacer(1, 2),
        P("<b>3. Fast Greedy Validation Engine:</b><br/>"
          "Replaced slow beam search with batched (batch size 16) greedy autoregressive decoding. <b>Validation epoch time dropped from 45 min to 90 seconds (30× speedup)</b>.", body_style)
    ]
    card_fixes = card_box("", col_fixes, bg_color=CARD_BG, border_color=colors.HexColor('#86EFAC'), width=365)

    t_slide14 = Table([[card_fail, card_fixes]], colWidths=[355, 365])
    t_slide14.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_slide14)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 15: Epoch 9 Breakthrough Analysis: Implicit Quad Recovery
    # =========================================================================
    story.extend(header_block(
        "ByT5 Dev-Set Milestone (Best Checkpoint — Epoch 9)",
        "Breaking the Implicit Barrier: First-Ever Implicit Quad Recovery",
        "Dev-set quantitative and qualitative evaluation of the Epoch 9 ByT5-Base model checkpoint."
    ))

    byt5_metrics = [
        ["Evaluation Metric", "Precision", "Recall", "F1 Score", "True Pos (TP)", "False Pos (FP)", "False Neg (FN)"],
        ["<b>Full Quadruple (A, C, S, O)</b>", "18.20%", "17.62%", "<b>17.90%</b>", "<b>182</b>", "818", "851"],
        ["<b>Implicit Quadruples (a=NULL / o=NULL)</b>", "<b>23.49%</b>", "<b>16.36%</b>", "<b>19.28%</b>", "<b>70</b>", "228", "358"],
        ["<b>Explicit Quadruples</b>", "15.95%", "18.51%", "17.14%", "112", "590", "493"],
        ["<b>Joint Category-Sentiment (C, S)</b>", "49.50%", "47.90%", "<b>48.70%</b>", "<b>495</b>", "505", "538"],
        ["<b>Aspect-Category-Sentiment (A, C, S)</b>", "30.30%", "29.33%", "<b>29.81%</b>", "303", "697", "730"],
        ["<b>AOPE Pair Extraction (A, O)</b>", "30.00%", "29.04%", "<b>29.51%</b>", "300", "700", "733"],
        ["<b>ASTE Triplet Extraction (A, S, O)</b>", "27.40%", "26.52%", "<b>26.96%</b>", "274", "726", "760"]
    ]
    t_byt5_m = Table([[P(c, body_style) for c in row] for row in byt5_metrics],
                     colWidths=[190, 75, 75, 75, 95, 95, 95])
    t_byt5_m.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('BACKGROUND', (0,2), (-1,2), ACCENT_LIGHT_GREEN),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_byt5_m)
    story.append(Spacer(1, 6))

    qual_sample = [
        P("<b>Qualitative Verification: Perfect Exact Match on Complex Implicit Sentence:</b>", body_bold),
        Spacer(1, 2),
        P("<b>Input:</b> <i>'ውህደቱ ለአገራዊ ለውጡ አስተዋጽኦ ያበረክታል በሚል ብዙዎች ተስፋ ጥለውበት የነበረ ቢሆንም በተግባር የታየው ግን ተስፋ አስቆረጣቸው።'</i><br/>"
          "<b>Ground Truth:</b> <code>[ ውህደቱ | GOVERNANCE#CITIZEN_ENGAGEMENT | NEGATIVE | ተስፋ አስቆረጣቸው ]</code><br/>"
          "<b>ByT5 Prediction:</b> <code>[ ውህደቱ | GOVERNANCE#CITIZEN_ENGAGEMENT | NEGATIVE | ተስፋ አስቆረጣቸው ]</code> &nbsp;<b>(Exact Match ✓)</b>", body_style),
        Spacer(1, 2),
        P("<b>Scientific Significance:</b> Extractive models scored <b>0.00% on implicit quads</b>. ByT5 extracted <b>70 true positive implicit quads</b> with <b>23.49% precision</b>, proving that byte-level autoregression successfully unifies explicit and implicit extraction.", ParagraphStyle('Sig', parent=body_style, textColor=ACCENT_GREEN))
    ]
    story.append(card_box("", qual_sample, bg_color=CARD_BG, border_color=colors.HexColor('#86EFAC')))
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 16: Comprehensive Research Benchmark & Strategic Roadmap
    # =========================================================================
    story.extend(header_block(
        "Master Benchmark & Future Roadmap",
        "Cross-Architecture Synthesis & Future Strategic Horizons",
        "Consolidated performance matrix across all 6 model generations and three concrete next steps."
    ))

    # Top: Master Table Full Width
    master_comp = [
        ["Architecture / Generation", "Paradigm Type", "Aspect F1", "AOPE Pair F1", "Full Quad F1", "Implicit Quad F1", "Core Scientific Finding"],
        ["1. 6-Stage Modular Pipeline", "Cascaded Extractive", "50.60%", "14.20%", "14.20%", "0.00%", "Error Compounding Collapse (F1 ≈ ∏ F1_i)"],
        ["2. Joint SDRN (BiLSTM+CRF)", "Token Joint Tagging", "48.00%", "34.85%", "N/A", "0.00%", "BIO Token Fragmentation Ceiling (33% Recall)"],
        ["3. Span-ASTE Run 1 (MLP)", "Span Enumeration", "50.59%", "30.80%", "N/A", "0.00%", "Ceiling raised to 81.1%; Imbalance skew"],
        ["4. Span-ASTE Run 2 (Biaffine)", "Span + Bilinear", "53.76%", "34.02%", "N/A", "0.00%", "Biaffine tensor boost; Boundary truncation"],
        ["5. Span-ASTE Run 3 (MeanPool)", "Span + Focal Pool", "<b>55.20%</b>", "<b>32.04%</b>", "N/A", "0.00%", "SOTA upstream detection; Extractive Wall"],
        ["6a. ByT5-Base (Dev Best Ep 9)", "Byte Autoregressive", "29.81%*", "29.51%", "<b>17.90%</b>", "<b>19.28%</b>", "First-ever Implicit Quad recovery"],
        ["6b. ByT5-Base (Final Test)", "Byte Autoregressive", "29.94%*", "26.69%", "<b>16.14%</b>", "<b>14.87%</b>", "<b>811 TPs / 274 implicit TPs on full test</b>"]
    ]
    t_master = Table([[P(c, body_style) for c in row] for row in master_comp],
                     colWidths=[150, 115, 65, 75, 75, 85, 155])
    t_master.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('BACKGROUND', (0,6), (-1,7), ACCENT_LIGHT_GREEN),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_master)
    story.append(Spacer(1, 4))
    story.append(P("<font size='7'>* Note: ByT5 Aspect metric reflects ACSE (Aspect-Category-Sentiment joint F1). Extractive ASTE models do not output 29-class categories natively.</font>", body_style))
    story.append(Spacer(1, 6))

    # Bottom Row: Left Figure 5, Right Horizons
    fig5_path = "docs/figures/fig5_cross_model_benchmark.png"
    if os.path.exists(fig5_path):
        img_fig5 = Image(fig5_path, width=355, height=175)
    else:
        img_fig5 = P("Figure 5 not found", body_style)

    horizons_content = [
        P("<b>Strategic Roadmap & Next Research Phases:</b>", body_bold),
        Spacer(1, 3),
        P("• <b>Horizon 1: Curriculum Fine-Tuning:</b> Pre-train ByT5 decoder on explicit quadruples first (stabilizing aspect extraction to match Afro-XLM-R's 55%) before joint fine-tuning on implicit NULL tuples.", body_style),
        Spacer(1, 2),
        P("• <b>Horizon 2: Constrained Trie / FSM Decoding:</b> Enforce a finite-state machine over the 29 aspect categories and 3 sentiment labels during autoregressive generation to eliminate category hallucinations.", body_style),
        Spacer(1, 2),
        P("• <b>Horizon 3: Foundation Model Scaling:</b> Scale from ByT5-Base (580M) to ByT5-Large (1.2B) using Low-Rank Adaptation (LoRA) to enrich contextual Semitic representations.", body_style)
    ]
    card_horizons = card_box("", horizons_content, bg_color=BRAND_LIGHT, border_color=colors.HexColor('#BFDBFE'), width=350)

    t_slide16_bot = Table([[img_fig5, card_horizons]], colWidths=[365, 355])
    t_slide16_bot.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_slide16_bot)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 17: Final Test Set Evaluation — Definitive ByT5-Base Results
    # =========================================================================
    story.extend(header_block(
        "Final Test Evaluation (10-Epoch ByT5-Base)",
        "Definitive Test Set Metrics: Full 5-Fold Cross-Validated Results",
        "Complete held-out test evaluation of the ByT5-Base model after 10 epochs of training (best checkpoint selected on dev set)."
    ))

    final_test_metrics = [
        ["Evaluation Metric", "Precision", "Recall", "F1 Score", "True Pos (TP)", "False Pos (FP)", "False Neg (FN)"],
        ["<b>Full Quadruple (A, C, S, O)</b>", "16.52%", "15.78%", "<b>16.14%</b>", "<b>811</b>", "4098", "4327"],
        ["<b>Explicit Quadruples</b>", "15.90%", "18.00%", "<b>16.88%</b>", "<b>537</b>", "2841", "2446"],
        ["<b>Implicit Quadruples (a=NULL / o=NULL)</b>", "<b>17.90%</b>", "<b>12.71%</b>", "<b>14.87%</b>", "<b>274</b>", "1257", "1881"],
        ["<b>AOPE Pair Extraction (A, O)</b>", "27.32%", "26.10%", "<b>26.69%</b>", "1341", "3568", "3797"],
        ["<b>ASTE Triplet Extraction (A, S, O)</b>", "24.24%", "23.16%", "<b>23.69%</b>", "1190", "3719", "3948"],
        ["<b>Aspect-Category-Sentiment (A, C, S)</b>", "30.64%", "29.27%", "<b>29.94%</b>", "1504", "3405", "3634"],
        ["<b>Joint Category-Sentiment (C, S)</b>", "49.40%", "47.20%", "<b>48.27%</b>", "<b>2425</b>", "2484", "2713"]
    ]
    t_final = Table([[P(c, body_style) for c in row] for row in final_test_metrics],
                     colWidths=[190, 75, 75, 75, 95, 95, 95])
    t_final.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('BACKGROUND', (0,3), (-1,3), ACCENT_LIGHT_GREEN),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_final)
    story.append(Spacer(1, 6))

    # Dev vs Test Comparison Summary
    dev_vs_test_content = [
        P("<b>Dev-Best vs. Final Test Comparison:</b>", body_bold),
        Spacer(1, 3),
        P("• <b>Full Quad F1:</b> Dev 17.90% → Test <b>16.14%</b> (−1.76 pts). Modest generalization gap confirms model stability.", body_style),
        P("• <b>Implicit Quad F1:</b> Dev 19.28% → Test <b>14.87%</b> (−4.41 pts). Implicit extraction remains the hardest sub-task; recall drops suggest long-tail distribution shift.", body_style),
        P("• <b>Cat-Sent F1:</b> Dev 48.70% → Test <b>48.27%</b> (−0.43 pts). Category+Sentiment joint classification is highly stable and generalizes robustly.", body_style),
        P("• <b>ACSE F1:</b> Dev 29.81% → Test <b>29.94%</b> (+0.13 pts). Aspect+Category+Sentiment triplets actually improved slightly on the test set.", body_style),
        Spacer(1, 3),
        P("<b>Key Takeaway:</b> The ByT5-Base generative model successfully produced <b>811 true positive full quadruples</b> and <b>274 implicit quadruples</b> on the full held-out test set — "
          "confirming that byte-level autoregressive generation is a viable paradigm for low-resource Amharic ACOS extraction, with substantial room for improvement via scaling, constrained decoding, and curriculum learning.",
          ParagraphStyle('FinalTakeaway', parent=body_style, textColor=ACCENT_GREEN))
    ]
    story.append(card_box("", dev_vs_test_content, bg_color=BRAND_LIGHT, border_color=colors.HexColor('#BFDBFE')))

    # Build the PDF
    print(f"Building presentation PDF: {output_filename} ...")
    doc.build(story, canvasmaker=PresentationCanvas)
    print("Presentation PDF generated successfully!")


if __name__ == '__main__':
    create_presentation_pdf()
