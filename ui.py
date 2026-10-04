"""
ui.py - Thành phần giao diện dùng chung (CSS + HTML nhỏ) cho app.py.
Bảng màu: nền xanh sân đêm, nhấn xanh cỏ #22C55E, vàng bóng #FFD60A.
"""
import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Be+Vietnam+Pro:wght@400;600;700&display=swap');
:root { --accent:#22C55E; --ball:#FFD60A; --card:#0F2418; --line:rgba(255,255,255,.08); --muted:#9DB8A6;
  --display:'Barlow Condensed','Arial Narrow',sans-serif;   /* tiêu đề, số lớn - có đủ dấu tiếng Việt */
  --body:'Be Vietnam Pro',system-ui,sans-serif; }
/* Font nội dung chỉ áp cho chữ, KHÔNG áp mọi phần tử (giữ font icon của Streamlit) */
.stApp p, .stApp label, .stMarkdown, .stTabs button p, .stTabs [data-testid="stTab"] p, .card, .score { font-family:var(--body); }
.block-container { padding-top: 1.2rem; max-width: 1280px; }

/* HERO: dải sân cỏ có vạch giữa sân */
.hero { position:relative; overflow:hidden; border-radius:20px; padding:28px 32px; margin-bottom:18px;
  background:
    radial-gradient(circle at 50% 50%, transparent 58px, rgba(255,255,255,.18) 59px, rgba(255,255,255,.18) 61px, transparent 62px),
    linear-gradient(90deg, transparent calc(50% - 1px), rgba(255,255,255,.18) calc(50% - 1px), rgba(255,255,255,.18) calc(50% + 1px), transparent calc(50% + 1px)),
    repeating-linear-gradient(90deg, #14532d 0 80px, #166534 80px 160px);
  box-shadow: 0 10px 30px rgba(0,0,0,.35); }
.hero h1 { font-family:var(--display); font-weight:800; letter-spacing:1px; text-transform:none; font-size:3rem; margin:0; color:#fff;
  text-shadow:0 3px 12px rgba(0,0,0,.5); }
.hero p { margin:.3rem 0 .8rem; color:#e6ffe9; max-width:720px; }
.badge { display:inline-block; padding:4px 12px; margin:0 6px 6px 0; border-radius:999px; font-size:.8rem;
  background:rgba(0,0,0,.35); border:1px solid rgba(255,255,255,.25); color:#fff; }

/* SCOREBOARD possession */
.score { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:16px 20px; margin-bottom:14px; }
.score-top { display:flex; justify-content:space-between; align-items:center; font-family:var(--display); font-weight:800; font-size:2.4rem; }
.score-mid { color:var(--muted); font-family:var(--body); font-weight:400; font-size:.8rem; letter-spacing:2px; text-transform:uppercase; }
.bar { display:flex; height:14px; border-radius:999px; overflow:hidden; margin-top:8px; box-shadow:inset 0 0 0 1px var(--line); }
.bar > div { transition: width 1.2s ease; }
.dot { display:inline-block; width:14px; height:14px; border-radius:50%; margin-right:8px; vertical-align:middle;
  border:2px solid rgba(255,255,255,.6); }

/* STAT CARDS */
.card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:16px 18px; height:100%;
  transition: transform .15s ease, border-color .15s ease; }
.card:hover { transform: translateY(-3px); border-color: var(--accent); }
.card .lbl { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:1px; }
.card .val { font-family:var(--display); font-weight:800; font-size:2.3rem; line-height:1.1; color:#fff; }
.card .sub { color:var(--muted); font-size:.8rem; }
.tip { cursor:help; color:var(--muted); font-size:.8rem; margin-left:4px; }
.sec { font-family:var(--display); font-weight:700; text-transform:uppercase; letter-spacing:1px; font-size:1.6rem; margin:10px 0 4px; }

/* Tabs */
/* Streamlit 1.65: tablist/tab dùng role + data-testid (không còn data-baseweb) */
.stTabs [role="tablist"] { gap:6px; }
.stTabs [data-testid="stTab"] { background:var(--card); border-radius:10px 10px 0 0; padding:8px 16px; }
.stTabs [aria-selected="true"] { background:#14532d; }
@media (max-width: 640px) {
  .hero { padding:20px 18px; }
  .hero h1 { font-size:2.1rem; }
  /* Scoreboard: nhãn lên 1 dòng riêng, 2 đội chia 2 bên (tránh "MAN / UNITED" bị gãy dòng) */
  .score { padding:12px 14px; }
  .score-top { flex-wrap:wrap; row-gap:2px; font-size:1.3rem; line-height:1.15; }
  .score-mid { order:-1; flex-basis:100%; text-align:center; font-size:.75rem; }
  .score-top > span:not(.score-mid) { white-space:nowrap; }
  .dot { width:11px; height:11px; margin-right:5px; }
  /* Tabs: xuống dòng thay vì trượt ngang (2 tab cuối bị khuất ở 390px) */
  .stTabs [role="tablist"] { flex-wrap:wrap; row-gap:6px; overflow-x:visible; }
  .stTabs [data-testid="stTab"] { padding:6px 10px; border-radius:10px; }
  .stTabs [aria-selected="true"] { box-shadow:inset 0 -2px 0 var(--accent); }
  /* Plotly: nhãn ID cầu thủ 9px quá nhỏ trên điện thoại */
  .js-plotly-plot .textpoint text { font-size:11px !important; }
  .js-plotly-plot { overflow:hidden; }
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def _rgb(c):
    return f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"


def hero(title, subtitle, badges=()):
    b = "".join(f'<span class="badge">{html.escape(x)}</span>' for x in badges)
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1>'
                f'<p>{html.escape(subtitle)}</p>{b}</div>', unsafe_allow_html=True)


def _tip(text):
    """Biểu tượng ⓘ, rê chuột (hoặc chạm giữ) để xem giải thích."""
    return f'<span class="tip" title="{html.escape(text, quote=True)}">ⓘ</span>' if text else ""


def scoreboard(p0, p1, c0, c1, label="Kiểm soát bóng", names=("Team 0", "Team 1"), help=""):
    n0, n1 = (html.escape(n) for n in names)
    st.markdown(f"""
<div class="score">
  <div class="score-top">
    <span><span class="dot" style="background:{_rgb(c0)}"></span>{n0} · {p0:.1f}%</span>
    <span class="score-mid">{html.escape(label)}{_tip(help)}</span>
    <span>{p1:.1f}% · {n1}<span class="dot" style="background:{_rgb(c1)};margin:0 0 0 8px"></span></span>
  </div>
  <div class="bar"><div style="width:{p0}%;background:{_rgb(c0)}"></div>
       <div style="width:{p1}%;background:{_rgb(c1)}"></div></div>
</div>""", unsafe_allow_html=True)


def stat_card(label, value, sub="", icon="", help=""):
    st.markdown(f'<div class="card"><div class="lbl">{icon} {html.escape(label)}{_tip(help)}</div>'
                f'<div class="val">{html.escape(str(value))}</div>'
                f'<div class="sub">{html.escape(sub)}</div></div>', unsafe_allow_html=True)


def section(title, icon=""):
    st.markdown(f'<div class="sec">{icon} {html.escape(title)}</div>', unsafe_allow_html=True)
