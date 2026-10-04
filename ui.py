"""
ui.py - Thành phần giao diện dùng chung (CSS + HTML nhỏ) cho app.py.
Bảng màu: nền xanh sân đêm, nhấn xanh cỏ #22C55E, vàng bóng #FFD60A.
"""
import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;800&display=swap');
:root { --accent:#22C55E; --ball:#FFD60A; --card:#0F2418; --line:rgba(255,255,255,.08); --muted:#9DB8A6; }
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.2rem; max-width: 1280px; }

/* HERO: dải sân cỏ có vạch giữa sân */
.hero { position:relative; overflow:hidden; border-radius:20px; padding:28px 32px; margin-bottom:18px;
  background:
    radial-gradient(circle at 50% 50%, transparent 58px, rgba(255,255,255,.18) 59px, rgba(255,255,255,.18) 61px, transparent 62px),
    linear-gradient(90deg, transparent calc(50% - 1px), rgba(255,255,255,.18) calc(50% - 1px), rgba(255,255,255,.18) calc(50% + 1px), transparent calc(50% + 1px)),
    repeating-linear-gradient(90deg, #14532d 0 80px, #166534 80px 160px);
  box-shadow: 0 10px 30px rgba(0,0,0,.35); }
.hero h1 { font-family:'Bebas Neue', sans-serif; font-size:3rem; letter-spacing:2px; margin:0; color:#fff;
  text-shadow:0 3px 12px rgba(0,0,0,.5); }
.hero p { margin:.3rem 0 .8rem; color:#e6ffe9; max-width:720px; }
.badge { display:inline-block; padding:4px 12px; margin:0 6px 6px 0; border-radius:999px; font-size:.8rem;
  background:rgba(0,0,0,.35); border:1px solid rgba(255,255,255,.25); color:#fff; }

/* SCOREBOARD possession */
.score { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:16px 20px; margin-bottom:14px; }
.score-top { display:flex; justify-content:space-between; align-items:center; font-family:'Bebas Neue'; font-size:2.4rem; }
.score-mid { color:var(--muted); font-family:'Inter'; font-size:.8rem; letter-spacing:2px; text-transform:uppercase; }
.bar { display:flex; height:14px; border-radius:999px; overflow:hidden; margin-top:8px; box-shadow:inset 0 0 0 1px var(--line); }
.bar > div { transition: width 1.2s ease; }
.dot { display:inline-block; width:14px; height:14px; border-radius:50%; margin-right:8px; vertical-align:middle;
  border:2px solid rgba(255,255,255,.6); }

/* STAT CARDS */
.card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:16px 18px; height:100%;
  transition: transform .15s ease, border-color .15s ease; }
.card:hover { transform: translateY(-3px); border-color: var(--accent); }
.card .lbl { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:1px; }
.card .val { font-family:'Bebas Neue'; font-size:2.3rem; line-height:1.1; color:#fff; }
.card .sub { color:var(--muted); font-size:.8rem; }
.sec { font-family:'Bebas Neue'; font-size:1.6rem; letter-spacing:1px; margin:10px 0 4px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:var(--card); border-radius:10px 10px 0 0; padding:8px 16px; }
.stTabs [aria-selected="true"] { background:#14532d; }
@media (max-width: 640px) { .hero h1 { font-size:2.1rem; } .score-top { font-size:1.7rem; } }
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


def scoreboard(p0, p1, c0, c1, label="Kiểm soát bóng"):
    st.markdown(f"""
<div class="score">
  <div class="score-top">
    <span><span class="dot" style="background:{_rgb(c0)}"></span>TEAM 0 · {p0:.1f}%</span>
    <span class="score-mid">{html.escape(label)}</span>
    <span>{p1:.1f}% · TEAM 1<span class="dot" style="background:{_rgb(c1)};margin:0 0 0 8px"></span></span>
  </div>
  <div class="bar"><div style="width:{p0}%;background:{_rgb(c0)}"></div>
       <div style="width:{p1}%;background:{_rgb(c1)}"></div></div>
</div>""", unsafe_allow_html=True)


def stat_card(label, value, sub="", icon=""):
    st.markdown(f'<div class="card"><div class="lbl">{icon} {html.escape(label)}</div>'
                f'<div class="val">{html.escape(str(value))}</div>'
                f'<div class="sub">{html.escape(sub)}</div></div>', unsafe_allow_html=True)


def section(title, icon=""):
    st.markdown(f'<div class="sec">{icon} {html.escape(title)}</div>', unsafe_allow_html=True)
