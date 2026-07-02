"""
Padel Analytics Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import subprocess
import threading
import queue
import os
import json
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Padel Analytics Pro",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .stat-card .value { font-family: 'Orbitron', sans-serif; }

    /* Ultra Dark Background */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #151515 0%, #000000 100%);
        color: #f0f0f0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 10, 0.95);
        border-right: 2px solid #ffea00;
    }

    /* Premium Stat Cards */
    .stat-card {
        background: linear-gradient(145deg, #1e1e1e, #121212);
        border-left: 4px solid #ffea00;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: left;
    }
    .stat-card:hover { 
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(255, 234, 0, 0.1);
    }
    .stat-card .value { 
        font-size: 2.2rem; 
        font-weight: 700; 
        color: #ffea00; 
        text-shadow: 0 0 10px rgba(255, 234, 0, 0.3);
    }
    .stat-card .label { 
        font-size: 0.8rem; 
        color: #888; 
        margin-bottom: 4px; 
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Terminal Log Box */
    .log-box {
        background: #050505;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 20px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.85rem;
        color: #ffea00;
        height: 400px;
        overflow-y: auto;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
    }

    /* Neon Yellow Buttons */
    .stButton > button {
        background: #ffea00 !important;
        color: #000 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 12px 24px !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        width: 100% !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 15px rgba(255, 234, 0, 0.2) !important;
    }
    .stButton > button:hover { 
        background: #fffb00 !important;
        box-shadow: 0 0 25px rgba(255, 234, 0, 0.4) !important;
        transform: scale(1.02);
    }
    .stButton > button:active { transform: scale(0.98); }

    /* Custom Input Styling */
    .stTextInput > div > div > input {
        background: #111 !important;
        border: 1px solid #333 !important;
        border-radius: 4px !important;
        color: #fff !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ffea00 !important;
        box-shadow: 0 0 10px rgba(255, 234, 0, 0.2) !important;
    }

    /* Tabs Neon Accent */
    .stTabs [data-baseweb="tab-list"] { background: transparent; }
    .stTabs [data-baseweb="tab"] {
        color: #888;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #ffea00 !important;
        border-bottom-color: #ffea00 !important;
    }

    /* Titles */
    h1 { letter-spacing: -1px; margin-bottom: 2rem !important; }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "log_lines": [],
    "running": False,
    "done": False,
    "process": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ────────────────────────────────────────────────────────────────────
def stream_output(proc, log_queue):
    """Read stdout+stderr from subprocess into a queue (runs in a thread)."""
    for line in iter(proc.stdout.readline, ""):
        log_queue.put(line)
    proc.stdout.close()
    proc.wait()
    log_queue.put(None)  # sentinel


def run_pipeline(video_path: str):
    """Patch config and launch main.py as a subprocess."""
    # Patch config.py INPUT_VIDEO_PATH on-the-fly via environment variable
    env = os.environ.copy()
    env["PADEL_INPUT_VIDEO"] = video_path

    # We inject the video path by writing a thin config override
    override = f"""
import os as _os
INPUT_VIDEO_PATH = _os.environ.get("PADEL_INPUT_VIDEO", "{video_path}")
"""
    with open("_dashboard_config_override.py", "w") as f:
        f.write(override)

    proc = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return proc


def load_analytics():
    path = "data.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎾 Padel Analytics")
    st.markdown("---")
    st.markdown("### 📂 Input Video")

    # File path text input
    video_path = st.text_input(
        "Video path",
        value="./examples/videos/rally.mp4",
        help="Relative or absolute path to your padel match video",
        label_visibility="collapsed",
        placeholder="./examples/videos/rally.mp4",
    )

    # OR upload a file
    uploaded = st.file_uploader("Or upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        save_dir = "./input"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        video_path = save_path
        st.success(f"Saved: {save_path}")

    st.markdown("---")

    # ── Court Keypoints ───────────────────────────────────────────────────
    st.markdown("### 🎯 Court Keypoints")

    kp_path = "./cache/fixed_keypoints_detection.json"
    kp_exists = os.path.exists(kp_path)

    if kp_exists:
        try:
            import json as _json
            kp_data = _json.load(open(kp_path))
            st.success(f"✅ {len(kp_data)} keypoints cached")
        except Exception:
            st.warning("⚠️ Keypoints file unreadable")
    else:
        st.warning("⚠️ No keypoints cached")

    reset_kp_btn = st.button("🔄 Reset Keypoints")
    if reset_kp_btn:
        import re as _re
        if os.path.exists(kp_path):
            os.remove(kp_path)
        with open("config.py", "r") as f:
            cfg = f.read()
        cfg = _re.sub(
            r'^FIXED_COURT_KEYPOINTS_LOAD_PATH\s*=.*$',
            'FIXED_COURT_KEYPOINTS_LOAD_PATH = None',
            cfg, flags=_re.MULTILINE,
        )
        cfg = _re.sub(
            r'^FIXED_COURT_KEYPOINTS_SAVE_PATH\s*=.*$',
            f'FIXED_COURT_KEYPOINTS_SAVE_PATH = "{kp_path}"',
            cfg, flags=_re.MULTILINE,
        )
        with open("config.py", "w") as f:
            f.write(cfg)
        st.warning(
            "Keypoints cleared! Run `python main.py` in your terminal. "
            "A window will pop up — click the 12 court points in order, then press any key."
        )
        st.rerun()

    with st.expander("📐 Keypoints order"):
        st.code(
            "k11--------------------k12\n"
            "|                       |\n"
            "k8-----------k9--------k10\n"
            "|            |          |\n"
            "k6----------------------k7\n"
            "|            |          |\n"
            "k3-----------k4---------k5\n"
            "|                       |\n"
            "k1----------------------k2\n\n"
            "Click: k1→k2→k3→k4→k5→k6→k7→k8→k9→k10→k11→k12",
            language="text",
        )

    st.markdown("---")
    # Config knobs
    st.markdown("### ⚙️ Settings")
    batch_size = st.slider("Batch size", 1, 16, 8)
    max_frames = st.number_input("Max frames (0 = all)", min_value=0, value=0)
    collect_data = st.toggle("Collect analytics data", value=True)

    st.markdown("---")
    run_btn = st.button("▶  Run Analysis", disabled=st.session_state.running)
    if st.session_state.running:
        stop_btn = st.button("⏹  Stop", type="secondary")
    else:
        stop_btn = False

    st.markdown("---")
    st.markdown(
        "<span style='color:#666;font-size:0.75rem'>Padel Analytics v1.0 · "
        "amonras/padel_analytics</span>",
        unsafe_allow_html=True,
    )


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("# 🎾 Padel Analytics Dashboard")
st.markdown("Select a video in the sidebar and click **Run Analysis** to start.")

tab_live, tab_video, tab_analytics = st.tabs(
    ["📋 Live Terminal", "🎬 Output Video", "📊 Analytics"]
)

# ── Handle Run ──────────────────────────────────────────────────────────────────
if run_btn and not st.session_state.running:
    if not video_path:
        st.sidebar.error("Please provide a video path.")
    else:
        # Update config file with current settings
        import re

        with open("config.py", "r") as f:
            cfg = f.read()

        def _set(cfg, key, val):
            pattern = rf'^{key}\s*=.*$'
            replacement = f'{key} = {val}'
            # Use a lambda so backslashes in 'replacement' are never
            # interpreted as regex escape sequences (critical for Windows paths)
            return re.sub(pattern, lambda m: replacement, cfg, flags=re.MULTILINE)

        cfg = _set(cfg, "INPUT_VIDEO_PATH", f'"{video_path.replace(chr(92), "/")}"')
        cfg = _set(cfg, "PLAYERS_TRACKER_BATCH_SIZE", batch_size)
        cfg = _set(cfg, "PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE", batch_size)
        cfg = _set(cfg, "BALL_TRACKER_BATCH_SIZE", batch_size)
        cfg = _set(cfg, "KEYPOINTS_TRACKER_BATCH_SIZE", batch_size)
        cfg = _set(cfg, "MAX_FRAMES", "None" if max_frames == 0 else max_frames)
        cfg = _set(cfg, "COLLECT_DATA", str(collect_data))

        with open("config.py", "w") as f:
            f.write(cfg)

        st.session_state.log_lines = []
        st.session_state.running = True
        st.session_state.done = False

        proc = subprocess.Popen(
            [sys.executable, "-u", "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        st.session_state.process = proc
        st.rerun()

if stop_btn and st.session_state.process:
    st.session_state.process.terminate()
    st.session_state.running = False
    st.session_state.done = False
    st.session_state.log_lines.append("\n⚠️  Analysis stopped by user.")
    st.rerun()

# ── Poll subprocess output ─────────────────────────────────────────────────────
if st.session_state.running and st.session_state.process:
    proc = st.session_state.process
    # Read all available lines without blocking
    import select
    import time

    # Drain any available output
    new_lines = []
    try:
        while True:
            line = proc.stdout.readline()
            if line:
                new_lines.append(line.rstrip())
            else:
                break
    except Exception:
        pass

    st.session_state.log_lines.extend(new_lines)

    # Check if finished
    if proc.poll() is not None:
        # Drain remaining output
        for line in proc.stdout:
            st.session_state.log_lines.append(line.rstrip())
        st.session_state.running = False
        st.session_state.done = True
        st.session_state.process = None
        st.rerun()
    else:
        # Still running — rerun to poll again
        time.sleep(0.5)
        st.rerun()


# ── TAB 1: Live Terminal ───────────────────────────────────────────────────────
with tab_live:
    if st.session_state.running:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px'>"
            "<span style='width:10px;height:10px;background:#ffea00;border-radius:50%;"
            "display:inline-block;animation:pulse 1s infinite'></span>"
            "<b style='color:#ffea00'>Running...</b></div>"
            "<style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}</style>",
            unsafe_allow_html=True,
        )
    elif st.session_state.done:
        st.success("✅ Analysis complete!")
    else:
        st.info("Run the analysis to see live output here.")

    log_text = "\n".join(st.session_state.log_lines) if st.session_state.log_lines else "Waiting to start..."
    st.markdown(
        f'<div class="log-box">{log_text}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.log_lines:
        log_str = "\n".join(st.session_state.log_lines)
        st.download_button("⬇ Download Log", log_str, file_name="padel_run.log", mime="text/plain")


# ── TAB 2: Output Video ────────────────────────────────────────────────────────
with tab_video:
    output_path = "results.mp4"
    if os.path.exists(output_path):
        st.markdown("### 🎬 Inference Result")
        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("⬇ Download Results Video", f, file_name="results.mp4", mime="video/mp4")
    else:
        st.info("The output video will appear here after the analysis is complete.")
        st.markdown(
            "<div style='text-align:center;padding:60px;color:#555'>"
            "<span style='font-size:4rem'>🎬</span>"
            "<p style='margin-top:12px'>No results yet</p></div>",
            unsafe_allow_html=True,
        )


# ── TAB 3: Analytics ──────────────────────────────────────────────────────────
with tab_analytics:
    df = load_analytics()
    if df is not None:
        st.markdown("### 📊 Player Statistics")

        # Load summary statistics (real shoots)
        summary = {}
        if os.path.exists("summary.json"):
            with open("summary.json", "r") as f:
                summary = json.load(f)
        
        real_shoots = summary.get("player_shoots", {})

        # Summary cards
        players_data = []
        for pid in range(1, 5):
            vcol = f"player{pid}_Vnorm4"
            dcol = f"player{pid}_distance"
            if vcol in df.columns and dcol in df.columns:
                
                # Use real shoots from summary, fallback to 0
                shoots = real_shoots.get(str(pid), 0)
                score = shoots # Simplified score
                
                players_data.append({
                    "Player": f"Player {pid}",
                    "Total Distance (m)": round(df[dcol].sum(), 1),
                    "Avg Speed (km/h)": round(df[vcol].abs().mean() * 3.6, 1),
                    "Max Speed (km/h)": round(df[vcol].abs().max() * 3.6, 1),
                    "Shoots": shoots,
                    "Score": score,
                })

        if players_data:
            # Determine best player dynamically based on score
            best_player_name = max(players_data, key=lambda x: x["Score"])["Player"]
            
            cols = st.columns(len(players_data))
            for i, p in enumerate(players_data):
                is_best = p["Player"] == best_player_name
                best_badge = "<div style='color:#ffea00; font-weight:800; font-size:0.9rem; margin-top:8px;'>🏆 BEST PLAYER</div>" if is_best else ""
                card_style = "border-left: 4px solid #ffea00; box-shadow: 0 0 20px rgba(255,234,0,0.3);" if is_best else ""
                
                with cols[i]:
                    st.markdown(
                        f"<div class='stat-card' style='{card_style}'>"
                        f"<div class='label'>{p['Player']}</div>"
                        f"<div class='value'>{p['Score']} <span style='font-size:1rem'>pts</span></div>"
                        f"<div class='label'>{p['Shoots']} Shoots</div>"
                        f"<hr style='border-color:rgba(255,255,255,0.1);margin:10px 0'>"
                        f"<div style='font-size:0.85rem;color:#ccc'>Max Spd: {p['Max Speed (km/h)']} km/h</div>"
                        f"<div style='font-size:0.85rem;color:#ccc'>Dist: {p['Total Distance (m)']} m</div>"
                        f"{best_badge}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # Stats table
            st.markdown("#### Full Stats Table")
            st.dataframe(
                pd.DataFrame(players_data).set_index("Player"),
                use_container_width=True,
            )

            st.markdown("---")

            # Velocity chart
            st.markdown("#### Player Speed Over Time")
            fig = go.Figure()
            # Gradient of yellows/golds for players
            colors      = ["#ffea00", "#ffd700", "#ffc107", "#ffb300"]
            fill_colors = [
                "rgba(255, 234, 0, 0.10)",
                "rgba(255, 215, 0, 0.10)",
                "rgba(255, 193, 7, 0.10)",
                "rgba(255, 179, 0, 0.10)",
            ]

            for i, pid in enumerate(range(1, 5)):
                vcol = f"player{pid}_Vnorm4"
                if vcol in df.columns and "time" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df["time"],
                        y=df[vcol].abs() * 3.6,
                        name=f"Player {pid}",
                        mode="lines",
                        line=dict(color=colors[i], width=3),
                        fill="tozeroy",
                        fillcolor=fill_colors[i],
                    ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Time (s)",
                yaxis_title="Speed (km/h)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=380,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(gridcolor="#222", zerolinecolor="#333"),
                yaxis=dict(gridcolor="#222", zerolinecolor="#333"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🗺️ Tactical Heatmaps")
            
            # Unified Court Heatmap
            h_fig = go.Figure()
            
            # Draw the court (blue background)
            h_fig.add_shape(type="rect", x0=-5, y0=-10, x1=5, y1=10, fillcolor="#1a4ca0", line=dict(color="white", width=4))
            # Draw court lines
            h_fig.add_shape(type="line", x0=-5, y0=0, x1=5, y1=0, line=dict(color="white", width=4, dash="dash")) # Net
            h_fig.add_shape(type="line", x0=-5, y0=3, x1=5, y1=3, line=dict(color="white", width=2)) # Service line top
            h_fig.add_shape(type="line", x0=-5, y0=-3, x1=5, y1=-3, line=dict(color="white", width=2)) # Service line bottom
            h_fig.add_shape(type="line", x0=0, y0=-3, x1=0, y1=3, line=dict(color="white", width=2)) # Center line
            
            # Custom transparent-to-solid colorscales
            colorscales = [
                [[0, 'rgba(255,0,0,0)'], [0.4, 'rgba(255,50,0,0.5)'], [1, 'rgba(255,255,0,0.9)']],   # Fire (Red->Yellow)
                [[0, 'rgba(0,150,255,0)'], [0.4, 'rgba(0,150,255,0.5)'], [1, 'rgba(0,255,255,0.9)']], # Ice (Blue->Cyan)
                [[0, 'rgba(255,0,255,0)'], [0.4, 'rgba(200,0,200,0.5)'], [1, 'rgba(100,0,255,0.9)']], # Purple/Magenta
                [[0, 'rgba(0,255,0,0)'], [0.4, 'rgba(50,200,50,0.5)'], [1, 'rgba(150,255,0,0.9)']],   # Green/Lime
            ]
            
            for i, pid in enumerate(range(1, 5)):
                xcol = f"player{pid}_x"
                ycol = f"player{pid}_y"
                if xcol in df.columns and ycol in df.columns:
                    player_df = df[[xcol, ycol]].dropna()
                    if not player_df.empty:
                        h_fig.add_trace(go.Histogram2dContour(
                            x=player_df[xcol],
                            y=player_df[ycol],
                            colorscale=colorscales[i % 4],
                            showscale=False,
                            ncontours=12,
                            nbinsx=40, nbinsy=80,
                            contours=dict(coloring='fill', showlines=True),
                            line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
                            name=f"Player {pid}",
                            hoverinfo='none',
                        ))
            
            h_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=700,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(range=[-6, 6], showgrid=False, zeroline=False, visible=False),
                yaxis=dict(range=[-11, 11], showgrid=False, zeroline=False, visible=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Use columns to center the unified court
            _, col_center, _ = st.columns([1, 2, 1])
            with col_center:
                st.plotly_chart(h_fig, use_container_width=True)

            # Raw CSV
            with st.expander("🔍 View raw data"):
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button("⬇ Download CSV", csv, file_name="padel_analytics.csv", mime="text/csv")
        else:
            st.warning("No player data columns found in data.csv.")
    else:
        st.info("Analytics data will appear here after the analysis completes with **Collect analytics data** enabled.")
        st.markdown(
            "<div style='text-align:center;padding:60px;color:#555'>"
            "<span style='font-size:4rem'>📊</span>"
            "<p style='margin-top:12px'>No data yet</p></div>",
            unsafe_allow_html=True,
        )
