import streamlit as st
import time
import random
import numpy as np

st.set_page_config(page_title="Reaction Time Test", layout="centered")

# Real data-based constants from your PVT analysis (unchanged)
STREAMLIT_DELAY_CORRECTION = 200  # ms - subtracted from user's measured time only
NORMAL_MEAN_RT = 319.2  # ms
NORMAL_STD_RT = 29.7  # ms
SLEEP_DEPRIVED_MEAN_RT = 327.4  # ms
SLEEP_DEPRIVED_STD_RT = 85.0  # ms

# Calculate delays for sleep deprived mode
SLEEP_DEPRIVED_BASE_DELAY = (SLEEP_DEPRIVED_MEAN_RT - NORMAL_MEAN_RT) / 1000  # seconds
SLEEP_DEPRIVED_EXTRA_VARIABILITY = (SLEEP_DEPRIVED_STD_RT - NORMAL_STD_RT) / 1000  # seconds

# Enhanced CSS that works in both light and dark modes
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --text-primary: #1a1a1a;
    --text-secondary: #6c757d;
    --bg-card: rgba(255, 255, 255, 0.9);
    --bg-info: #f8f9fa;
    --bg-data: #fff3cd;
    --border-color: #e9ecef;
    --accent-color: #007bff;
    --data-text: #856404;
}

@media (prefers-color-scheme: dark) {
    :root {
        --text-primary: #e0e0e0;
        --text-secondary: #b0b0b0;
        --bg-card: rgba(40, 40, 40, 0.9);
        --bg-info: rgba(30, 30, 30, 0.8);
        --bg-data: rgba(255, 193, 7, 0.15);
        --border-color: rgba(255, 255, 255, 0.2);
        --accent-color: #4fc3f7;
        --data-text: #fff3cd;
    }
}

.main > div {
    max-width: 600px;
    margin: 0 auto;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.reaction-container {
    background: #1a1a1a;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    margin: 20px 0;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    position: relative;
    overflow: hidden;
    transition: all 0.2s ease;
}

.reaction-container.waiting {
    background: #dc3545;
}

.reaction-container.ready {
    background: #28a745;
}

.reaction-container.clicked {
    background: #007bff;
}

.reaction-container.error {
    background: #dc3545;
}

.reaction-text {
    color: white;
    font-size: 2.5rem;
    font-weight: 600;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.reaction-subtext {
    color: rgba(255,255,255,0.8);
    font-size: 1.1rem;
    margin-top: 10px;
    font-weight: 400;
}

.stats-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin: 20px 0;
}

.stat-card {
    background: var(--bg-card);
    border: 2px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
    backdrop-filter: blur(10px);
}

.stat-card:hover {
    transform: translateY(-2px);
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-color);
    margin-bottom: 5px;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

.stat-label {
    color: var(--text-secondary);
    font-size: 0.9rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.sleep-deprived {
    filter: blur(0.8px) brightness(0.9);
}

.sleep-deprived .reaction-container {
    background: #495057;
}

.sleep-deprived .reaction-container.waiting {
    background: #a94442;
}

.sleep-deprived .reaction-container.ready {
    background: #5cb85c;
}

.instructions {
    background: var(--bg-info);
    border-left: 4px solid var(--accent-color);
    padding: 15px 20px;
    margin: 20px 0;
    border-radius: 0 8px 8px 0;
    color: var(--text-primary);
    backdrop-filter: blur(10px);
}

.instructions strong {
    color: var(--text-primary);
    font-weight: 600;
}

.mode-indicator {
    position: absolute;
    top: 10px;
    right: 15px;
    background: rgba(0,0,0,0.3);
    color: white;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.percentile-info {
    background: var(--bg-info);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 15px;
    margin: 15px 0;
    text-align: center;
    color: var(--text-primary);
    backdrop-filter: blur(10px);
}

.percentile-info strong {
    color: var(--accent-color);
    font-weight: 600;
}

.percentile-info small {
    color: var(--text-secondary) !important;
}

.data-info {
    background: var(--bg-data);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
    margin: 15px 0;
    font-size: 0.9rem;
    color: var(--data-text);
    backdrop-filter: blur(10px);
}

.data-info strong {
    color: var(--data-text);
    font-weight: 600;
}

.correction-info {
    background: var(--bg-info);
    border: 1px solid var(--accent-color);
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-align: center;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


def get_realistic_sleep_deprived_delay():
    """Generate realistic delay based on your actual PVT data"""
    # Base delay from mean difference
    base_delay = SLEEP_DEPRIVED_BASE_DELAY

    # Additional variability - use exponential distribution to simulate
    # the increased inconsistency seen in sleep deprivation
    extra_variability = np.random.exponential(SLEEP_DEPRIVED_EXTRA_VARIABILITY * 0.5)

    # Sometimes sleep deprivation can actually make you faster (lapses work both ways)
    # But usually makes you slower
    if random.random() < 0.15:  # 15% chance of being faster due to compensation
        variability_modifier = -extra_variability * 0.3
    else:
        variability_modifier = extra_variability

    total_delay = base_delay + variability_modifier

    # Ensure delay is not negative
    return max(0, total_delay)


def get_percentile(reaction_time_ms, is_sleep_deprived):
    """Calculate percentile based on your actual data distribution (using original values)"""
    if is_sleep_deprived:
        # Based on your sleep deprived data: mean=327.4, std=85.0
        z_score = (reaction_time_ms - SLEEP_DEPRIVED_MEAN_RT) / SLEEP_DEPRIVED_STD_RT
    else:
        # Based on your normal data: mean=319.2, std=29.7
        z_score = (reaction_time_ms - NORMAL_MEAN_RT) / NORMAL_STD_RT

    # Convert z-score to percentile (approximate)
    if z_score <= -2:
        return 98
    elif z_score <= -1:
        return 84
    elif z_score <= 0:
        return 50
    elif z_score <= 1:
        return 16
    elif z_score <= 2:
        return 2
    else:
        return 1


# Sidebar
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Choose Mode", ["Normal", "Sleep-Deprived"])
is_sleep = mode == "Sleep-Deprived"

# Show data statistics in sidebar (original values)
st.sidebar.subheader("📊 Real PVT Data")
st.sidebar.metric("Normal Mean RT", f"{NORMAL_MEAN_RT:.1f} ms")
st.sidebar.metric("Normal Std Dev", f"{NORMAL_STD_RT:.1f} ms")
st.sidebar.metric("Sleep Deprived Mean", f"{SLEEP_DEPRIVED_MEAN_RT:.1f} ms")
st.sidebar.metric("Sleep Deprived Std Dev", f"{SLEEP_DEPRIVED_STD_RT:.1f} ms")
st.sidebar.metric("Mean Difference", f"{SLEEP_DEPRIVED_MEAN_RT - NORMAL_MEAN_RT:.1f} ms")

# Add correction info to sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Correction Applied")
st.sidebar.info(f"Scores adjusted by -{STREAMLIT_DELAY_CORRECTION}ms to match other platforms like Human Benchmark")

# Main content wrapper
if is_sleep:
    st.markdown('<div class="sleep-deprived">', unsafe_allow_html=True)

# Title
st.title("🧠 Reaction Time Test")
if is_sleep:
    st.caption("🥱 Sleep-Deprived Mode - Based on real EEG study data")

# Show data source info
st.markdown(f"""
<div class="data-info">
<strong>📈 Data-Driven Simulation:</strong> This test uses reaction time patterns from real participants in 
{'sleep-deprived' if is_sleep else 'normal'} conditions. 
{f'Sleep deprivation increases variability by {((SLEEP_DEPRIVED_STD_RT / NORMAL_STD_RT - 1) * 100):.0f}%!' if is_sleep else ''}
</div>
""", unsafe_allow_html=True)

# Add correction notice
st.markdown(f"""
<div class="correction-info">
⚡ Scores corrected by -{STREAMLIT_DELAY_CORRECTION}ms for accuracy vs other platforms
</div>
""", unsafe_allow_html=True)

# Instructions
st.markdown("""
<div class="instructions">
<strong>How to play:</strong><br>
<span style="color: #e0e0e0;">When the red box turns green, click as quickly as you can. Click too soon and you'll have to start again.</span>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if 'state' not in st.session_state:
    st.session_state.state = "idle"
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0
if 'wait_until' not in st.session_state:
    st.session_state.wait_until = 0
if 'reaction_time' not in st.session_state:
    st.session_state.reaction_time = None
if 'error' not in st.session_state:
    st.session_state.error = False
if 'best_time' not in st.session_state:
    st.session_state.best_time = None
if 'attempt_count' not in st.session_state:
    st.session_state.attempt_count = 0
if 'all_times' not in st.session_state:
    st.session_state.all_times = []
if 'sleep_delay' not in st.session_state:
    st.session_state.sleep_delay = 0


def reset_state():
    st.session_state.state = "idle"
    st.session_state.start_time = 0
    st.session_state.wait_until = 0
    st.session_state.reaction_time = None
    st.session_state.error = False
    st.session_state.sleep_delay = 0


def start_test():
    st.session_state.state = "waiting"
    st.session_state.reaction_time = None
    st.session_state.wait_until = time.time() + random.uniform(2, 5)
    st.session_state.error = False
    # Pre-calculate sleep deprived delay
    if is_sleep:
        st.session_state.sleep_delay = get_realistic_sleep_deprived_delay()
    else:
        st.session_state.sleep_delay = 0
    st.rerun()


# Main game logic
if st.session_state.state == "idle":
    st.markdown(f"""
    <div class="reaction-container">
        <div class="reaction-text">Click to Start</div>
        <div class="reaction-subtext">Test your reaction time</div>
        {f'<div class="mode-indicator">{mode}</div>' if is_sleep else ''}
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Test", key="start", use_container_width=True, type="primary"):
        start_test()

elif st.session_state.state == "waiting":
    st.markdown(f"""
    <div class="reaction-container waiting">
        <div class="reaction-text">Wait for Green</div>
        <div class="reaction-subtext">Don't click yet...</div>
        {f'<div class="mode-indicator">{mode}</div>' if is_sleep else ''}
    </div>
    """, unsafe_allow_html=True)

    # Hidden button to detect early clicks
    if st.button("WAIT", key="wait", use_container_width=True):
        st.session_state.state = "error"
        st.session_state.error = True
        st.rerun()

    # Check if it's time to turn green
    if time.time() >= st.session_state.wait_until:
        st.session_state.state = "ready"
        st.session_state.start_time = time.time()
        st.rerun()
    else:
        time.sleep(0.1)
        st.rerun()

elif st.session_state.state == "ready":
    st.markdown(f"""
    <div class="reaction-container ready">
        <div class="reaction-text">CLICK!</div>
        <div class="reaction-subtext">Click as fast as you can</div>
        {f'<div class="mode-indicator">{mode}</div>' if is_sleep else ''}
    </div>
    """, unsafe_allow_html=True)

    if st.button("CLICK NOW!", key="click", use_container_width=True):
        # Apply realistic delay for sleep deprived mode
        if is_sleep:
            time.sleep(st.session_state.sleep_delay)

        # Calculate reaction time and apply correction
        raw_reaction_time = time.time() - st.session_state.start_time
        st.session_state.reaction_time = max(0.001, raw_reaction_time - (STREAMLIT_DELAY_CORRECTION / 1000))

        # Update statistics
        st.session_state.attempt_count += 1
        st.session_state.all_times.append(st.session_state.reaction_time * 1000)

        if (st.session_state.best_time is None or
                st.session_state.reaction_time < st.session_state.best_time):
            st.session_state.best_time = st.session_state.reaction_time

        st.session_state.state = "result"
        st.rerun()

elif st.session_state.state == "error":
    st.markdown(f"""
    <div class="reaction-container error">
        <div class="reaction-text">Too Soon!</div>
        <div class="reaction-subtext">You clicked before it turned green</div>
        {f'<div class="mode-indicator">{mode}</div>' if is_sleep else ''}
    </div>
    """, unsafe_allow_html=True)

    if st.button("Try Again", key="retry_error", use_container_width=True):
        reset_state()
        st.rerun()

elif st.session_state.state == "result":
    rt_ms = int(st.session_state.reaction_time * 1000)
    percentile = get_percentile(rt_ms, is_sleep)

    st.markdown(f"""
    <div class="reaction-container clicked">
        <div class="reaction-text">{rt_ms} ms</div>
        <div class="reaction-subtext">Your reaction time</div>
        {f'<div class="mode-indicator">{mode}</div>' if is_sleep else ''}
    </div>
    """, unsafe_allow_html=True)

    # Performance feedback
    comparison_text = "normal participants" if not is_sleep else "sleep-deprived participants"
    st.markdown(f"""
    <div class="percentile-info">
        <strong>You're faster than {percentile}% of {comparison_text}</strong><br>
        <small>Based on real PVT study data from {NORMAL_MEAN_RT:.0f}±{NORMAL_STD_RT:.0f}ms (normal) and {SLEEP_DEPRIVED_MEAN_RT:.0f}±{SLEEP_DEPRIVED_STD_RT:.0f}ms (sleep-deprived)</small>
    </div>
    """, unsafe_allow_html=True)

    # Statistics
    if st.session_state.attempt_count > 0:
        avg_time = np.mean(st.session_state.all_times)
        best_time = min(st.session_state.all_times)

        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-value">{int(best_time)} ms</div>
                <div class="stat-label">Best Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{int(avg_time)} ms</div>
                <div class="stat-label">Average</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Try Again", key="retry_result", use_container_width=True, type="primary"):
        reset_state()
        st.rerun()

# Show attempt counter and session stats
if st.session_state.attempt_count > 0:
    st.sidebar.subheader("📈 Your Session")
    st.sidebar.metric("Attempts", st.session_state.attempt_count)
    if st.session_state.all_times:
        best_ms = min(st.session_state.all_times)
        avg_ms = np.mean(st.session_state.all_times)
        st.sidebar.metric("Best Time", f"{best_ms:.0f} ms")
        st.sidebar.metric("Average", f"{avg_ms:.0f} ms")
        st.sidebar.metric("Consistency", f"{np.std(st.session_state.all_times):.0f} ms std")

if is_sleep:
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(
    f"💡 **Data Source:** Reaction time delays based on real PVT measurements from EEG study participants. Scores corrected by -{STREAMLIT_DELAY_CORRECTION}ms to account for Streamlit/browser latency.")