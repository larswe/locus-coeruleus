import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Sidebar configuration
st.sidebar.header("Simulation Parameters")
n_trials = st.sidebar.slider("Number of trials", 10, 200, 50)
us_intensity = st.sidebar.slider("Shock intensity (US)", 0.0, 1.0, 0.5, 0.05) # min 0.0, max 1.0, default 0.8, step 0.05
ne_threshold = st.sidebar.slider("NE threshold for plasticity", 0.0, 0.75, 0.20, 0.01) 
learning_rate = st.sidebar.slider("Learning rate η", 0.0, 0.25, 0.05, 0.01)
W_us_la = st.sidebar.slider("US→LA weight", 0.0, 2.0, 1.0, 0.1)  # US to LA weight

# Added in v0.0.1: beta-adrenoreceptor activity in LA
st.sidebar.markdown("### NE_LA = m * (LC activity) + c")
beta_ar_modulation = st.sidebar.slider("LA β-AR modulation factor (m)", 0.0, 2.0, 1.0, 0.1)
baseline_ne_la = st.sidebar.slider("Baseline NE in LA (c)", 0.0, 1.0, 0.10, 0.025)

if st.sidebar.checkbox("Apply propranolol (m = 0.5, c = 0.05)"):
    beta_ar_modulation = 0.5
    baseline_ne_la = 0.05
if st.sidebar.checkbox("Apply isoproterenol (m = 2.0, c = 0.2)"):
    beta_ar_modulation = 2.0
    baseline_ne_la = 0.2


# Static parameters 
n_cs_units = 10
n_la_units = 10
active_cs_fraction = 0.5
mean_CS_LA_starting_weight = 0.005

st.sidebar.markdown("### Fixed Parameters")
st.sidebar.markdown(f"Number of CS units: `{n_cs_units}`")
st.sidebar.markdown(f"Number of LA units: `{n_la_units}`")
st.sidebar.markdown(f"Fraction of active CS units: `{active_cs_fraction}`")
st.sidebar.markdown(f"Mean initial CS→LA weight: `{mean_CS_LA_starting_weight}`")


# Initialise weights and variables
np.random.seed(0)
W_cs_la = np.random.normal(mean_CS_LA_starting_weight, 0.0025, size=(n_cs_units, n_la_units))
W_cs_la = np.maximum(W_cs_la, 0.0) 
W_la_cea = np.ones(n_la_units) / n_la_units

# Recording arrays
fear_output = []
ne_trace = []
cs_weight_trace = []
la_activation_trace = []

# Randomly choose a subset of CS units to respond to this CS
n_active_cs = int(active_cs_fraction * n_cs_units)
active_cs_indices = np.random.choice(n_cs_units, size=n_active_cs, replace=False)

# Simulation loop
for trial in range(n_trials):
    cs_input = np.zeros(n_cs_units)
    cs_input[active_cs_indices] = 1.0
    us = us_intensity if trial < 3 * n_trials // 4 and trial >= n_trials // 4 else 0.0  # US presented in the second quarter of trials

    # LA activation
    la_input = np.dot(cs_input, W_cs_la) + W_us_la * us
    la_input = la_input / (1.0 + np.abs(la_input))  # squash input
    la_activation = np.clip(la_input, 0, 1)  # cap at 1
    la_activation_trace.append(np.mean(la_activation))

    # CeA output as mean LA activation
    cea_output = np.mean(la_activation)
    fear_output.append(cea_output)

    # LC/NE activation = aversive prediction error
    lc_activity = max(us - cea_output, 0)  # phasic LC firing
    ne_la = beta_ar_modulation * lc_activity + baseline_ne_la  # NE concentration in LA
    ne_trace.append(ne_la)

    # Hebbian + NE-gated plasticity
    if ne_la > ne_threshold:
        for i in range(n_cs_units):
            for j in range(n_la_units):
                delta_w = learning_rate * cs_input[i] * la_activation[j]
                W_cs_la[i, j] += delta_w

    # Track mean weight
    cs_weight_trace.append(np.mean(W_cs_la))

# Plotting
fig, ax = plt.subplots(4, 1, figsize=(8, 13), sharex=True)

ax[0].plot(fear_output, label="Fear output (CeA)")
ax[0].set_ylabel("Fear (CeA)")
ax[0].legend()

ax[1].plot(ne_trace, label="NE (LC activity)", color="orange")
ax[1].axhline(ne_threshold, color="red", linestyle="--", label="NE threshold")
ax[1].set_ylabel("NE")
ax[1].legend()

ax[2].plot(cs_weight_trace, label="Mean CS→LA weight", color="green")
ax[2].set_ylabel("CS→LA weight")
ax[2].legend()

# Mean LA activation
ax[3].plot(la_activation_trace, label="Mean LA activation", color="purple")
ax[3].set_ylabel("LA activation")
ax[3].set_xlabel("Trial")
ax[3].legend()
ax[3].set_ylim(0, 1)
ax[3].set_xlim(0, n_trials - 1)
ax[3].set_xticks(np.arange(0, n_trials + 1, n_trials // 10))
ax[3].set_xticklabels(np.arange(0, n_trials + 1, n_trials // 10))
ax[3].grid(True)

st.pyplot(fig)
