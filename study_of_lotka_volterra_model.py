import numpy as np
np.seterr(all='ignore')
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.linalg import eig as scipy_eig
from scipy.signal import correlate, find_peaks, windows
from scipy.ndimage import gaussian_filter
from functools import lru_cache
import plotly.graph_objects as go
import pandas as pd
import unittest
from plotly.subplots import make_subplots

# ====================== core simulator ======================
EPSILON = 1e-9  # small value to avoid division by zero
DEFAULT_PARAMS = {'alpha': 0.5, 'beta': 0.01, 'delta': 0.001, 'gamma': 0.2}

# pre-defined ecological scenarios (key: scenario name, value: parameters)
SCENARIOS = {
    'Baseline': DEFAULT_PARAMS.copy(),
    'Krill Overfishing': {'alpha': 0.2, 'beta': 0.01, 'delta': 0.001, 'gamma': 0.2},
    'Whale Hunting': {'alpha': 0.5, 'beta': 0.01, 'delta': 0.001, 'gamma': 0.4},
    'Climate Change': {'alpha': 0.5, 'beta': 0.01, 'delta': 0.0003, 'gamma': 0.2},
    'Collapse Risk': {'alpha': 0.1, 'beta': 0.01, 'delta': 0.001, 'gamma': 0.5}
}

class LotkaVolterraSimulator:
    """Simulates predator-prey dynamics using the Lotka-Volterra model for educational purposes.

    This class models the ecological interactions between whales (predators) and krill (prey) using
    differential equations, providing numerical solutions and stability analysis to teach ecological dynamics.

    Attributes:
        params (dict): Model parameters (alpha, beta, delta, gamma).
        initial_conditions (np.array): Initial populations [prey, predator].
        t_span (list): Time range [start, end] for simulation.
        dt (float): Time step size for numerical methods.
    """
    def __init__(self):
        self.params = DEFAULT_PARAMS.copy()
        self.initial_conditions = np.array([200.0, 15.0])  # [prey, predator]
        self.t_span = [0, 100]  # default simulation duration
        self.dt = 0.1  # time step
        self._clear_cache()

    def _clear_cache(self):
        """Clear cached Jacobian calculations when parameters change."""
        self.jacobian.cache_clear()

    def update_params(self, params):
        """Update model parameters and reset cache.

        Args:
            params (dict): Dictionary of parameters to update (alpha, beta, delta, gamma).
        """
        self.params.update(params)
        self._clear_cache()

    def lotka_volterra(self, t, state):
        """Lotka-Volterra differential equations for predator-prey dynamics.

        Args:
            t (float): Current time (unused, for compatibility with solvers).
            state (np.array): Current populations [prey, predator].

        Returns:
            np.array: Derivatives [d(prey)/dt, d(predator)/dt].
        """
        x, y = np.maximum(0, state)  # ensure non-negative populations
        dxdt = self.params['alpha'] * x - self.params['beta'] * x * y
        dydt = self.params['delta'] * x * y - self.params['gamma'] * y
        return np.array([dxdt, dydt])

    def solve(self, method='euler'):
        """Solve the system using the specified numerical method.

        Args:
            method (str): Numerical method to use ('euler', 'rk4', or 'reference' for lsoda).

        Returns:
            tuple: (time array, solution array with shape [time, 2] for prey and predator).
        """
        if method == 'euler':
            return self._euler_solve()
        elif method == 'rk4':
            return self._rk4_solve()
        else:
            return self._reference_solve()

    def _euler_solve(self):
        """Solve using the forward euler method (1st-order accuracy).

        Returns:
            tuple: (time array, solution array).
        """
        t = np.arange(self.t_span[0], self.t_span[1] + self.dt, self.dt)
        y = np.zeros((len(t), 2))
        y[0] = self.initial_conditions
        for i in range(len(t)-1):
            y[i+1] = y[i] + self.dt * self.lotka_volterra(t[i], y[i])
        return t, y

    def _rk4_solve(self):
        """Solve using the runge-kutta 4th-order method (higher accuracy).

        Returns:
            tuple: (time array, solution array).
        """
        t = np.arange(self.t_span[0], self.t_span[1] + self.dt, self.dt)
        y = np.zeros((len(t), 2))
        y[0] = self.initial_conditions
        for i in range(len(t)-1):
            h = self.dt
            k1 = self.lotka_volterra(t[i], y[i])
            k2 = self.lotka_volterra(t[i] + h/2, y[i] + h/2 * k1)
            k3 = self.lotka_volterra(t[i] + h/2, y[i] + h/2 * k2)
            k4 = self.lotka_volterra(t[i] + h, y[i] + h * k3)
            y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        return t, y

    def _reference_solve(self):
        """Solve using scipy's lsoda method for high-precision reference.

        Returns:
            tuple: (time array, solution array).
        """
        sol = solve_ivp(
            self.lotka_volterra, self.t_span, self.initial_conditions,
            method='LSODA', rtol=1e-6, atol=1e-8, dense_output=True
        )
        t = np.linspace(self.t_span[0], self.t_span[1], 500)
        return t, sol.sol(t)

    @lru_cache(maxsize=10)
    def jacobian(self, x_val, y_val):
        """Compute the jacobian matrix at a given state for stability analysis.

        Args:
            x_val (float): Prey population.
            y_val (float): Predator population.

        Returns:
            np.array: 2x2 jacobian matrix.
        """
        alpha, beta = self.params['alpha'], self.params['beta']
        delta, gamma = self.params['delta'], self.params['gamma']
        return np.array([
            [alpha - beta * y_val, -beta * x_val],
            [delta * y_val, delta * x_val - gamma]
        ])

    def find_equilibria(self):
        """Find equilibrium points where dx/dt = dy/dt = 0.

        Returns:
            np.array: Array of equilibrium points [prey, predator].
        """
        extinction = np.array([0.0, 0.0])  # trivial equilibrium
        equilibria = [extinction]
        delta, beta = self.params['delta'], self.params['beta']
        gamma, alpha = self.params['gamma'], self.params['alpha']

        if abs(delta) > EPSILON and abs(beta) > EPSILON:
            try:
                x_eq = gamma / delta  # prey equilibrium
                y_eq = alpha / beta   # predator equilibrium
                if x_eq >= 0 and y_eq >= 0:
                    equilibria.append(np.array([x_eq, y_eq]))
            except ZeroDivisionError:
                pass
        return np.array(equilibria)

    def analyze_stability(self, equilibria):
        """Analyze the stability of equilibrium points using eigenvalues.

        Args:
            equilibria (np.array): Array of equilibrium points to analyze.

        Returns:
            list: List of dictionaries containing stability information for each equilibrium.
        """
        stability_info = []
        for eq in equilibria:
            J = self.jacobian(eq[0], eq[1])
            eigenvalues = scipy_eig(J)[0]
            real_parts = np.real(eigenvalues)
            imag_parts = np.imag(eigenvalues)

            if np.all(real_parts < -EPSILON):
                stability = "Stable Node" if np.all(np.abs(imag_parts) < EPSILON) else "Stable Spiral"
            elif np.all(real_parts > EPSILON):
                stability = "Unstable Node" if np.all(np.abs(imag_parts) < EPSILON) else "Unstable Spiral"
            elif np.any(real_parts > EPSILON) and np.any(real_parts < -EPSILON):
                stability = "Saddle Point"
            else:
                stability = "Neutrally Stable (Center)"

            stability_info.append({
                'point': eq, 'jacobian': J,
                'eigenvalues': eigenvalues, 'stability': stability
            })
        return stability_info

# ====================== visualization functions ======================
def create_population_plot(t_euler, sol_euler, t_rk4, sol_rk4, t_ref, sol_ref):
    """Create a plot of population dynamics over time for different numerical methods.

    Args:
        t_euler, t_rk4, t_ref (np.array): Time points for Euler, RK4, and reference methods.
        sol_euler, sol_rk4, sol_ref (np.array): Solutions for each method [prey, predator].

    Returns:
        go.Figure: Plotly figure showing prey and predator population trajectories.
    """
    fig = go.Figure()
    colors = {
        'euler_prey': '#377eb8', 'rk4_prey': '#4daf4a', 'reference_prey': '#999999',
        'euler_pred': '#ff7f00', 'rk4_pred': '#e41a1c', 'reference_pred': '#984ea3'
    }  # colorblind-friendly palette

    fig.add_trace(go.Scatter(x=t_euler, y=sol_euler[:,0], name='Prey (Euler)', line=dict(color=colors['euler_prey'], width=2),
                             hovertemplate="Time: %{x:.2f} years<br>Prey: %{y:.1f}"))
    fig.add_trace(go.Scatter(x=t_rk4, y=sol_rk4[:,0], name='Prey (RK4)', line=dict(color=colors['rk4_prey'], width=2, dash='dot'),
                             hovertemplate="Time: %{x:.2f} years<br>Prey: %{y:.1f}"))
    fig.add_trace(go.Scatter(x=t_ref, y=sol_ref[0], name='Prey (Reference)', line=dict(color=colors['reference_prey'], width=2, dash='dash'),
                             hovertemplate="Time: %{x:.2f} years<br>Prey: %{y:.1f}"))
    fig.add_trace(go.Scatter(x=t_euler, y=sol_euler[:,1], name='Predator (Euler)', line=dict(color=colors['euler_pred'], width=2),
                             hovertemplate="Time: %{x:.2f} years<br>Predator: %{y:.1f}"))
    fig.add_trace(go.Scatter(x=t_rk4, y=sol_rk4[:,1], name='Predator (RK4)', line=dict(color=colors['rk4_pred'], width=2, dash='dot'),
                             hovertemplate="Time: %{x:.2f} years<br>Predator: %{y:.1f}"))
    fig.add_trace(go.Scatter(x=t_ref, y=sol_ref[1], name='Predator (Reference)', line=dict(color=colors['reference_pred'], width=2, dash='dash'),
                             hovertemplate="Time: %{x:.2f} years<br>Predator: %{y:.1f}"))

    fig.update_layout(
        title="Population Trajectories: Euler vs RK4 vs Reference Methods",
        xaxis_title="Time (years)",
        yaxis_title="Population Density",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig

def create_phase_portrait(sol_euler, sol_rk4, equilibria, stability, alpha, beta, delta, gamma):
    """Create a phase portrait with nullclines, equilibria, and trajectories.

    Args:
        sol_euler, sol_rk4 (np.array): Solutions from Euler and RK4 methods.
        equilibria (np.array): Equilibrium points.
        stability (list): Stability analysis results.
        alpha, beta, delta, gamma (float): Model parameters.

    Returns:
        go.Figure: Plotly figure of phase space with nullclines and stability markers.
    """
    fig = go.Figure()
    colors = {
        'euler_prey': '#377eb8', 'rk4_prey': '#4daf4a',
        'nullcline_prey': '#984ea3', 'nullcline_pred': '#ff7f00'
    }

    fig.add_trace(go.Scatter(x=sol_euler[:,0], y=sol_euler[:,1], name='Euler Trajectory', line=dict(color=colors['euler_prey'], width=2),
                             hovertemplate="Prey: %{x:.1f}<br>Predator: %{y:.1f}"))
    fig.add_trace(go.Scatter(x=sol_rk4[:,0], y=sol_rk4[:,1], name='RK4 Trajectory', line=dict(color=colors['rk4_prey'], width=2, dash='dot'),
                             hovertemplate="Prey: %{x:.1f}<br>Predator: %{y:.1f}"))

    x_max = max(500, sol_euler[:,0].max()*1.2)
    y_max = max(100, sol_euler[:,1].max()*1.2)

    if abs(beta) > EPSILON:
        y_null = alpha / beta
        fig.add_shape(type='line', x0=0, x1=x_max, y0=y_null, y1=y_null,
                      line=dict(color=colors['nullcline_prey'], width=2, dash='dash'))
        fig.add_annotation(
            x=x_max, y=y_null * 1.05,
            text="Prey Nullcline (dx/dt=0)",
            showarrow=False,
            font=dict(color=colors['nullcline_prey'], size=12),
            xanchor='center',
            yanchor='bottom'
        )

    if abs(delta) > EPSILON:
        x_null = gamma / delta
        fig.add_shape(type='line', x0=x_null, x1=x_null, y0=0, y1=y_max,
                      line=dict(color=colors['nullcline_pred'], width=2, dash='dash'))
        fig.add_annotation(
            x=x_null * 1.05, y=y_max,
            text="Predator Nullcline (dy/dt=0)",
            showarrow=False,
            font=dict(color=colors['nullcline_pred'], size=12),
            xanchor='left',
            yanchor='middle'
        )

    for eq in stability:
        color = '#e41a1c' if 'Unstable' in eq['stability'] else '#4daf4a'
        fig.add_trace(go.Scatter(
            x=[eq['point'][0]], y=[eq['point'][1]],
            mode='markers',
            marker=dict(size=14, color=color),
            hovertemplate=f"Point: ({eq['point'][0]:.1f}, {eq['point'][1]:.1f})<br>Stability: {eq['stability']}"
        ))

    fig.update_layout(
        title="Phase Space Analysis with Nullclines",
        xaxis_title="Prey Population (Krill)",
        yaxis_title="Predator Population (Whales)",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig

def create_error_analysis(t_euler, sol_euler, t_rk4, sol_rk4, t_ref, sol_ref):
    """Plot numerical errors relative to the reference solution for educational comparison.

    Args:
        t_euler, t_rk4, t_ref (np.array): Time points for each method.
        sol_euler, sol_rk4, sol_ref (np.array): Solutions for each method.

    Returns:
        tuple: (relative error figure, cumulative error figure).
    """
    sol_ref_array = np.array(sol_ref)
    ref_euler = np.zeros_like(sol_euler)
    ref_rk4 = np.zeros_like(sol_rk4)
    for i in range(2):
        ref_euler[:, i] = np.interp(t_euler, t_ref, sol_ref_array[i])
        ref_rk4[:, i] = np.interp(t_rk4, t_ref, sol_ref_array[i])

    error_euler = np.abs(sol_euler - ref_euler)
    error_rk4 = np.abs(sol_rk4 - ref_rk4)

    rel_error_euler = error_euler / (ref_euler + EPSILON)
    rel_error_rk4 = error_rk4 / (ref_rk4 + EPSILON)

    cum_error_euler = np.cumsum(np.sum(error_euler, axis=1))
    cum_error_rk4 = np.cumsum(np.sum(error_rk4, axis=1))

    fig_rel = go.Figure()
    colors = {'error_euler': '#8c564b', 'error_rk4': '#e377c2'}
    fig_rel.add_trace(go.Scatter(x=t_euler, y=rel_error_euler[:,0], name='Prey (Euler)', line=dict(color=colors['error_euler'], width=2),
                                 hovertemplate="Time: %{x:.2f} years<br>Error: %{y:.3f}"))
    fig_rel.add_trace(go.Scatter(x=t_euler, y=rel_error_euler[:,1], name='Predator (Euler)', line=dict(color=colors['error_euler'], width=2, dash='dot'),
                                 hovertemplate="Time: %{x:.2f} years<br>Error: %{y:.3f}"))
    fig_rel.add_trace(go.Scatter(x=t_rk4, y=rel_error_rk4[:,0], name='Prey (RK4)', line=dict(color=colors['error_rk4'], width=2),
                                 hovertemplate="Time: %{x:.2f} years<br>Error: %{y:.3f}"))
    fig_rel.add_trace(go.Scatter(x=t_rk4, y=rel_error_rk4[:,1], name='Predator (RK4)', line=dict(color=colors['error_rk4'], width=2, dash='dot'),
                                 hovertemplate="Time: %{x:.2f} years<br>Error: %{y:.3f}"))

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=t_euler, y=cum_error_euler, name='Euler', line=dict(color=colors['error_euler'], width=2),
                                 hovertemplate="Time: %{x:.2f} years<br>Cum. Error: %{y:.3f}"))
    fig_cum.add_trace(go.Scatter(x=t_rk4, y=cum_error_rk4, name='RK4', line=dict(color=colors['error_rk4'], width=2),
                                 hovertemplate="Time: %{x:.2f} years<br>Cum. Error: %{y:.3f}"))

    for fig in [fig_rel, fig_cum]:
        fig.update_layout(
            template="plotly_white",
            height=400,
            xaxis_title="Time (years)",
            yaxis_type="log",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
    fig_rel.update_layout(title="Relative Numerical Errors", yaxis_title="Relative Error")
    fig_cum.update_layout(title="Cumulative Absolute Error", yaxis_title="Cumulative Error")

    return fig_rel, fig_cum

def create_cross_correlation(sol_euler, dt, t_euler, t_max):
    """Compute and plot cross-correlation with subplots for peaks and population comparisons.

    Args:
        sol_euler (np.array): Euler solution [prey, predator].
        dt (float): Time step.
        t_euler (np.array): Time points for Euler method.
        t_max (float): Simulation duration for dynamic axis scaling.

    Returns:
        tuple: (plotly figure with subplots, peak info text, confidence threshold).
    """
    try:
        window = windows.tukey(len(sol_euler[:,0]), alpha=0.1)
        prey = (sol_euler[:,0] - np.mean(sol_euler[:,0])) * window
        pred = (sol_euler[:,1] - np.mean(sol_euler[:,1])) * window

        corr = correlate(prey, pred, mode='full', method='fft')
        corr /= np.sqrt(np.sum(prey**2) * np.sum(pred**2)) + EPSILON
        lags = np.arange(-len(sol_euler)+1, len(sol_euler)) * dt

        peaks, properties = find_peaks(corr, height=0.1)
        peak_info = "No significant peaks found. Try adjusting parameters."
        if len(peaks) > 0:
            peak_info = "<br>".join([
                f"Peak {i+1}: r = {corr[peak]:.3f} at lag {lags[peak]:.2f} years"
                for i, peak in enumerate(peaks)
            ])

        prey_norm = (sol_euler[:,0] - np.min(sol_euler[:,0])) / (np.max(sol_euler[:,0]) - np.min(sol_euler[:,0]) + EPSILON)
        pred_norm = (sol_euler[:,1] - np.min(sol_euler[:,1])) / (np.max(sol_euler[:,1]) - np.min(sol_euler[:,1]) + EPSILON)

        prey_peaks, _ = find_peaks(prey_norm, height=0.5)
        pred_peaks, _ = find_peaks(pred_norm, height=0.5)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                            subplot_titles=("Cross-Correlation Peaks", "Krill and Whale Population Peaks"),
                            vertical_spacing=0.15)

        # subplot 1: cross-correlation
        fig.add_trace(go.Scatter(
            x=lags, y=corr,
            name='Cross-Correlation',
            line=dict(color='#377eb8', width=2),
            hovertemplate="Lag: %{x:.2f} years<br>Correlation: %{y:.3f}"
        ), row=1, col=1)

        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=lags[peaks], y=corr[peaks],
                mode='markers',
                marker=dict(size=12, color='#e41a1c', line=dict(width=1, color='DarkSlateGrey')),
                name='Correlation Peaks',
                hoverinfo="skip"
            ), row=1, col=1)

        confidence_threshold = 0.5/np.sqrt(len(sol_euler))
        fig.add_hrect(
            y0=-confidence_threshold, y1=confidence_threshold,
            fillcolor="rgba(200,200,200,0.2)",
            line_width=0,
            row=1, col=1
        )

        # subplot 2: population peaks
        fig.add_trace(go.Scatter(
            x=t_euler, y=prey_norm,
            name='Krill (Normalized)',
            line=dict(color='#4daf4a', width=2, dash='dash'),
            hovertemplate="Time: %{x:.2f} years<br>Krill: %{y:.3f}"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=t_euler, y=pred_norm,
            name='Whale (Normalized)',
            line=dict(color='#ff7f00', width=2, dash='dot'),
            hovertemplate="Time: %{x:.2f} years<br>Whale: %{y:.3f}"
        ), row=2, col=1)

        if len(prey_peaks) > 0:
            fig.add_trace(go.Scatter(
                x=t_euler[prey_peaks], y=prey_norm[prey_peaks],
                mode='markers',
                marker=dict(size=10, color='#4daf4a', symbol='circle'),
                name='Krill Peaks',
                hoverinfo="skip"
            ), row=2, col=1)
        if len(pred_peaks) > 0:
            fig.add_trace(go.Scatter(
                x=t_euler[pred_peaks], y=pred_norm[pred_peaks],
                mode='markers',
                marker=dict(size=10, color='#ff7f00', symbol='circle'),
                name='Whale Peaks',
                hoverinfo="skip"
            ), row=2, col=1)

        lag_range = min(t_max * 0.2, 20)
        fig.update_layout(
            title="Cross-Correlation and Population Peak Analysis",
            template="plotly_white",
            height=800,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        fig.update_xaxes(title_text="Lag (years)", range=[-lag_range, lag_range], row=1, col=1)
        fig.update_xaxes(title_text="Time (years)", row=2, col=1)
        fig.update_yaxes(title_text="Normalized Correlation", range=[-1, 1], row=1, col=1)
        fig.update_yaxes(title_text="Normalized Population", range=[0, 1.2], row=2, col=1)

        return fig, peak_info, confidence_threshold

    except Exception as e:
        return go.Figure(), f"Error computing cross-correlation: {str(e)}", 0.0

def create_direction_field(simulator, sol_euler, sol_rk4, sol_ref):
    """Plot a simplified direction field with a nearly periodic curved arrow and numerical trajectories.

    Args:
        simulator (LotkaVolterraSimulator): Simulator instance for computing dynamics.
        sol_euler, sol_rk4, sol_ref (np.array): Solutions for Euler, RK4, and reference methods.

    Returns:
        go.Figure: Plotly figure with a curved arrow and trajectories.
    """
    fig = go.Figure()

    all_x = np.concatenate([sol_euler[:,0], sol_rk4[:,0], sol_ref[0]])
    all_y = np.concatenate([sol_euler[:,1], sol_rk4[:,1], sol_ref[1]])
    x_max = np.max(all_x) * 1.2
    y_max = np.max(all_y) * 1.2
    x_min, y_min = 0, 0

    t = np.linspace(0, 2*np.pi*0.8, 100)
    x_center = 0.88 * x_max
    y_center = 0.88 * y_max
    radius_x = 0.07 * x_max
    radius_y = 0.07 * y_max
    x_arrow = x_center + radius_x * np.cos(t)
    y_arrow = y_center + radius_y * np.sin(t)
    fig.add_trace(go.Scatter(
        x=x_arrow, y=y_arrow,
        mode='lines',
        line=dict(color='#ef00e9', width=2.0),
        name='Direction (Counterclockwise Cycle)',
        hoverinfo='skip'
    ))
    last_t = t[-1]
    dx_dt = -radius_x * np.sin(last_t)
    dy_dt = radius_y * np.cos(last_t)
    angle = np.arctan2(dy_dt, dx_dt) * 160 / np.pi
    fig.add_trace(go.Scatter(
        x=[x_arrow[-1]], y=[y_arrow[-1]],
        mode='markers',
        marker=dict(size=8, color='#ef00e9', symbol='triangle-right', angle=angle),
        showlegend=False,
        hoverinfo='skip'
    ))

    methods = [
        {'data': sol_euler, 'color': '#377eb8', 'name': 'Euler', 'dash': None, 'line_width': 2.5},
        {'data': sol_rk4, 'color': '#4daf4a', 'name': 'RK4', 'dash': 'dot', 'line_width': 2.5},
        {'data': np.array(sol_ref).T, 'color': '#984ea3', 'name': 'Reference', 'dash': 'dash', 'line_width': 2.5}
    ]

    for method in methods:
        x = method['data'][:, 0]
        y = method['data'][:, 1]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            name=method['name'],
            line=dict(color=method['color'], width=method['line_width'], dash=method['dash']),
            mode='lines',
            hovertemplate="Prey: %{x:.1f}<br>Predator: %{y:.1f}"
        ))
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[y[0]],
            mode='markers',
            marker=dict(size=10, color=method['color'], symbol='circle'),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title="System Direction Field",
        xaxis_title="Prey Population (Krill)",
        yaxis_title="Predator Population (Whales)",
        template="plotly_white",
        height=600,
        xaxis=dict(range=[x_min, x_max], gridcolor='rgba(200,200,200,0.2)'),
        yaxis=dict(range=[y_min, y_max], gridcolor='rgba(200,200,200,0.2)'),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor='rgba(245,245,245,1)'
    )
    return fig

def create_eigenvalue_plot(stability):
    """Plot eigenvalues in the complex plane with stability regions.

    Args:
        stability (list): Stability analysis results.

    Returns:
        go.Figure: Plotly figure showing eigenvalues and stability regions.
    """
    fig = go.Figure()

    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        line=dict(color='rgba(200,200,200,0.7)', width=1),
        name='Unit Circle (|λ|=1)'
    ))

    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-1.5, y0=-1.5, x1=1.5, y1=1.5,
                  fillcolor="rgba(255,255,255,0)",
                  line_color="rgba(200,200,200,0.5)")
    fig.add_shape(type="rect",
                  x0=-1.5, y0=-1.5, x1=0, y1=1.5,
                  fillcolor="rgba(0,204,150,0.1)",
                  line_width=0,
                  name="Stable Region")

    stability_data = {
        'Stable Node': {'color': '#4daf4a', 'symbol': 'circle'},
        'Stable Spiral': {'color': '#377eb8', 'symbol': 'diamond'},
        'Unstable Node': {'color': '#e41a1c', 'symbol': 'circle'},
        'Unstable Spiral': {'color': '#ff7f00', 'symbol': 'diamond'},
        'Saddle Point': {'color': '#984ea3', 'symbol': 'x'},
        'Neutrally Stable': {'color': '#fecb52', 'symbol': 'star'}
    }

    for eq in stability:
        stability_type = eq['stability'].split(' (')[0]
        style = stability_data.get(stability_type, {'color': '#377eb8', 'symbol': 'circle'})

        for i, eig in enumerate(eq['eigenvalues']):
            fig.add_trace(go.Scatter(
                x=[np.real(eig)], y=[np.imag(eig)],
                mode='markers',
                marker=dict(size=14, color=style['color'], symbol=style['symbol'], line=dict(width=1, color='DarkSlateGrey')),
                name=f"{eq['stability']}",
                hovertemplate=f"<b>{eq['stability']}</b><br>Eigenvalue: {np.real(eig):.3f} + {np.imag(eig):.3f}i<br>Position: ({eq['point'][0]:.1f}, {eq['point'][1]:.1f})"
            ))

    fig.update_layout(
        title="Stability Analysis: Eigenvalues in Complex Plane",
        xaxis_title="Real Part (Re(λ))",
        yaxis_title="Imaginary Part (Im(λ))",
        template="plotly_white",
        height=600,
        xaxis=dict(range=[-1.5, 1.5], zeroline=True),
        yaxis=dict(range=[-1.5, 1.5], zeroline=True, scaleanchor="x"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig

def create_stability_table(stability):
    """Generate a table summarizing equilibrium stability for educational display.

    Args:
        stability (list): Stability analysis results.

    Returns:
        list: Table data for streamlit display.
    """
    table_data = []
    for i, eq in enumerate(stability):
        eigenvalues = []
        for j, eig in enumerate(eq['eigenvalues']):
            eigenvalues.append(
                f"λ{j+1} = {np.real(eig):.3f} {'+' if np.imag(eig) >= 0 else ''}{np.imag(eig):.3f}i"
            )

        table_data.append([
            f"Equilibrium {i+1}",
            f"({eq['point'][0]:.2f}, {eq['point'][1]:.2f})",
            eq['stability'],
            ", ".join(eigenvalues),
            "Stable" if "Stable" in eq['stability'] else "Unstable"
        ])

    return table_data

# ====================== unit tests ======================
class TestLotkaVolterraSimulator(unittest.TestCase):
    """Unit tests for LotkaVolterraSimulator to ensure reliability."""

    def setUp(self):
        self.sim = LotkaVolterraSimulator()
        self.sim.params = {'alpha': 0.5, 'beta': 0.01, 'delta': 0.001, 'gamma': 0.2}
        self.sim.initial_conditions = np.array([200.0, 15.0])

    def test_lotka_volterra(self):
        """Test differential equations output."""
        state = np.array([100.0, 10.0])
        dxdt, dydt = self.sim.lotka_volterra(0, state)
        self.assertAlmostEqual(dxdt, 0.5 * 100 - 0.01 * 100 * 10, places=5)
        self.assertAlmostEqual(dydt, 0.001 * 100 * 10 - 0.2 * 10, places=5)

    def test_equilibria(self):
        """Test equilibrium points calculation."""
        equilibria = self.sim.find_equilibria()
        self.assertEqual(len(equilibria), 2)
        self.assertTrue(np.allclose(equilibria[0], [0.0, 0.0]))
        self.assertTrue(np.allclose(equilibria[1], [0.2/0.001, 0.5/0.01]))

# ====================== streamlit app ======================
def main():
    """Main function to run the Streamlit app for educational predator-prey simulation.

    This function sets up an interactive web application to explore the Lotka-Volterra model,
    allowing users to adjust parameters, visualize dynamics, and learn ecological concepts.
    """
    st.set_page_config(page_title="Predator-Prey Simulator", layout="wide")
    st.markdown("""
        <style>
        .main {padding: 1rem;}
        @media (max-width: 600px) {
            .main {padding: 0.5rem;}
            .stSlider > div {font-size: 12px;}
        }
        </style>
        <script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>
    """, unsafe_allow_html=True)  # responsive design and mathjax

    st.title("🐳🦐 Lotka-Volterra Predator-Prey Simulator")
    st.markdown("""
    **Explore ecological dynamics through the Lotka-Volterra model**

    This interactive simulator visualizes the predator-prey relationship between whales and krill,
    designed for teaching and learning ecological and mathematical concepts. Adjust parameters to
    explore population dynamics, stability, and numerical methods.
    """)

    # sidebar controls
    with st.sidebar:
        st.header("Simulation Controls")
        scenario = st.radio(
            "Select Ecological Scenario:",
            list(SCENARIOS.keys()),
            index=0,
            help="Pre-configured scenarios demonstrating ecological crises"
        )

        st.subheader("Model Parameters")
        alpha = st.slider(
            "$\\alpha$: Krill Growth", 0.01, 1.0, SCENARIOS[scenario]['alpha'], 0.01,
            help="How fast krill reproduce without predators (higher = faster growth)"
        )
        beta = st.slider(
            "$\\beta$: Predation Rate", 0.001, 0.1, SCENARIOS[scenario]['beta'], 0.001, format="%.3f",
            help="How effectively whales hunt krill (higher = more predation)"
        )
        delta = st.slider(
            "$\\delta$: Conversion Efficiency", 0.0001, 0.01, SCENARIOS[scenario]['delta'], 0.0001, format="%.4f",
            help="Efficiency of converting eaten krill into new whales (higher = better conversion)"
        )
        gamma = st.slider(
            "$\\gamma$: Whale Mortality", 0.01, 1.0, SCENARIOS[scenario]['gamma'], 0.01,
            help="Natural whale death rate (higher = faster population decline)"
        )

        st.subheader("Initial Populations")
        x0 = st.slider("Krill Population", 10.0, 500.0, 200.0, 10.0)
        y0 = st.slider("Whale Population", 1.0, 50.0, 15.0, 1.0)

        eq_prey = gamma/delta if delta > EPSILON else 0
        eq_pred = alpha/beta if beta > EPSILON else 0

        if np.abs(x0 - eq_prey) < 1e-3 and np.abs(y0 - eq_pred) < 1e-3:
            st.warning("Initial populations set to equilibrium values. Small perturbations are needed to observe dynamics.")

        st.subheader("Simulation Settings")
        t_max = st.slider("Duration (years)", 10, 500, 100, 10)
        dt = st.slider("Time Step", 0.01, 1.0, 0.1, 0.01)

    # main content
    with st.expander("🔍 ECOLOGICAL PARAMETER GUIDE", expanded=True):
        st.markdown("""
        ### Parameter Effects on Ecosystem Dynamics
        | Parameter | Biological Meaning | High Value Effect | Low Value Effect |
        |-----------|--------------------|-------------------|------------------|
        | **$\\alpha$** | Krill reproduction rate | Faster krill growth, larger cycles | Slower recovery, smaller populations |
        | **$\\beta$** | Whale hunting efficiency | Stronger predator control, prey suppression | Weaker control, prey dominance |
        | **$\\delta$** | Energy conversion (krill→whales) | Faster whale growth, larger predator cycles | Slower whale growth, predator decline |
        | **$\\gamma$** | Whale mortality rate | Faster predator decline, prey release | Persistent predators, stable cycles |

        **Key Insight:** Small parameter changes can lead to dramatic shifts in ecosystem stability, potentially causing collapses or booms.
        """)

    simulator = LotkaVolterraSimulator()
    simulator.params = {'alpha': alpha, 'beta': beta, 'delta': delta, 'gamma': gamma}
    simulator.initial_conditions = np.array([x0, y0])
    simulator.t_span = [0, t_max]
    simulator.dt = dt

    with st.spinner("Computing ecosystem dynamics..."):
        t_euler, sol_euler = simulator.solve('euler')
        t_rk4, sol_rk4 = simulator.solve('rk4')
        t_ref, sol_ref = simulator.solve('reference')
        equilibria = simulator.find_equilibria()
        stability = simulator.analyze_stability(equilibria)

    # visualizations
    with st.expander("📈 POPULATION DYNAMICS", expanded=True):
        fig_pop = create_population_plot(t_euler, sol_euler, t_rk4, sol_rk4, t_ref, sol_ref)
        st.plotly_chart(fig_pop, use_container_width=True)
        st.markdown("""
        ### Interpreting Population Cycles
        1. **Predator-Prey Lag**: Whale peaks follow krill peaks by ~1/4 cycle due to predation delay.
        2. **Numerical Accuracy**:
           - Euler (solid lines) accumulates errors, causing divergence.
           - RK4 (dotted) closely matches the reference (dashed).
        3. **Amplitude & Frequency**:
           - Higher $\\alpha/\\delta$ increases oscillation frequency.
           - Higher $\\beta/\\gamma$ amplifies boom-bust cycles.
        """)

    with st.expander("🔄 PHASE SPACE ANALYSIS", expanded=True):
        fig_phase = create_phase_portrait(sol_euler, sol_rk4, equilibria, stability, alpha, beta, delta, gamma)
        st.plotly_chart(fig_phase, use_container_width=True)
        st.markdown("""
        ### Phase Space Key Features
        - **Nullclines**: Lines where prey (purple) or predator (orange) growth stops.
        - **Equilibrium Points**:
          - Red: Unstable, system diverges.
          - Green: Stable, system converges.
        - **Closed Orbits**: Indicate periodic population cycles (neutral stability).
        - **Trajectories**: Show evolution from initial conditions.
        """)

    with st.expander("🧭 SYSTEM DIRECTION FIELD", expanded=True):
        fig_field = create_direction_field(simulator, sol_euler, sol_rk4, np.array(sol_ref))
        st.plotly_chart(fig_field, use_container_width=True)
        st.markdown("""
        ### Direction Field Insights
        - **Curved Arrow**: Indicates the direction of population cycles (counterclockwise in stable systems).
        - **Trajectories**:
          - Euler (blue): May diverge due to numerical errors.
          - RK4 (green): More accurate, follows reference (purple).
        - **Cycle Behavior**: Closed loops suggest sustained oscillations.
        """)

    st.divider()
    with st.expander("🔍 NUMERICAL ERROR ANALYSIS", expanded=True):
        fig_rel, fig_cum = create_error_analysis(t_euler, sol_euler, t_rk4, sol_rk4, t_ref, sol_ref)
        st.plotly_chart(fig_rel, use_container_width=True)
        st.plotly_chart(fig_cum, use_container_width=True)
        st.markdown("""
        ### Error Analysis Takeaways
        1. **Relative Error**:
           - Spikes occur near zero populations due to division artifacts.
           - RK4 maintains lower error than Euler.
        2. **Cumulative Error**:
           - Euler error grows linearly over time.
           - RK4 error grows slower due to higher-order accuracy.
        3. **Implications**:
           - RK4 is preferred for long-term simulations.
        """)

    with st.expander("⏳ CROSS-CORRELATION ANALYSIS", expanded=True):
        fig_corr, peak_info, threshold = create_cross_correlation(sol_euler, dt, t_euler, t_max)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown(f"""
        ### Cross-Correlation Insights
        **Peak Detection:**

        {peak_info}

        **Confidence Threshold:** ±{threshold:.3f}

        **Interpretation:**
        - **Top Plot**: Peaks show the lag between krill and whale cycles (positive = whales follow krill).
        - **Bottom Plot**: Krill and whale peaks highlight timing differences in population cycles.
        - Strongest peak lag ≈ 1/4 cycle length, reflecting predation delay.
        """, unsafe_allow_html=True)

    st.divider()
    with st.expander("📊 STABILITY ANALYSIS", expanded=True):
        st.markdown("""
        ### System Stability Characteristics
        Eigenvalues determine equilibrium behavior:
        - **Left half-plane ($\\Re(\\lambda) < 0$)**: Stable, system returns to equilibrium.
        - **Right half-plane ($\\Re(\\lambda) > 0$)**: Unstable, system diverges.
        - **Imaginary part ($\\Im(\\lambda)$)**: Indicates oscillatory behavior (spirals or centers).
        """)

        fig_eigen = create_eigenvalue_plot(stability)
        st.plotly_chart(fig_eigen, use_container_width=True)

        st.markdown("### Equilibrium Point Analysis")
        table_data = create_stability_table(stability)
        st.table(pd.DataFrame(
            table_data,
            columns=["Equilibrium", "Position", "Type", "Eigenvalues", "Stability"]
        ))

        st.markdown("""
        **Interpretation Guide:**
        - **Stable Node/Spiral**: System returns to equilibrium after disturbances.
        - **Unstable Node/Spiral**: System diverges from equilibrium.
        - **Saddle Point**: Stable in some directions, unstable in others.
        - **Neutrally Stable (Center)**: System sustains perfect cycles.
        """)

    with st.expander("🌍 ECOLOGICAL INTERPRETATION", expanded=True):
        prey_min, prey_max = np.min(sol_ref[0]), np.max(sol_ref[0])
        pred_min, pred_max = np.min(sol_ref[1]), np.max(sol_ref[1])

        cols = st.columns(2)
        cols[0].metric("Krill Population Range", f"{prey_min:.1f} - {prey_max:.1f}")
        cols[1].metric("Whale Population Range", f"{pred_min:.1f} - {pred_max:.1f}")

        st.markdown("""
        ### Real-World Implications
        1. **Overfishing (Low $\\alpha$)**: Reduces krill recovery, leading to smaller whale populations and potential ecosystem imbalance.
        2. **Hunting (High $\\gamma$)**: Increases whale mortality, risking predator collapse and prey overpopulation.
        3. **Climate Change (Low $\\delta$)**: Decreases energy conversion efficiency, limiting whale growth and cycle amplitude.
        4. **Trophic Cascade**: Changes in one population (e.g., krill decline) can cascade, affecting the entire food web.
        5. **Conservation Strategies**:
           - Protecting krill habitats ensures prey availability.
           - Regulating whale hunting stabilizes predator populations.
           - Mitigating climate impacts supports energy transfer efficiency.
        """)


    with st.expander("🧮 MATHEMATICAL REFERENCE", expanded=False):
        st.markdown(r"""
        ## Core Concepts
        ### Differential Equations
        The Lotka-Volterra model consists of two coupled, nonlinear ordinary differential equations:

        $\frac{dx}{dt} = \alpha x - \beta xy$ (Prey population)

        $\frac{dy}{dt} = \delta xy - \gamma y$ (Predator population)

        Where:
        - $x(t)$: Prey population (krill) at time $t$
        - $y(t)$: Predator population (whales) at time $t$
        - $\alpha$: Prey birth rate (1/time)
        - $\beta$: Predation rate (1/(predator·time))
        - $\delta$: Conversion efficiency of prey to predators (predator/prey)
        - $\gamma$: Predator death rate (1/time)

        ### Biological Interpretation
        - $\alpha x$: Exponential prey growth in absence of predators.
        - $-\beta xy$: Prey loss due to predation (mass-action kinetics).
        - $\delta xy$: Predator growth proportional to prey consumption.
        - $-\gamma y$: Natural predator mortality.

        ## Detailed Analysis
        ### Equilibrium Points
        The system has two equilibrium points where $\frac{dx}{dt} = \frac{dy}{dt} = 0$:
        1. **Trivial Extinction (0, 0)**:
           - Both populations extinct.
           - Always exists but biologically unstable.
        2. **Coexistence Equilibrium $\left( \frac{\gamma}{\delta}, \frac{\alpha}{\beta} \right)$**:
           - Non-zero populations for both species.
           - Exists when $\delta \neq 0$ and $\beta \neq 0$.
           - Typically neutrally stable, leading to cycles.

        ### Jacobian Matrix
        The Jacobian matrix captures local linear behavior:
        $$
        J = \begin{bmatrix}
            \frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} \\
            \frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y}
        \end{bmatrix}
        = \begin{bmatrix}
            \alpha - \beta y & -\beta x \\
            \delta y & \delta x - \gamma
        \end{bmatrix}
        $$
        Where $f_1 = \frac{dx}{dt}$ and $f_2 = \frac{dy}{dt}$.

        ### Stability Analysis
        1. **At (0,0)**:
           - Eigenvalues: $\lambda_1 = \alpha$, $\lambda_2 = -\gamma$
           - Saddle point (unstable) due to one positive and one negative eigenvalue.
        2. **At $\left( \frac{\gamma}{\delta}, \frac{\alpha}{\beta} \right)$**:
           - Eigenvalues: $\lambda = \pm i \sqrt{\alpha \gamma}$
           - Purely imaginary, indicating a center (neutrally stable cycles).
           - Small perturbations may introduce stability or instability in real systems.

        ### Numerical Methods
        Three integration methods are implemented:
        1. **Forward Euler (1st-order)**:
        $$
        y_{n+1} = y_n + \Delta t \cdot f(t_n, y_n)
        $$
           - Simple but prone to error accumulation (error $\propto \Delta t$).
           - Suitable for short simulations or educational demonstrations.
        2. **Runge-Kutta 4 (4th-order)**:
        $$
           \begin{align}
           k_1 &= f(t_n, y_n) \\
           k_2 &= f\left(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2} \cdot k_1\right) \\
           k_3 &= f\left(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2} \cdot k_2\right) \\
           k_4 &= f\left(t_n + \Delta t, y_n + \Delta t \cdot k_3\right) \\
           y_{n+1} &= y_n + \frac{\Delta t}{6} \cdot (k_1 + 2k_2 + 2k_3 + k_4)
           \end{align}
        $$
           - Highly accurate (error $\propto \Delta t^4$).
           - Preferred for long-term simulations.
        3. **LSODA (Reference)**:
           - Adaptive step-size method from SciPy.
           - Combines Adams (non-stiff) and BDF (stiff) methods.
           - Provides high precision for benchmarking.

        ### Phase Space Analysis
        - **Nullclines**:
          - Prey nullcline ($\frac{dx}{dt} = 0$): $y = \frac{\alpha}{\beta}$ or $x = 0$
          - Predator nullcline ($\frac{dy}{dt} = 0$): $x = \frac{\gamma}{\delta}$ or $y = 0$
        - Intersections are equilibrium points.
        - Closed orbits indicate periodic solutions, typical in Lotka-Volterra systems.

        ### Cross-Correlation Analysis
        Measures the lag between predator and prey populations:
        - **Positive peak lag**: Predators follow prey, reflecting biological delay.
        - **Correlation strength**: Indicates coupling between populations (1 = perfect correlation).
        - **Normalized to [-1, 1]**: Allows comparison across simulations.
        - **Educational Value**: Demonstrates temporal relationships in ecological systems.

        ### Limitations and Extensions
        - **Model Simplifications**:
          - Assumes constant parameters (real ecosystems vary).
          - Ignores spatial dynamics, stochasticity, and external factors (e.g., climate).
        - **Possible Extensions**:
          - Add carrying capacity for prey (logistic growth).
          - Include time-varying parameters for environmental changes.
          - Model multiple species for complex food webs.
        - **Real-World Relevance**:
          - Used in ecology to predict population cycles.
          - Informs conservation strategies by highlighting sensitive parameters.
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
