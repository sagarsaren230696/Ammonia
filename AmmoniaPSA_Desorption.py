import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyomo.environ import *
from pyomo.dae import ContinuousSet, DerivativeVar


# ============================================================
# READ FINAL ADSORPTION STATE
# ============================================================

adsorption_csv = "breakthrough_results/breakthrough_state.csv"

df_ads = pd.read_csv(adsorption_csv)

# Final adsorption time
t_ads_final = df_ads["time_s"].max()

df_final = df_ads[
    np.isclose(df_ads["time_s"], t_ads_final)
].copy()

df_final = df_final.sort_values("x")

x_saved = df_final["x"].to_numpy()


def interp_final(column, x):
    """
    Interpolate final adsorption solution onto the current
    spatial discretization.
    """
    return np.interp(
        float(x),
        x_saved,
        df_final[column].to_numpy()
    )


def y_ads_final(s, x):
    return interp_final(f"y_{s}", x)


def q_ads_final(s, x):
    return interp_final(f"q_{s}_mol_kg", x)


def qstar_ads_final(s, x):
    return interp_final(f"qstar_{s}_mol_kg", x)


def P_ads_final(x):
    return interp_final("P_Pa", x)


def T_ads_final(x):
    return interp_final("T_K", x)


def u_ads_final(x):
    return interp_final("u_m_s", x)


def rho_ads_final(x):
    return interp_final("rho_g_kg_m3", x)

# ============================================================
# DESORPTION CONDITIONS
# ============================================================

Ru = 8.314462618

P_des = 1.0e4                # Pa = 10 kPa

t_phys = 50.0                # desorption time [s]
T_initial = 473.15        # K

# ============================================================
# BED
# ============================================================

L = 1.0                   # m

epsilon = 0.40

# IMPORTANT:
# rho_b = kg adsorbent / m3 BED
rho_b = 700.0

dp = 2.0e-3               # m
mu = 2.0e-5               # Pa s

Cp_g = 30.0               # J/mol/K
Cp_s = 1000.0             # J/kg/K

k_eff = 0.5               # W/m/K


# ============================================================
# DISCRETIZATION
# ============================================================

Nfex = 40

Nfet = 5
NCP = 5


# ============================================================
# COMPONENTS
# ============================================================

components = ['N2', 'H2', 'NH3']

y_feed = {
    'N2': 0.23,
    'H2': 0.71,
    'NH3': 0.06
}

MW_data = {
    'N2': 28.0134e-3,
    'H2': 2.01588e-3,
    'NH3': 17.0305e-3
}


# ============================================================
# HEAT OF ADSORPTION
# positive magnitude
# ============================================================

dHads_data = {
    'N2': 15000.0,
    'H2': 10000.0,
    'NH3': 40000.0
}


# ============================================================
# PLACEHOLDER LANGMUIR-FREUNDLICH
# ============================================================

qsat_data = {
    'N2': 1.0,
    'H2': 0.5,
    'NH3': 5.0
}

b_data = {
    'N2': 0.01,      # 1/bar
    'H2': 0.005,
    'NH3': 1.0
}

n_data = {
    'N2': 1.0,
    'H2': 1.0,
    'NH3': 0.8
}

kLDF_data = {
    'N2': 0.05,
    'H2': 0.05,
    'NH3': 0.20
}


# ============================================================
# INITIAL BED COMPOSITION
#
# Start almost NH3-free.
# This gives an actual breakthrough problem.
# ============================================================

y_initial = {
    'N2': 0.25,
    'H2': 0.749999,
    'NH3': 1.0e-6
}


# ============================================================
# NUMERIC EQUILIBRIUM FUNCTION
# ============================================================

def qeq_numeric(s, y, P_pa):

    Pbar = P_pa/1.0e5

    denominator = 1.0 + sum(
        (b_data[k]*max(y[k]*Pbar, 1e-12))**n_data[k]
        for k in components
    )

    numerator = (
        qsat_data[s]
        * (b_data[s]*max(y[s]*Pbar, 1e-12))**n_data[s]
    )

    return numerator/denominator


q_initial = {
    s: qeq_numeric(s, y_initial, P_des)
    for s in components
}


# ============================================================
# MODEL
# ============================================================

m = ConcreteModel()

m.S = Set(initialize=components)

# normalized axial coordinate: x = z/L
m.x = ContinuousSet(bounds=(0.0, 1.0))

# normalized time: tau = t/t_phys
m.t = ContinuousSet(bounds=(0.0, 1.0))


# ============================================================
# PARAMETERS
# ============================================================

m.epsilon = Param(initialize=epsilon)
m.rho_b = Param(initialize=rho_b)

m.dp = Param(initialize=dp)
m.mu = Param(initialize=mu)

m.Cp_g = Param(initialize=Cp_g)
m.Cp_s = Param(initialize=Cp_s)
m.k_eff = Param(initialize=k_eff)

m.MW = Param(m.S, initialize=MW_data)
m.dHads = Param(m.S, initialize=dHads_data)

m.qsat = Param(m.S, initialize=qsat_data)
m.b = Param(m.S, initialize=b_data)
m.n = Param(m.S, initialize=n_data)
m.kLDF = Param(m.S, initialize=kLDF_data)


# ============================================================
# STATE VARIABLES
# ============================================================

m.y = Var(
    m.S, m.t, m.x,
    bounds=(1.0e-10, 1.0),
    initialize=lambda m,s,t,x: y_initial[s]
)

m.P = Var(
    m.t, m.x,
    bounds=(1.0e3, 2.0e6),
    initialize=1.0e6
)

m.T = Var(
    m.t, m.x,
    bounds=(300.0, 800.0),
    initialize=T_initial
)

m.u = Var(
    m.t, m.x,
    bounds=(0.0, 30.0),
    initialize=0.01
)

m.q = Var(
    m.S, m.t, m.x,
    bounds=(0.0, None),
    initialize=lambda m,s,t,x: q_initial[s]
)

m.qstar = Var(
    m.S, m.t, m.x,
    bounds=(0.0, None),
    initialize=lambda m,s,t,x: q_initial[s]
)

m.Pi = Var(
    m.S, m.t, m.x,
    bounds=(1.0e-10, None),
    initialize=lambda m,s,t,x:
        max(y_initial[s]*P_des/1e5, 1e-8)
)


# ============================================================
# GAS DENSITY
# ============================================================

m.rho_g = Var(
    m.t, m.x,
    bounds=(1e-6, None),
    initialize=1.0
)


# ============================================================
# DERIVATIVES
# ============================================================

m.dydt = DerivativeVar(m.y, wrt=m.t)
m.dydx = DerivativeVar(m.y, wrt=m.x)

m.dPdt = DerivativeVar(m.P, wrt=m.t)
m.dPdx = DerivativeVar(m.P, wrt=m.x)

m.dTdt = DerivativeVar(m.T, wrt=m.t)
m.dTdx = DerivativeVar(m.T, wrt=m.x)

m.dudx = DerivativeVar(m.u, wrt=m.x)

m.dqdt = DerivativeVar(m.q, wrt=m.t)


# ============================================================
# GAS DENSITY
# ============================================================

def gas_density_rule(m,t,x):

    MWmix = sum(
        m.y[s,t,x]*m.MW[s]
        for s in m.S
    )

    return (
        m.rho_g[t,x]*Ru*m.T[t,x]
        ==
        MWmix*m.P[t,x]
    )

m.gas_density = Constraint(
    m.t, m.x,
    rule=gas_density_rule
)


# # ============================================================
# # MOLE FRACTION CLOSURE
# # ============================================================

# def mole_fraction_sum_rule(m,t,x):

#     return sum(
#         m.y[s,t,x]
#         for s in m.S
#     ) == 1.0

# m.mole_fraction_sum = Constraint(
#     m.t, m.x,
#     rule=mole_fraction_sum_rule
# )


# ============================================================
# PARTIAL PRESSURES
# bar
# ============================================================

def partial_pressure_rule(m,s,t,x):

    return (
        m.Pi[s,t,x]
        ==
        m.y[s,t,x]*m.P[t,x]/1.0e5
    )

m.partial_pressure = Constraint(
    m.S, m.t, m.x,
    rule=partial_pressure_rule
)


# ============================================================
# LANGMUIR-FREUNDLICH
# ============================================================

def isotherm_rule(m,s,t,x):

    numerator = (
        m.qsat[s]
        *
        (m.b[s]*m.Pi[s,t,x])**m.n[s]
    )

    denominator = (
        1.0
        +
        sum(
            (m.b[k]*m.Pi[k,t,x])**m.n[k]
            for k in m.S
        )
    )

    return (
        m.qstar[s,t,x]*denominator
        ==
        numerator
    )

m.isotherm = Constraint(
    m.S, m.t, m.x,
    rule=isotherm_rule
)


# ============================================================
# LDF
#
# t is normalized:
#
# dq/dtau = t_phys * k * (qstar-q)
# ============================================================

def ldf_rule(m,s,t,x):

    if t == m.t.first():
        return Constraint.Skip

    return (
        m.dqdt[s,t,x]
        ==
        t_phys*m.kLDF[s]
        *(m.qstar[s,t,x] - m.q[s,t,x])
    )

m.ldf = Constraint(
    m.S, m.t, m.x,
    rule=ldf_rule
)


# ============================================================
# COMPONENT MASS BALANCE
#
# species balance - yi * overall balance
#
# P/tphys * dy/dtau
#
# + P*u/(eps*L) * dy/dx
#
# + rho_b*R*T/(eps*tphys)
#       * (dqi/dtau - yi*sum(dqj/dtau))
#
# = 0
# ============================================================

def component_mass_balance_rule(m,s,t,x):

    if t == m.t.first():
        return Constraint.Skip

    if x == m.x.first():
        return Constraint.Skip

    accumulation = (
        m.P[t,x]
        * m.dydt[s,t,x]
        / t_phys
    )

    convection = (
        m.P[t,x]
        * m.u[t,x]
        * m.dydx[s,t,x]
        / (m.epsilon*L)
    )

    adsorption = (
        m.rho_b
        * Ru
        * m.T[t,x]
        / (m.epsilon*t_phys)
        *
        (
            m.dqdt[s,t,x]
            -
            m.y[s,t,x]
            * sum(
                m.dqdt[k,t,x]
                for k in m.S
            )
        )
    )

    return (
        accumulation
        + convection
        + adsorption
        == 0.0
    )

m.component_mass_balance = Constraint(
    m.S, m.t, m.x,
    rule=component_mass_balance_rule
)


# ============================================================
# OVERALL MASS / VELOCITY BALANCE
#
# dP/dt
#
# - P/T dT/dt
#
# + u/(eps) dP/dz
#
# + P/(eps) du/dz
#
# - uP/(eps T) dT/dz
#
# + rho_b R T / eps * sum(dq/dt)
#
# = 0
# ============================================================

def overall_mass_balance_rule(m,t,x):

    if t == m.t.first():
        return Constraint.Skip

    if x == m.x.first():
        return Constraint.Skip

    pressure_accumulation = (
        m.dPdt[t,x]
        / t_phys
    )

    temperature_accumulation = (
        -
        m.P[t,x]
        / m.T[t,x]
        * m.dTdt[t,x]
        / t_phys
    )

    pressure_convection = (
        m.u[t,x]
        / (m.epsilon*L)
        * m.dPdx[t,x]
    )

    velocity_term = (
        m.P[t,x]
        / (m.epsilon*L)
        * m.dudx[t,x]
    )

    temperature_convection = (
        -
        m.u[t,x]
        * m.P[t,x]
        / (
            m.epsilon
            * m.T[t,x]
            * L
        )
        * m.dTdx[t,x]
    )

    adsorption = (
        m.rho_b
        * Ru
        * m.T[t,x]
        / (m.epsilon*t_phys)
        *
        sum(
            m.dqdt[s,t,x]
            for s in m.S
        )
    )

    return (
        pressure_accumulation
        + temperature_accumulation
        + pressure_convection
        + velocity_term
        + temperature_convection
        + adsorption
        == 0.0
    )

m.overall_mass_balance = Constraint(
    m.t, m.x,
    rule=overall_mass_balance_rule
)


# ============================================================
# ERGUN PRESSURE DROP
#
# x is normalized, so:
#
# -dP/dz = -(1/L)dP/dx
# ============================================================

def ergun_rule(m,t,x):

    if x == m.x.first():
        return Constraint.Skip

    if t == m.t.first():
        return Constraint.Skip

    viscous = (
        150.0
        * m.mu
        * (1.0-m.epsilon)**2
        /
        (m.dp**2*m.epsilon**3)
        * m.u[t,x]
    )

    inertial = (
        1.75
        * (1.0-m.epsilon)
        /
        (m.dp*m.epsilon**3)
        * m.rho_g[t,x]
        * m.u[t,x]**2
    )

    return (
        -m.dPdx[t,x]/L
        ==
        viscous + inertial
    )

m.ergun = Constraint(
    m.t, m.x,
    rule=ergun_rule
)


# ============================================================
# ENERGY BALANCE
#
# simplified one-temperature packed-bed balance
#
# [eps*Cgas*Cp_g + rho_b*Cp_s] dT/dt
# + Cgas*Cp_g*u dT/dz
# - keff d2T/dz2
# - rho_b sum(dHads*dq/dt)
# = 0
#
# For the first test, omit axial conduction.
# Add it after the basic model converges.
# ============================================================

def energy_balance_rule(m,t,x):

    if t == m.t.first():
        return Constraint.Skip

    if x == m.x.first():
        return Constraint.Skip

    Ctot = m.P[t,x]/(Ru*m.T[t,x])

    heat_capacity = (
        m.epsilon*Ctot*m.Cp_g
        + m.rho_b*m.Cp_s
    )

    accumulation = (
        heat_capacity
        * m.dTdt[t,x]
        / t_phys
    )

    convection = (
        Ctot
        * m.Cp_g
        * m.u[t,x]
        * m.dTdx[t,x]
        / L
    )

    adsorption_heat = (
        m.rho_b
        * sum(
            m.dHads[s]
            * m.dqdt[s,t,x]
            / t_phys
            for s in m.S
        )
    )

    return (
        1.0e-5
        *
        (
            accumulation
            + convection
            - adsorption_heat
        )
        == 0.0
    )

m.energy_balance = Constraint(
    m.t, m.x,
    rule=energy_balance_rule
)


# ============================================================
# INITIAL CONDITIONS FROM FINAL ADSORPTION STATE
# ============================================================

def initial_y_rule(m, s, x):
    return (
        m.y[s, m.t.first(), x]
        ==
        y_ads_final(s, x)
    )

m.initial_y = Constraint(
    m.S,
    m.x,
    rule=initial_y_rule
)


def initial_P_rule(m, x):
    return (
        m.P[m.t.first(), x]
        ==
        P_ads_final(x)
    )

m.initial_P = Constraint(
    m.x,
    rule=initial_P_rule
)


def initial_T_rule(m, x):
    return (
        m.T[m.t.first(), x]
        ==
        T_ads_final(x)
    )

m.initial_T = Constraint(
    m.x,
    rule=initial_T_rule
)


def initial_q_rule(m, s, x):
    return (
        m.q[s, m.t.first(), x]
        ==
        q_ads_final(s, x)
    )

m.initial_q = Constraint(
    m.S,
    m.x,
    rule=initial_q_rule
)


# ============================================================
# INLET BOUNDARY CONDITIONS
# ============================================================

def inlet_y_rule(m,s,t):

    if t == m.t.first():
        return Constraint.Skip

    return (
        m.dydx[s,t,m.x.first()]
        ==
        0.0
    )

m.inlet_y = Constraint(
    m.S, m.t,
    rule=inlet_y_rule
)


def inlet_T_rule(m,t):

    if t == m.t.first():
        return Constraint.Skip

    return (
        m.dTdx[t,m.x.first()]
        ==
        0.0
    )

m.inlet_T = Constraint(
    m.t,
    rule=inlet_T_rule
)

# ============================================================
# CLOSED INLET
# ============================================================

def inlet_velocity_rule(m, t):

    return (
        m.u[t, m.x.first()]
        ==
        0.0
    )

m.inlet_velocity = Constraint(
    m.t,
    rule=inlet_velocity_rule
)


# ============================================================
# DESORPTION OUTLET PRESSURE
# ============================================================

def outlet_pressure_rule(m, t):

    if t == m.t.first():
        return Constraint.Skip

    return (
        m.P[t, m.x.last()]
        ==
        P_des
    )

m.outlet_pressure = Constraint(
    m.t,
    rule=outlet_pressure_rule
)


# ============================================================
# DISCRETIZATION
# ============================================================

time_discretizer = TransformationFactory(
    'dae.collocation'
)

space_discretizer = TransformationFactory(
    'dae.finite_difference'
)

time_discretizer.apply_to(
    m,
    wrt=m.t,
    nfe=Nfet,
    ncp=NCP,
    scheme='LAGRANGE-RADAU'
)

space_discretizer.apply_to(
    m,
    wrt=m.x,
    nfe=Nfex,
    scheme='BACKWARD'
)


# ============================================================
# INITIALIZE AFTER DISCRETIZATION
# ============================================================

Pdrop_guess = 3000.0       # Pa across bed; only an initial guess
Tout_rise_guess = 8.0      # K; only an initial guess


# ============================================================
# INITIALIZE DESORPTION TRAJECTORY
# ============================================================

for t in m.t:

    tau = float(t)

    for x in m.x:

        xx = float(x)

        # ----------------------------------------------------
        # Initial adsorption-end values
        # ----------------------------------------------------

        P0 = P_ads_final(x)
        T0 = T_ads_final(x)

        # ----------------------------------------------------
        # PRESSURE GUESS
        #
        # At tau = 0:
        #     exact adsorption-end pressure
        #
        # At tau > 0:
        #     outlet fixed near P_des
        #     closed end initially remains at higher pressure
        # ----------------------------------------------------

        if tau == 0.0:

            Pguess = P0

        else:

            P_closed_guess = (
                P_des
                +
                (P_ads_final(0.0) - P_des)
                * np.exp(-5.0*tau)
            )

            Pguess = (
                P_des
                +
                (P_closed_guess - P_des)
                * (1.0 - xx)
            )

        m.P[t,x].set_value(Pguess)

        # ----------------------------------------------------
        # TEMPERATURE GUESS
        #
        # Desorption cools the bed slightly.
        # ----------------------------------------------------

        Tguess = (
            T0
            - 10.0
            * (1.0 - np.exp(-4.0*tau))
        )

        m.T[t,x].set_value(Tguess)

        # ----------------------------------------------------
        # VELOCITY GUESS
        #
        # u = 0 at closed end
        # velocity increases toward vacuum end
        # ----------------------------------------------------

        umax_guess = (
            3.0
            * np.exp(-3.0*tau)
            + 0.05
        )

        uguess = (
            umax_guess
            * xx
        )

        m.u[t,x].set_value(uguess)

        # ----------------------------------------------------
        # GAS COMPOSITION
        #
        # Start from adsorption-end composition.
        # ----------------------------------------------------

        yguess = {}

        for s in components:

            yguess[s] = max(
                y_ads_final(s,x),
                1.0e-10
            )

            m.y[s,t,x].set_value(
                yguess[s]
            )

        # ----------------------------------------------------
        # PARTIAL PRESSURES
        # ----------------------------------------------------

        for s in components:

            Pi_guess = max(
                yguess[s]
                * Pguess
                / 1.0e5,
                1.0e-10
            )

            m.Pi[s,t,x].set_value(
                Pi_guess
            )

        # ----------------------------------------------------
        # EQUILIBRIUM LOADING
        # ----------------------------------------------------

        denominator = (
            1.0
            +
            sum(
                (
                    b_data[k]
                    * max(
                        yguess[k]
                        * Pguess
                        / 1.0e5,
                        1.0e-10
                    )
                )**n_data[k]
                for k in components
            )
        )

        for s in components:

            qstar_guess = (
                qsat_data[s]
                *
                (
                    b_data[s]
                    * max(
                        yguess[s]
                        * Pguess
                        / 1.0e5,
                        1.0e-10
                    )
                )**n_data[s]
                /
                denominator
            )

            m.qstar[s,t,x].set_value(
                qstar_guess
            )

            # -----------------------------------------------
            # LOADING GUESS
            #
            # Begin from adsorption-end loading and relax
            # toward low-pressure equilibrium.
            # -----------------------------------------------

            q0 = q_ads_final(s,x)

            qguess = (
                qstar_guess
                +
                (q0 - qstar_guess)
                * np.exp(
                    -kLDF_data[s]
                    * t_phys
                    * tau
                )
            )

            m.q[s,t,x].set_value(
                max(qguess, 0.0)
            )

        # ----------------------------------------------------
        # GAS DENSITY
        # ----------------------------------------------------

        MWguess = sum(
            yguess[s]*MW_data[s]
            for s in components
        )

        rhoguess = (
            Pguess
            * MWguess
            /
            (Ru*Tguess)
        )

        m.rho_g[t,x].set_value(
            max(rhoguess, 1e-8)
        )

# ============================================================
# EXACT INITIAL STATE FROM ADSORPTION
# ============================================================

t0 = m.t.first()

for x in m.x:

    m.P[t0,x].set_value(
        P_ads_final(x)
    )

    m.T[t0,x].set_value(
        T_ads_final(x)
    )

    # Velocity is algebraic.
    # Desorption inlet is closed.
    m.u[t0,x].set_value(
        max(
            0.0,
            u_ads_final(x)
        )
    )

    for s in components:

        m.y[s,t0,x].set_value(
            y_ads_final(s,x)
        )

        m.q[s,t0,x].set_value(
            q_ads_final(s,x)
        )

        m.qstar[s,t0,x].set_value(
            qstar_ads_final(s,x)
        )

        m.Pi[s,t0,x].set_value(
            max(
                y_ads_final(s,x)
                * P_ads_final(x)
                / 1e5,
                1e-10
            )
        )

    m.rho_g[t0,x].set_value(
        rho_ads_final(x)
    )
# ============================================================
# SOLVER
# ============================================================

solver = SolverFactory('ipopt')

solver.options['tol'] = 1.0e-6
# solver.options['acceptable_tol'] = 1.0e-5
# solver.options['acceptable_iter'] = 10

solver.options['max_iter'] = 5000

# solver.options['mu_strategy'] = 'adaptive'

# solver.options['nlp_scaling_method'] = 'gradient-based'

solver.options['linear_solver'] = 'ma97'

# solver.options['ma97_scaling'] = 'dynamic'


results = solver.solve(
    m,
    tee=True
)

# ============================================================
# POST-PROCESSING AND STATE SAVE
# ============================================================

import os
import pickle
import pandas as pd

from pyomo.opt import TerminationCondition


# ------------------------------------------------------------
# Check solution
# ------------------------------------------------------------

tc = results.solver.termination_condition

print("\nTermination condition:", tc)

if tc not in [
    TerminationCondition.optimal,
    TerminationCondition.locallyOptimal,
]:
    print("WARNING: Solver did not report an optimal solution.")


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

output_dir = "breakthrough_results"
os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Discretized coordinates
# ------------------------------------------------------------

t_points = sorted(list(m.t))
x_points = sorted(list(m.x))

time_sec = np.array([
    float(t)*t_phys
    for t in t_points
])

z_m = np.array([
    float(x)*L
    for x in x_points
])

t_out = t_points
x_in = m.x.first()
x_out = m.x.last()

# ============================================================
# OUTLET MOLE FRACTIONS
# ============================================================

plt.figure(figsize=(8, 5))

for s in components:

    yout = np.array([
        value(m.y[s, t, x_out])
        for t in t_points
    ])

    plt.plot(
        time_sec,
        yout,
        marker='o',
        markersize=3,
        label=s
    )

plt.xlabel("Time [s]")
plt.ylabel("Outlet mole fraction [-]")
plt.title("Breakthrough curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "breakthrough_mole_fraction.png"),
    dpi=300
)

# plt.show()

plt.figure(figsize=(8, 5))

yout_NH3 = np.array([
    value(m.y['NH3', t, x_out])
    for t in t_points
])

plt.plot(
    time_sec,
    yout_NH3,
    marker='o',
    markersize=3
)

plt.axhline(
    y_feed['NH3'],
    linestyle='--',
    label="Feed"
)

plt.xlabel("Time [s]")
plt.ylabel("NH3 mole fraction [-]")
plt.title("NH3 breakthrough")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "NH3_breakthrough.png"),
    dpi=300
)

# plt.show()

# ============================================================
# PRESSURE
# ============================================================

Pin = np.array([
    value(m.P[t, x_in])/1e5
    for t in t_points
])

Pout = np.array([
    value(m.P[t, x_out])/1e5
    for t in t_points
])

plt.figure(figsize=(8, 5))

plt.plot(
    time_sec,
    Pin,
    marker='o',
    markersize=3,
    label="Inlet"
)

plt.plot(
    time_sec,
    Pout,
    marker='s',
    markersize=3,
    label="Outlet"
)

plt.xlabel("Time [s]")
plt.ylabel("Pressure [bar]")
plt.title("Bed pressure")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "pressure_vs_time.png"),
    dpi=300
)

# plt.show()

plt.figure(figsize=(8, 5))

deltaP = (Pin - Pout)*1e5

plt.plot(
    time_sec,
    deltaP,
    marker='o',
    markersize=3
)

plt.xlabel("Time [s]")
plt.ylabel("Pressure drop [Pa]")
plt.title("Bed pressure drop")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "pressure_drop.png"),
    dpi=300
)

# plt.show()

# ============================================================
# TEMPERATURE
# ============================================================

Tin = np.array([
    value(m.T[t, x_in])
    for t in t_points
])

Tout = np.array([
    value(m.T[t, x_out])
    for t in t_points
])

plt.figure(figsize=(8, 5))

plt.plot(
    time_sec,
    Tin,
    label="Inlet"
)

plt.plot(
    time_sec,
    Tout,
    label="Outlet"
)

plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.title("Bed temperature")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "temperature_vs_time.png"),
    dpi=300
)

# plt.show()

# ============================================================
# TEMPERATURE PROFILES
# ============================================================

sample_times = [0, 10, 20, 30, 40, 50]

plt.figure(figsize=(8, 5))

for target_time in sample_times:

    tau_target = target_time/t_phys

    t_nearest = min(
        t_points,
        key=lambda t: abs(float(t) - tau_target)
    )

    Tprofile = np.array([
        value(m.T[t_nearest, x])
        for x in x_points
    ])

    plt.plot(
        z_m,
        Tprofile,
        label=f"{float(t_nearest)*t_phys:.1f} s"
    )

plt.xlabel("Axial position [m]")
plt.ylabel("Temperature [K]")
plt.title("Temperature profiles")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "temperature_profiles.png"),
    dpi=300
)

# plt.show()

# ============================================================
# NH3 LOADING PROFILES
# ============================================================

plt.figure(figsize=(8, 5))

for target_time in sample_times:

    tau_target = target_time/t_phys

    t_nearest = min(
        t_points,
        key=lambda t: abs(float(t) - tau_target)
    )

    qprofile = np.array([
        value(m.q['NH3', t_nearest, x])
        for x in x_points
    ])

    plt.plot(
        z_m,
        qprofile,
        label=f"{float(t_nearest)*t_phys:.1f} s"
    )

plt.xlabel("Axial position [m]")
plt.ylabel("NH3 loading [mol/kg]")
plt.title("NH3 adsorption profiles")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "NH3_loading_profiles.png"),
    dpi=300
)

# plt.show()

# ============================================================
# SAVE DESORPTION STATE
# ============================================================

rows = []

for t in sorted(m.t):

    physical_time = float(t)*t_phys

    for x in sorted(m.x):

        row = {
            "tau": float(t),
            "time_s": physical_time,
            "x": float(x),
            "z_m": float(x)*L,

            "P_Pa": value(m.P[t,x]),
            "P_bar": value(m.P[t,x])/1e5,

            "T_K": value(m.T[t,x]),
            "u_m_s": value(m.u[t,x]),
            "rho_g_kg_m3": value(m.rho_g[t,x]),
        }

        for s in components:

            row[f"y_{s}"] = value(
                m.y[s,t,x]
            )

            row[f"q_{s}_mol_kg"] = value(
                m.q[s,t,x]
            )

            row[f"qstar_{s}_mol_kg"] = value(
                m.qstar[s,t,x]
            )

            row[f"Pi_{s}_bar"] = value(
                m.Pi[s,t,x]
            )

        row["sum_y"] = sum(
            value(m.y[s,t,x])
            for s in components
        )

        rows.append(row)


df_des = pd.DataFrame(rows)

df_des.to_csv(
    "desorption_state.csv",
    index=False
)

print("Saved:", "desorption_state.csv")

print(
    "\nMole fraction sum range:",
    df_des["sum_y"].min(),
    df_des["sum_y"].max()
)

print(
    "Maximum |sum(y)-1|:",
    np.max(
        np.abs(df_des["sum_y"] - 1.0)
    )
)