import math
from pyomo.environ import *
from pyomo.dae import *

model = ConcreteModel()

# --------------------------------------------------
# Domains
# --------------------------------------------------

L = 1.0
t_final = 1000.0

model.z = ContinuousSet(bounds=(0.0, L))
model.t = ContinuousSet(bounds=(0.0, 1))

specs = ['N2', 'H2', 'NH3']
model.S = Set(initialize=specs)

# --------------------------------------------------
# Parameters
# --------------------------------------------------

R = 8.31446261815324
# T = 700.0
P = 50e5
T_in = 700.0       # K
T_ref = 700.0      # reference temperature for Keq

model.Ru = Param(initialize=R)
# model.T = Param(initialize=T)
model.Tin = Param(initialize=T_in)
model.Tref = Param(initialize=T_ref)
model.P = Param(initialize=P)

model.epsilon = Param(initialize=0.45)
model.Dax = Param(initialize=1e-5)
model.u = Param(initialize=0.01)

model.k0 = Param(initialize=1.0)       # TEMPORARY sensible scaling
model.Ea = Param(initialize=80000.0)
model.Keq = Param(initialize=1.0)
model.Kads = Param(initialize=1.0)

# model.Ctot = Param(
#     initialize=P / (R*T)
# )

# Inlet total concentration
Ctot_in = P / (R * T_in)

model.Ctot = Param(
    initialize=Ctot_in
)

nu = {
    'N2': -1.0,
    'H2': -3.0,
    'NH3': 2.0
}

model.nu = Param(model.S, initialize=nu)

y_in = {
    'N2': 0.25,
    'H2': 0.75,
    'NH3': 0.0
}

# Inlet species concentrations [mol/m3]
C_in = {
    s: y_in[s] * Ctot_in
    for s in specs
}

model.Cin = Param(model.S, initialize=C_in)

# --------------------------------------------------
# Thermal parameters
# --------------------------------------------------

# Gas molar heat capacity
model.Cpg = Param(
    initialize=30.0        # J/mol/K
)

# Catalyst/solid density
model.rho_cat = Param(
    initialize=1500.0      # kg/m3 solid
)

# Catalyst/solid specific heat
model.Cp_cat = Param(
    initialize=1000.0      # J/kg/K
)

# Effective axial thermal conductivity
model.k_eff = Param(
    initialize=0.5         # W/m/K
)

# Heat of reaction
# N2 + 3H2 -> 2NH3
model.dHrxn = Param(
    initialize=-92000.0    # J/mol reaction
)

# --------------------------------------------------
# Variables
# --------------------------------------------------

model.C = Var(
    model.S,
    model.z,
    model.t,
    domain=NonNegativeReals,
    initialize=lambda m, s, z, t: value(m.Cin[s])
)

model.dCdt = DerivativeVar(
    model.C,
    wrt=model.t
)

model.dCdz = DerivativeVar(
    model.C,
    wrt=model.z
)

model.d2Cdz2 = DerivativeVar(
    model.C,
    wrt=(model.z, model.z)
)

model.Temp = Var(
    model.z,
    model.t,
    bounds=(300.0, 1200.0),
    initialize=T_in
)

model.dTdt = DerivativeVar(
    model.Temp,
    wrt=model.t
)

model.dTdz = DerivativeVar(
    model.Temp,
    wrt=model.z
)

model.d2Tdz2 = DerivativeVar(
    model.dTdz,
    wrt=model.z
)

# --------------------------------------------------
# Reaction rate
# --------------------------------------------------

def reaction_rate(m, z, t):

    Tlocal = m.Temp[z, t]

    # partial pressures
    pN2 = m.C['N2', z, t] * m.Ru * Tlocal
    pH2 = m.C['H2', z, t] * m.Ru * Tlocal
    pNH3 = m.C['NH3', z, t] * m.Ru * Tlocal

    # dimensionless activities based on 1 bar
    pref = 1e5

    aN2 = pN2 / pref
    aH2 = pH2 / pref
    aNH3 = pNH3 / pref

    k = m.k0 * exp(
        -m.Ea/(m.Ru*Tlocal)
    )

    # ----------------------------------------------
    # Temperature-dependent equilibrium constant
    #
    # ln(K/Kref) =
    # -dH/R * (1/T - 1/Tref)
    # ----------------------------------------------

    Keq_T = m.Keq * exp(
        -m.dHrxn / m.Ru
        * (
            1.0/Tlocal
            - 1.0/m.Tref
        )
    )

    # ----------------------------------------------
    # Reaction driving force
    # ----------------------------------------------

    driving_force = (
        aN2 * aH2**3
        - aNH3**2 / Keq_T
    )

    denominator = (
        1.0 + m.Kads*aNH3
    )**2

    return (
        k
        * driving_force
        / denominator
    )


# --------------------------------------------------
# PDE
# --------------------------------------------------

def mass_balance_rule(m, s, z, t):

    # Initial condition handles t = 0
    if t == m.t.first():
        return Constraint.Skip

    # Boundary conditions handle z = 0 and L
    if z == m.z.first() or z == m.z.last():
        return Constraint.Skip

    source = (
        (1.0 - m.epsilon)
        / m.epsilon
        * m.nu[s]
        * reaction_rate(m, z, t)
    )

    return (
        m.dCdt[s,z,t] / t_final
        + m.u*m.dCdz[s,z,t]
        - m.Dax*m.d2Cdz2[s,z,t]
        == source
    )

model.mass_balance = Constraint(
    model.S,
    model.z,
    model.t,
    rule=mass_balance_rule
)

# --------------------------------------------------
# Energy balance
# --------------------------------------------------

def energy_balance_rule(m, z, t):

    # Initial condition handles tau = 0
    if t == m.t.first():
        return Constraint.Skip

    # Boundary conditions handle z = 0 and z = L
    if z == m.z.first() or z == m.z.last():
        return Constraint.Skip

    Tlocal = m.Temp[z, t]

    # Local total gas concentration [mol/m3]
    Ctot_local = (
        m.P / (m.Ru*Tlocal)
    )

    # Gas volumetric heat capacity [J/m3/K]
    rhoCp_gas = (
        Ctot_local * m.Cpg
    )

    # Effective bed volumetric heat capacity [J/m3-bed/K]
    rhoCp_eff = (
        m.epsilon * rhoCp_gas
        +
        (1.0 - m.epsilon)
        * m.rho_cat
        * m.Cp_cat
    )

    # Reaction heat generation [W/m3-bed]
    Qreaction = (
        -(1.0 - m.epsilon)
        * m.dHrxn
        * reaction_rate(m, z, t)
    )

    return (
        rhoCp_eff
        * m.dTdt[z, t]
        / t_final

        + rhoCp_gas
        * m.u
        * m.dTdz[z, t]

        - m.k_eff
        * m.d2Tdz2[z, t]

        == Qreaction
    )


model.energy_balance = Constraint(
    model.z,
    model.t,
    rule=energy_balance_rule
)

# --------------------------------------------------
# Initial condition
# --------------------------------------------------

def initial_condition_rule(m, s, z):

    if z == m.z.first():
        return Constraint.Skip

    return (
        m.C[s,z,m.t.first()]
        == m.Cin[s]
    )

model.initial_condition = Constraint(
    model.S,
    model.z,
    rule=initial_condition_rule
)

# --------------------------------------------------
# Temperature initial condition
# --------------------------------------------------

def initial_temperature_rule(m, z):

    if z == m.z.first():
        return Constraint.Skip

    return (
        m.Temp[z, m.t.first()]
        == m.Tin
    )


model.initial_temperature = Constraint(
    model.z,
    rule=initial_temperature_rule
)

# --------------------------------------------------
# Inlet BC
# --------------------------------------------------

def inlet_rule(m, s, t):

    return (
        m.C[s,m.z.first(),t]
        == m.Cin[s]
    )

model.inlet_bc = Constraint(
    model.S,
    model.t,
    rule=inlet_rule
)

# --------------------------------------------------
# Temperature inlet BC
# --------------------------------------------------

def temperature_inlet_rule(m, t):

    return (
        m.Temp[m.z.first(), t]
        == m.Tin
    )


model.temperature_inlet = Constraint(
    model.t,
    rule=temperature_inlet_rule
)

# --------------------------------------------------
# Temperature outlet BC
# --------------------------------------------------

def temperature_outlet_rule(m, t):

    if t == m.t.first():
        return Constraint.Skip

    return (
        m.dTdz[m.z.last(), t]
        == 0.0
    )


model.temperature_outlet = Constraint(
    model.t,
    rule=temperature_outlet_rule
)

# --------------------------------------------------
# Outlet BC
# dC/dz = 0
# --------------------------------------------------

def outlet_rule(m, s, t):

    if t == m.t.first():
        return Constraint.Skip

    return (
        m.dCdz[s,m.z.last(),t]
        == 0
    )

model.outlet_bc = Constraint(
    model.S,
    model.t,
    rule=outlet_rule
)

# --------------------------------------------------
# Discretization
# --------------------------------------------------

n_z = 50
n_t = 40

TransformationFactory(
    'dae.finite_difference'
).apply_to(
    model,
    wrt=model.z,
    nfe=n_z-1,
    scheme='BACKWARD'
)

TransformationFactory(
    'dae.finite_difference'
).apply_to(
    model,
    wrt=model.t,
    nfe=n_t-1,
    scheme='BACKWARD'
)

# --------------------------------------------------
# Solve
# --------------------------------------------------

solver = SolverFactory('ipopt')

solver.options['max_iter'] = 3000
solver.options['tol'] = 1e-7

# Start with MUMPS unless MA97 installation is verified
solver.options['linear_solver'] = 'ma97'

results = solver.solve(
    model,
    tee=True
)

print(results.solver.termination_condition)

# --------------------------------------------------
# Results
# --------------------------------------------------

zL = model.z.last()

print(
    "Time, y_N2_out, y_H2_out, y_NH3_out"
)

for t in model.t:

    CN2 = value(model.C['N2',zL,t])
    CH2 = value(model.C['H2',zL,t])
    CNH3 = value(model.C['NH3',zL,t])

    Csum = CN2 + CH2 + CNH3

    yN2 = CN2/Csum
    yH2 = CH2/Csum
    yNH3 = CNH3/Csum

    print(
        f"{float(t):8.4f}, "
        f"{yN2:10.6f}, "
        f"{yH2:10.6f}, "
        f"{yNH3:10.6f}"
    )

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

# Physical times [s]
times_to_plot = np.linspace(0.0, t_final, 61)

z_values = np.array([float(z) for z in model.z])

# Single-color sequential colormap:
# early time = light, latest time = dark
cmap = colormaps['Blues']

# Avoid the almost-white very beginning of the colormap
colors = cmap(np.linspace(0.25, 1.0, len(times_to_plot)))

for s in model.S:

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, t_target in enumerate(times_to_plot):

        # Physical time -> normalized time
        tau_target = t_target / t_final

        # Closest discretized normalized time
        tau_plot = min(
            model.t,
            key=lambda tau: abs(float(tau) - tau_target)
        )

        # Actual physical time corresponding to discretized point
        physical_time = float(tau_plot) * t_final

        # Mole-fraction profile
        y_profile = np.array([
            value(model.C[s, z, tau_plot]) /
            sum(
                value(model.C[sp, z, tau_plot])
                for sp in model.S
            )
            for z in model.z
        ])

        ax.plot(
            z_values,
            y_profile,
            color=colors[i],
            marker='o',
            markersize=3,
            linewidth=1.5,
            label=f'{physical_time:.1f} s'
        )

    ax.set_xlabel('Axial position, z [m]')
    ax.set_ylabel('Mole fraction [-]')
    ax.set_title(f'{s} mole fraction profile')

    ax.legend(
        title='Time',
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )

    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Plot temperature profiles
# --------------------------------------------------

times_to_plot = np.linspace(
    0.0,
    t_final,
    61
)

z_values = np.array([
    float(z)
    for z in model.z
])

cmap = colormaps['Reds']

# earliest = light
# latest = dark
colors = cmap(
    np.linspace(
        0.25,
        1.0,
        len(times_to_plot)
    )
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

for i, physical_time_target in enumerate(times_to_plot):

    # Convert physical time to normalized time
    tau_target = (
        physical_time_target
        / t_final
    )

    # Closest discretized point
    tau_plot = min(
        model.t,
        key=lambda tau:
            abs(float(tau) - tau_target)
    )

    physical_time = (
        float(tau_plot)
        * t_final
    )

    T_profile = np.array([
        value(
            model.Temp[z, tau_plot]
        )
        for z in model.z
    ])

    ax.plot(
        z_values,
        T_profile,
        color=colors[i],
        marker='o',
        markersize=3,
        linewidth=1.5,
        label=f'{physical_time:.1f} s'
    )

ax.set_xlabel(
    'Axial position, z [m]'
)

ax.set_ylabel(
    'Temperature [K]'
)

ax.set_title(
    'Bed temperature profile'
)

ax.legend(
    title='Time',
    bbox_to_anchor=(1.02, 1),
    loc='upper left'
)

ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()