import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import *
from pyomo.dae import ContinuousSet, DerivativeVar


# ============================================================
# BASIC CONDITIONS
# ============================================================

Ru = 8.314462618                   # J/mol/K

T = 473.15                         # K
P = 30.0e5                         # Pa

L = 1.0                            # m
epsilon = 0.40                     # bed void fraction
rho_b = 700.0                      # kg adsorbent/m3 bed
Dax = 1.0e-5                       # m2/s

u_in = 0.1                        # m/s

t_final = 500.0                    # s
Nz = 40
Nt = 100

dz = L/Nz

Ctot = P/(Ru*T)                    # mol/m3


# ============================================================
# COMPONENTS
# ============================================================

components = ['N2', 'H2', 'NH3']


# Feed composition
y_feed = {
    'N2': 0.25,
    'H2': 0.70,
    'NH3': 0.05
}


# Initial gas in the bed
# For a breakthrough experiment, this can be a weakly adsorbing
# purge gas / previous-feed composition.
y_initial = {
    'N2': 0.9998,
    'H2': 0.0001,
    'NH3': 0.0001
}


# ============================================================
# LANGMUIR-FREUNDLICH PARAMETERS
# ============================================================
#
# These are only PLACEHOLDER values.
# Replace with your experimental isotherm parameters.
#
# qsat : mol/kg
# b    : 1/bar
# n    : dimensionless
#

qsat_data = {
    'N2': 1.0,
    'H2': 0.5,
    'NH3': 5.0
}

b_data = {
    'N2': 0.01,
    'H2': 0.005,
    'NH3': 1.0
}

n_data = {
    'N2': 1.0,
    'H2': 1.0,
    'NH3': 0.8
}


# ============================================================
# LDF PARAMETERS
# ============================================================
#
# kLDF : 1/s
#

kLDF_data = {
    'N2': 0.05,
    'H2': 0.05,
    'NH3': 0.2
}


# ============================================================
# MODEL
# ============================================================

model = ConcreteModel()

model.S = Set(initialize=components)

# Cell indices: 1,...,Nz
model.J = RangeSet(1, Nz)

# Face indices: 0,...,Nz
model.F = RangeSet(0, Nz)

model.t = ContinuousSet(bounds=(0.0, 1))


# ============================================================
# PARAMETERS
# ============================================================

model.epsilon = Param(initialize=epsilon)
model.rho_b = Param(initialize=rho_b)
model.Dax = Param(initialize=Dax)
model.Ctot = Param(initialize=Ctot)
model.P = Param(initialize=P)
model.T = Param(initialize=T)
model.u_in = Param(initialize=u_in)

model.qsat = Param(model.S, initialize=qsat_data)
model.b = Param(model.S, initialize=b_data)
model.n = Param(model.S, initialize=n_data)
model.kLDF = Param(model.S, initialize=kLDF_data)


# ============================================================
# VARIABLES
# ============================================================

# Gas concentration [mol/m3]
model.C = Var(model.S, model.J, model.t, bounds=(1e-10, None), initialize=lambda m,s,j,t: y_initial[s]*Ctot)

# Adsorbed loading [mol/kg]
model.q = Var(model.S, model.J, model.t, bounds=(0.0, None), initialize=0.0)

# Gas mole fraction
model.y = Var(model.S, model.J, model.t, bounds=(1e-12, 1.0), initialize=lambda m,s,j,t: y_initial[s])

# Superficial velocity at cell faces [m/s]
model.u = Var(model.F, model.t, bounds=(1e-8, None), initialize=u_in)

# Component molar flux at cell faces [mol/m2/s]
model.N = Var(model.S, model.F, model.t, initialize=lambda m,s,f,t: u_in*y_initial[s]*Ctot)


# Time derivatives
model.dCdt = DerivativeVar(model.C, wrt=model.t)
model.dqdt = DerivativeVar(model.q, wrt=model.t)


# ============================================================
# GAS COMPOSITION
# ============================================================

def total_concentration_rule(m,j,t):
    return sum(m.C[s,j,t] for s in m.S) == m.Ctot

model.total_concentration = Constraint(model.J, model.t, rule=total_concentration_rule)


def mole_fraction_rule(m,s,j,t):
    return m.y[s,j,t]*m.Ctot == m.C[s,j,t]

model.mole_fraction = Constraint(model.S, model.J, model.t, rule=mole_fraction_rule)


# ============================================================
# PARTIAL PRESSURE
# ============================================================

model.Pi = Var(model.S, model.J, model.t, bounds=(1e-6, None), initialize=lambda m,s,j,t: max(y_initial[s]*P/1e5, 1e-6))
def partial_pressure_rule(m,s,j,t):
    return m.Pi[s,j,t] == m.y[s,j,t]*m.P/1e5

model.partial_pressure = Constraint(model.S, model.J, model.t, rule=partial_pressure_rule)

# ============================================================
# LANGMUIR-FREUNDLICH ISOTHERM
# ============================================================

p_eps = 1e-12

model.qstar = Var(model.S, model.J, model.t, bounds=(0.0, None), initialize=0.1)
def isotherm_rule(m,s,j,t):
    numerator = m.qsat[s]*(m.b[s]*m.Pi[s,j,t])**m.n[s]
    denominator = 1.0 + sum((m.b[k]*m.Pi[k,j,t])**m.n[k] for k in m.S)
    return m.qstar[s,j,t]*denominator == numerator

model.isotherm = Constraint(model.S, model.J, model.t, rule=isotherm_rule)

# ============================================================
# LDF KINETICS
# ============================================================

def ldf_rule(m,s,j,t):
    if t == m.t.first():
        return Constraint.Skip
    return m.dqdt[s,j,t] == m.kLDF[s]*(m.qstar[s,j,t] - m.q[s,j,t]) * t_final

model.ldf = Constraint(model.S, model.J, model.t, rule=ldf_rule)

# ============================================================
# COMPONENT FLUXES
# ============================================================
#
# Face 0      : prescribed feed molar flux
# Faces 1..Nz-1: upwind convection + central diffusion
# Face Nz     : zero diffusive flux
#
# Flow direction is assumed positive.
# ============================================================

def flux_rule(m,s,f,t):

    # --------------------------------------------------------
    # Inlet
    # --------------------------------------------------------

    if f == 0:
        return m.N[s,f,t] == m.u_in*m.Ctot*y_feed[s]

    # --------------------------------------------------------
    # Outlet
    # --------------------------------------------------------

    elif f == Nz:
        return m.N[s,f,t] == m.u[f,t]*m.C[s,Nz,t]

    # --------------------------------------------------------
    # Interior faces
    # --------------------------------------------------------

    else:

        C_left = m.C[s,f,t]
        C_right = m.C[s,f+1,t]

        convection = m.u[f,t]*C_left

        diffusion = -m.epsilon*m.Dax*(C_right - C_left)/dz

        return m.N[s,f,t] == convection + diffusion

model.component_flux = Constraint(model.S, model.F, model.t, rule=flux_rule)


# ============================================================
# COMPONENT MASS BALANCE
# ============================================================

def component_balance_rule(m,s,j,t):

    if t == m.t.first():
        return Constraint.Skip

    accumulation = m.epsilon*m.dCdt[s,j,t]/t_final

    flux_divergence = (m.N[s,j,t] - m.N[s,j-1,t])/dz

    adsorption = m.rho_b*m.dqdt[s,j,t]/t_final

    return accumulation + flux_divergence + adsorption == 0.0

model.component_balance = Constraint(model.S, model.J, model.t, rule=component_balance_rule)


# ============================================================
# INLET VELOCITY
# ============================================================

def inlet_velocity_rule(m,t):
    return m.u[0,t] == m.u_in

model.inlet_velocity = Constraint(model.t, rule=inlet_velocity_rule)


# ============================================================
# INITIAL GAS COMPOSITION
# ============================================================

def initial_concentration_rule(m,s,j):
    return m.C[s,j,m.t.first()] == y_initial[s]*m.Ctot

model.initial_concentration = Constraint(model.S, model.J, rule=initial_concentration_rule)


# ============================================================
# INITIAL ADSORBED LOADING
# ============================================================
#
# Bed initially at equilibrium with y_initial.
# ============================================================

def initial_loading_rule(m,s,j):

    t0 = m.t.first()

    Pbar = P/1e5

    numerator = qsat_data[s]*(b_data[s]*(y_initial[s]*Pbar) + p_eps)**n_data[s]

    denominator = 1.0 + sum((b_data[k]*(y_initial[k]*Pbar) + p_eps)**n_data[k] for k in components)

    q0 = numerator/denominator

    return m.q[s,j,t0] == q0

model.initial_loading = Constraint(model.S, model.J, rule=initial_loading_rule)


# ============================================================
# TIME DISCRETIZATION
# ============================================================

TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=Nt, scheme='BACKWARD')


# ============================================================
# SOLVER
# ============================================================

solver = SolverFactory('ipopt')

solver.options['tol'] = 1e-7
solver.options['max_iter'] = 3000
solver.options['linear_solver'] = 'ma97'

results = solver.solve(model, tee=True)


# ============================================================
# FINAL OUTLET CONDITIONS
# ============================================================

tf = model.t.last()

print()
print("Final outlet conditions")
print("-----------------------")
print(f"Time       = {float(tf)*t_final:.4f} s")

for s in model.S:
    print(f"y_{s:<8} = {value(model.y[s,Nz,tf]):.8f}")

print(f"sum(y)     = {sum(value(model.y[s,Nz,tf]) for s in model.S):.8f}")
print(f"u_out      = {value(model.u[Nz,tf]):.8f} m/s")

times = np.array([float(t)*t_final for t in model.t])

plt.figure(figsize=(8,5))

for s in model.S:
    yout = np.array([value(model.y[s,Nz,t]) for t in model.t])
    plt.plot(times, yout, linewidth=2, label=s)

plt.xlabel('Time [s]')
plt.ylabel('Outlet mole fraction [-]')
plt.title('Breakthrough curve')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# ADSORBED UPTAKE PROFILES
# ============================================================

times_to_plot = np.linspace(0.0, t_final, 11)

z_centers = np.array([(j - 0.5)*dz for j in model.J])

for s in model.S:

    plt.figure(figsize=(8,5))

    for t_target in times_to_plot:

        tau_target = t_target/t_final
        tau_plot = min(model.t, key=lambda tau: abs(float(tau) - tau_target))
        physical_time = float(tau_plot)*t_final

        q_profile = np.array([value(model.q[s,j,tau_plot]) for j in model.J])

        plt.plot(z_centers, q_profile, linewidth=1.5, label=f'{physical_time:.0f} s')

    plt.xlabel('Axial position [m]')
    plt.ylabel('Adsorbed uptake [mol/kg]')
    plt.title(f'{s} uptake profile')
    plt.legend(title='Time')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# ACTUAL vs EQUILIBRIUM UPTAKE AT FINAL TIME
# ============================================================

tf = model.t.last()

for s in model.S:

    q_profile = np.array([value(model.q[s,j,tf]) for j in model.J])
    qstar_profile = np.array([value(model.qstar[s,j,tf]) for j in model.J])

    plt.figure(figsize=(8,5))

    plt.plot(z_centers, q_profile, linewidth=2, label='q')
    plt.plot(z_centers, qstar_profile, '--', linewidth=2, label='q*')

    plt.xlabel('Axial position [m]')
    plt.ylabel('Adsorbed uptake [mol/kg]')
    plt.title(f'{s}: actual and equilibrium uptake at {float(tf)*t_final:.0f} s')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# AVERAGE BED UPTAKE vs TIME
# ============================================================

times = np.array([float(t)*t_final for t in model.t])

plt.figure(figsize=(8,5))

for s in model.S:

    q_avg = np.array([
        sum(value(model.q[s,j,t]) for j in model.J)/Nz
        for t in model.t
    ])

    plt.plot(times, q_avg, linewidth=2, label=s)

plt.xlabel('Time [s]')
plt.ylabel('Average adsorbed uptake [mol/kg]')
plt.title('Average bed uptake')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()