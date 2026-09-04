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
model.t = ContinuousSet(bounds=(0.0, 1.0))

specs = ['N2', 'H2', 'NH3']
model.S = Set(initialize=specs)

# --------------------------------------------------
# Operating parameters
# --------------------------------------------------

R = 8.31446261815324
T_in = 700.0
T_ref = 700.0
P_in = 50e5
u_in = 0.01

model.Ru = Param(initialize=R)
model.Tin = Param(initialize=T_in)
model.Tref = Param(initialize=T_ref)
model.Pin = Param(initialize=P_in)
model.uin = Param(initialize=u_in)

model.epsilon = Param(initialize=0.45)
model.Dax = Param(initialize=1e-5)

Ctot_in = P_in/(R*T_in)

# --------------------------------------------------
# Reaction parameters
# --------------------------------------------------

model.k0 = Param(initialize=1.0)
model.Ea = Param(initialize=80000.0)
model.Keq = Param(initialize=1.0)
model.Kads = Param(initialize=1.0)

nu = {'N2': -1.0, 'H2': -3.0, 'NH3': 2.0}
model.nu = Param(model.S, initialize=nu)

nu_total = sum(nu.values())
model.nu_total = Param(initialize=nu_total)

y_in = {'N2': 0.25, 'H2': 0.75, 'NH3': 0.0}

# --------------------------------------------------
# Thermal parameters
# --------------------------------------------------

model.Cpg = Param(initialize=30.0)
model.rho_cat = Param(initialize=1500.0)
model.Cp_cat = Param(initialize=1000.0)
model.k_eff = Param(initialize=0.5)
model.dHrxn = Param(initialize=-92000.0)

# --------------------------------------------------
# Ergun parameters
# --------------------------------------------------

model.dp = Param(initialize=2.0e-3)
model.mu = Param(initialize=3.0e-5)

MW = {'N2': 28.0134e-3, 'H2': 2.01588e-3, 'NH3': 17.0305e-3}
model.MW = Param(model.S, initialize=MW)

# --------------------------------------------------
# Mole fractions
# --------------------------------------------------

model.y = Var(model.S, model.z, model.t, bounds=(0.0, 1.0), initialize=lambda m,s,z,t: y_in[s])

model.dydt = DerivativeVar(model.y, wrt=model.t)
model.dydz = DerivativeVar(model.y, wrt=model.z)
model.d2ydz2 = DerivativeVar(model.dydz, wrt=model.z)

# --------------------------------------------------
# Temperature
# --------------------------------------------------

model.Temp = Var(model.z, model.t, bounds=(300.0, 1200.0), initialize=T_in)

model.dTdt = DerivativeVar(model.Temp, wrt=model.t)
model.dTdz = DerivativeVar(model.Temp, wrt=model.z)
model.d2Tdz2 = DerivativeVar(model.dTdz, wrt=model.z)

# --------------------------------------------------
# Scaled pressure
#
# P = Pin*Phat
# --------------------------------------------------

model.Phat = Var(model.z, model.t, bounds=(0.1, 1.1), initialize=1.0)

model.dPhatdt = DerivativeVar(model.Phat, wrt=model.t)
model.dPhatdz = DerivativeVar(model.Phat, wrt=model.z)

# --------------------------------------------------
# Scaled velocity
#
# u = uin*uhat
# --------------------------------------------------

model.uhat = Var(model.z, model.t, bounds=(0.01, 10.0), initialize=1.0)

model.duhatdz = DerivativeVar(model.uhat, wrt=model.z)

# --------------------------------------------------
# Reaction rate
# --------------------------------------------------

def reaction_rate(m, z, t):
    Tlocal = m.Temp[z,t]
    Plocal = m.Pin*m.Phat[z,t]

    pN2 = m.y['N2',z,t]*Plocal
    pH2 = m.y['H2',z,t]*Plocal
    pNH3 = m.y['NH3',z,t]*Plocal

    pref = 1e5

    aN2 = pN2/pref
    aH2 = pH2/pref
    aNH3 = pNH3/pref

    k = m.k0*exp(-m.Ea/(m.Ru*Tlocal))

    Keq_T = m.Keq*exp(-m.dHrxn/m.Ru*(1.0/Tlocal - 1.0/m.Tref))

    driving_force = aN2*aH2**3 - aNH3**2/Keq_T
    denominator = (1.0 + m.Kads*aNH3)**2

    return k*driving_force/denominator

# --------------------------------------------------
# Component mole-fraction balances
#
# Obtained from:
#
# component balance - yi*(overall balance)
#
# No explicit sum(yi)=1 constraint
# --------------------------------------------------

def component_balance_rule(m, s, z, t):
    if t == m.t.first():
        return Constraint.Skip
    if z == m.z.first() or z == m.z.last():
        return Constraint.Skip

    yi = m.y[s,z,t]
    Tlocal = m.Temp[z,t]
    Phat = m.Phat[z,t]
    ulocal = m.uin*m.uhat[z,t]

    Ctot = m.Pin*Phat/(m.Ru*Tlocal)

    dlnCtot_dz = m.dPhatdz[z,t]/Phat - m.dTdz[z,t]/Tlocal

    accumulation = m.dydt[s,z,t]/t_final
    convection = ulocal/m.epsilon*m.dydz[s,z,t]
    dispersion = m.Dax*(m.d2ydz2[s,z,t] + dlnCtot_dz*m.dydz[s,z,t])
    source = (1.0 - m.epsilon)/(m.epsilon*Ctot)*(m.nu[s] - yi*m.nu_total)*reaction_rate(m,z,t)

    return accumulation + convection - dispersion == source

model.component_balance = Constraint(model.S, model.z, model.t, rule=component_balance_rule)

# --------------------------------------------------
# Overall molar balance
#
# Ctot = P/(RT)
#
# (1/Ctot)dCtot/dt =
# (1/P)dP/dt - (1/T)dT/dt
#
# (1/Ctot)dCtot/dz =
# (1/P)dP/dz - (1/T)dT/dz
#
# No overall dispersion term because sum(Ji)=0
# --------------------------------------------------

def overall_balance_rule(m, z, t):
    if t == m.t.first():
        return Constraint.Skip
    if z == m.z.first():
        return Constraint.Skip

    Tlocal = m.Temp[z,t]
    Phat = m.Phat[z,t]
    ulocal = m.uin*m.uhat[z,t]
    dudZ = m.uin*m.duhatdz[z,t]

    Ctot = m.Pin*Phat/(m.Ru*Tlocal)

    accumulation = (m.dPhatdt[z,t]/Phat - m.dTdt[z,t]/Tlocal)/t_final
    convection = (dudZ + ulocal*(m.dPhatdz[z,t]/Phat - m.dTdz[z,t]/Tlocal))/m.epsilon
    source = (1.0 - m.epsilon)/(m.epsilon*Ctot)*m.nu_total*reaction_rate(m,z,t)

    return accumulation + convection == source

model.overall_balance = Constraint(model.z, model.t, rule=overall_balance_rule)

# --------------------------------------------------
# Energy balance
# --------------------------------------------------

def energy_balance_rule(m, z, t):
    if t == m.t.first():
        return Constraint.Skip
    if z == m.z.first() or z == m.z.last():
        return Constraint.Skip

    Tlocal = m.Temp[z,t]
    Phat = m.Phat[z,t]
    ulocal = m.uin*m.uhat[z,t]

    Ctot = m.Pin*Phat/(m.Ru*Tlocal)

    rhoCp_gas = Ctot*m.Cpg
    rhoCp_eff = m.epsilon*rhoCp_gas + (1.0 - m.epsilon)*m.rho_cat*m.Cp_cat
    Qreaction = -(1.0 - m.epsilon)*m.dHrxn*reaction_rate(m,z,t)

    return rhoCp_eff*m.dTdt[z,t]/t_final + rhoCp_gas*ulocal*m.dTdz[z,t] - m.k_eff*m.d2Tdz2[z,t] == Qreaction

model.energy_balance = Constraint(model.z, model.t, rule=energy_balance_rule)

# --------------------------------------------------
# Ergun pressure-drop equation
#
# Scaled using Phat = P/Pin and uhat = u/uin
# --------------------------------------------------

def pressure_drop_rule(m, z, t):
    if z == m.z.first():
        return Constraint.Skip

    Tlocal = m.Temp[z,t]
    Phat = m.Phat[z,t]

    Ctot = m.Pin*Phat/(m.Ru*Tlocal)
    MWmix = sum(m.y[s,z,t]*m.MW[s] for s in m.S)
    rho_g = Ctot*MWmix

    viscous_scaled = L/m.Pin*150.0*m.mu*(1.0 - m.epsilon)**2/(m.epsilon**3*m.dp**2)*m.uin*m.uhat[z,t]
    inertial_scaled = L/m.Pin*1.75*rho_g*(1.0 - m.epsilon)/(m.epsilon**3*m.dp)*m.uin**2*m.uhat[z,t]**2

    return -L*m.dPhatdz[z,t] == viscous_scaled + inertial_scaled

model.pressure_drop = Constraint(model.z, model.t, rule=pressure_drop_rule)

# --------------------------------------------------
# Initial composition
# --------------------------------------------------

def composition_initial_rule(m, s, z):
    return m.y[s,z,m.t.first()] == y_in[s]

model.composition_initial = Constraint(model.S, model.z, rule=composition_initial_rule)

# --------------------------------------------------
# Initial temperature
# --------------------------------------------------

def temperature_initial_rule(m, z):
    if z == m.z.first():
        return Constraint.Skip
    return m.Temp[z,m.t.first()] == m.Tin

model.temperature_initial = Constraint(model.z, rule=temperature_initial_rule)

# --------------------------------------------------
# Initial velocity
#
# Choose initially uniform superficial velocity.
# Ergun + inlet pressure then determines the initial
# pressure profile.
# --------------------------------------------------

def velocity_initial_rule(m, z):
    if z == m.z.first():
        return Constraint.Skip
    return m.uhat[z,m.t.first()] == 1.0

model.velocity_initial = Constraint(model.z, rule=velocity_initial_rule)

# --------------------------------------------------
# Inlet composition
# --------------------------------------------------

def composition_inlet_rule(m, s, t):
    return m.y[s,m.z.first(),t] == y_in[s]

model.composition_inlet = Constraint(model.S, model.t, rule=composition_inlet_rule)

# --------------------------------------------------
# Inlet temperature
# --------------------------------------------------

def temperature_inlet_rule(m, t):
    return m.Temp[m.z.first(),t] == m.Tin

model.temperature_inlet = Constraint(model.t, rule=temperature_inlet_rule)

# --------------------------------------------------
# Inlet pressure
# --------------------------------------------------

def pressure_inlet_rule(m, t):
    return m.Phat[m.z.first(),t] == 1.0

model.pressure_inlet = Constraint(model.t, rule=pressure_inlet_rule)

# --------------------------------------------------
# Inlet velocity
# --------------------------------------------------

def velocity_inlet_rule(m, t):
    return m.uhat[m.z.first(),t] == 1.0

model.velocity_inlet = Constraint(model.t, rule=velocity_inlet_rule)

# --------------------------------------------------
# Outlet composition BC
#
# Zero dispersive flux:
# dyi/dz = 0
# --------------------------------------------------

def composition_outlet_rule(m, s, t):
    if t == m.t.first():
        return Constraint.Skip
    return m.dydz[s,m.z.last(),t] == 0.0

model.composition_outlet = Constraint(model.S, model.t, rule=composition_outlet_rule)

# --------------------------------------------------
# Outlet temperature BC
# --------------------------------------------------

def temperature_outlet_rule(m, t):
    if t == m.t.first():
        return Constraint.Skip
    return m.dTdz[m.z.last(),t] == 0.0

model.temperature_outlet = Constraint(model.t, rule=temperature_outlet_rule)

# --------------------------------------------------
# Discretization
# --------------------------------------------------

n_z = 40
n_t = 400

TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.z, nfe=n_z - 1, scheme='BACKWARD')
TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=n_t - 1, scheme='BACKWARD')

# --------------------------------------------------
# Initial guesses after discretization
# --------------------------------------------------

pressure_drop_guess = 0.0002

for z in model.z:
    z_norm = float(z)/L
    for t in model.t:
        model.Phat[z,t].set_value(1.0 - pressure_drop_guess*z_norm)
        model.uhat[z,t].set_value(1.0)
        model.Temp[z,t].set_value(T_in)
        for s in model.S:
            model.y[s,z,t].set_value(y_in[s])

# --------------------------------------------------
# Solve
# --------------------------------------------------

solver = SolverFactory('ipopt')
solver.options['max_iter'] = 3000
solver.options['tol'] = 1e-7
solver.options['linear_solver'] = 'ma97'

results = solver.solve(model, tee=True)

print(results.solver.termination_condition)

# --------------------------------------------------
# Basic result check
# --------------------------------------------------

zL = model.z.last()
tauL = model.t.last()

print("\nFinal outlet conditions")
print("-----------------------")
print(f"Time       = {float(tauL)*t_final:.4f} s")
print(f"y_N2       = {value(model.y['N2',zL,tauL]):.8f}")
print(f"y_H2       = {value(model.y['H2',zL,tauL]):.8f}")
print(f"y_NH3      = {value(model.y['NH3',zL,tauL]):.8f}")
print(f"sum(y)     = {sum(value(model.y[s,zL,tauL]) for s in model.S):.8f}")
print(f"T_out      = {value(model.Temp[zL,tauL]):.4f} K")
print(f"P_out      = {value(model.Pin*model.Phat[zL,tauL])/1e5:.6f} bar")
print(f"u_out      = {value(model.uin*model.uhat[zL,tauL]):.8f} m/s")