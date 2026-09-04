import math
from pyomo.environ import *
from pyomo.dae import *

model = ConcreteModel()

# --------------------------------------------------
# Domains
# --------------------------------------------------

L = 0.2 #1.0
t_final = 1000.0

model.z = ContinuousSet(bounds=(0.0, L))
model.t = ContinuousSet(bounds=(0.0, 1.0))

specs = ['N2', 'H2', 'NH3']
model.S = Set(initialize=specs)

# --------------------------------------------------
# Operating parameters
# --------------------------------------------------

R = 8.31446261815324
T_in = 673.15 #700.0
T_ref = 673.15 #700.0
P_in = 30e5 #50e5
u_in = 0.1 #0.01

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

# model.k0 = Param(initialize=1.0)
# model.Ea = Param(initialize=80000.0)
# model.Keq = Param(initialize=1.0)
# model.Kads = Param(initialize=1.0)
model.k1_prefactor = Param(initialize=1.79e4)
model.km1_prefactor = Param(initialize=2.57e16)

model.Ea1 = Param(initialize=-87090.0)        # J/mol
model.Eam1 = Param(initialize=-198464.0)      # J/mol

model.alpha = Param(initialize=0.5)
model.beta = Param(initialize=0.5)

model.eta = Param(initialize=4.75)

nu = {'N2': -1.0, 'H2': -3.0, 'NH3': 2.0}
model.nu = Param(model.S, initialize=nu)

nu_total = sum(nu.values())
model.nu_total = Param(initialize=nu_total)

y_in = {'N2': 0.25, 'H2': 0.7499, 'NH3': 1e-4}

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
# Variables
# --------------------------------------------------

model.y = Var(model.S, model.z, model.t, bounds=(0.0, 1.0), initialize=lambda m,s,z,t: y_in[s])
model.C = Var(model.z, model.t, domain=PositiveReals, initialize=Ctot_in)
model.Phat = Var(model.z, model.t, bounds=(0.1, 1.1), initialize=1.0)
model.uhat = Var(model.z, model.t, bounds=(0.01, 10.0), initialize=1.0)
model.Temp = Var(model.z, model.t, bounds=(300.0, 1200.0), initialize=T_in)
model.N = Var(model.S, model.z, model.t, initialize=lambda m,s,z,t: u_in*Ctot_in*y_in[s])

# spatial derivatives
model.dydz = DerivativeVar(model.y, wrt=model.z)
model.dCdz = DerivativeVar(model.C, wrt=model.z)
model.dPhatdz = DerivativeVar(model.Phat, wrt=model.z)
model.duhatdz = DerivativeVar(model.uhat, wrt=model.z)
model.dTdz = DerivativeVar(model.Temp, wrt=model.z)
model.dNdz = DerivativeVar(model.N, wrt=model.z)

# time derivatives
model.dCdt = DerivativeVar(model.C, wrt=model.t)
model.dydt = DerivativeVar(model.y, wrt=model.t)
model.dTdt = DerivativeVar(model.Temp, wrt=model.t)

# second derivatives, if axial dispersion / conduction retained
# model.d2ydz2 = DerivativeVar(model.dydz, wrt=model.z)
# model.d2Cdz2 = DerivativeVar(model.dCdz, wrt=model.z)
# model.d2Tdz2 = DerivativeVar(model.dTdz, wrt=model.z)
model.d2ydz2 = Var(model.S, model.z, model.t, initialize=0.0)
model.d2Cdz2 = Var(model.z, model.t, initialize=0.0)
model.d2Tdz2 = Var(model.z, model.t, initialize=0.0)

# def total_concentration_rule(m,z,t):
#     return m.Ctot[z,t] == sum(m.Ci[s,z,t] for s in m.S)

# model.total_concentration = Constraint(model.z, model.t, rule=total_concentration_rule)


# def component_concentration_rule(m,s,z,t):
#     return m.Ci[s,z,t] == m.y[s,z,t]*m.Ctot[z,t]

# model.component_concentration = Constraint(model.S, model.z, model.t, rule=component_concentration_rule)

def ideal_gas_rule(m,z,t):
    return m.C[z,t]*m.Ru*m.Temp[z,t] == m.Pin*m.Phat[z,t]

model.ideal_gas = Constraint(model.z, model.t, rule=ideal_gas_rule)

def component_flux_rule(m,s,z,t):
    return m.N[s,z,t] == m.y[s,z,t]*m.C[z,t]*m.uin*m.uhat[z,t]

model.component_flux = Constraint(model.S, model.z, model.t, rule=component_flux_rule)

# model.Jdisp = Var(model.S, model.z, model.t, initialize=0.0)
# model.dJdispdz = DerivativeVar(model.Jdisp, wrt=model.z)

# def dispersion_flux_rule(m,s,z,t):
#     return m.Jdisp[s,z,t] == -m.epsilon*m.Dax*m.Ctot[z,t]*m.dydz[s,z,t]

# model.dispersion_flux = Constraint(model.S, model.z, model.t, rule=dispersion_flux_rule)

# --------------------------------------------------
# Reaction rate
# --------------------------------------------------
model.reaction_factor = Param(initialize=0.0, mutable=True)

def reaction_rate(m,z,t):
    Tlocal = m.Temp[z,t]
    Pbar = m.Pin*m.Phat[z,t]/1e5

    PN2 = m.y['N2',z,t]*Pbar
    PH2 = m.y['H2',z,t]*Pbar
    PNH3 = m.y['NH3',z,t]*Pbar

    PH2_safe = PH2 + 1e-10
    PNH3_safe = PNH3 + 1e-10

    k1 = m.k1_prefactor*exp(m.Ea1/(m.Ru*Tlocal))
    km1 = m.km1_prefactor*exp(m.Eam1/(m.Ru*Tlocal))

    eta_converted = m.eta/m.epsilon*1000.0/3600.0

    forward = k1*PN2*(PH2_safe**3/PNH3_safe**2)**m.alpha
    backward = km1*(PNH3_safe**2/PH2_safe**3)**m.beta

    return m.reaction_factor*eta_converted*(forward - backward)
# --------------------------------------------------
# Component mole-fraction balances
#
# Obtained from:
#
# component balance - yi*(overall balance)
#
# No explicit sum(yi)=1 constraint
# --------------------------------------------------
def component_balance_rule(m,s,z,t):
    if t == m.t.first():
        return Constraint.Skip

    if z == m.z.first():
        return Constraint.Skip

    elif z == m.z.last():
        return m.dydz[s,z,t] == 0.0

    else:
        accumulation = m.epsilon*(m.C[z,t]*m.dydt[s,z,t] + m.y[s,z,t]*m.dCdt[z,t])/t_final

        convection = m.dNdz[s,z,t]

        dispersion = m.epsilon*m.Dax*(m.d2Cdz2[z,t]*m.y[s,z,t] + 2.0*m.dydz[s,z,t]*m.dCdz[z,t] + m.d2ydz2[s,z,t]*m.C[z,t])

        source = (1.0 - m.epsilon)*m.nu[s]*reaction_rate(m,z,t)

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
model.dPhatdt = DerivativeVar(model.Phat, wrt=model.t)

def velocity_balance_rule(m,z,t):
    if t == m.t.first():
        return Constraint.Skip

    if z == m.z.first():
        return Constraint.Skip

    elif z == m.z.last():
        return m.duhatdz[z,t] == 0.0

    else:
        Tlocal = m.Temp[z,t]
        Phat = m.Phat[z,t]
        ulocal = m.uin*m.uhat[z,t]
        dudZ = m.uin*m.duhatdz[z,t]

        accumulation = m.epsilon*(m.dPhatdt[z,t]/Phat - m.dTdt[z,t]/Tlocal)/t_final

        convection = dudZ + ulocal*(m.dPhatdz[z,t]/Phat - m.dTdz[z,t]/Tlocal)

        dispersion_sum = sum(m.epsilon*m.Dax*(m.d2Cdz2[z,t]*m.y[s,z,t] + 2.0*m.dydz[s,z,t]*m.dCdz[z,t] + m.d2ydz2[s,z,t]*m.C[z,t]) for s in m.S)/m.C[z,t]

        reaction_sum = (1.0 - m.epsilon)*sum(m.nu[s] for s in m.S)*reaction_rate(m,z,t)/m.C[z,t]

        return accumulation + convection - dispersion_sum - reaction_sum == 0.0

model.velocity_balance = Constraint(model.z, model.t, rule=velocity_balance_rule)
# --------------------------------------------------
# Energy balance
# --------------------------------------------------
def energy_balance_rule(m,z,t):
    if t == m.t.first():
        return Constraint.Skip

    if z == m.z.first():
        return Constraint.Skip

    elif z == m.z.last():
        return m.dTdz[z,t] == 0.0

    else:
        Ctot = m.C[z,t]
        ulocal = m.uin*m.uhat[z,t]

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
def pressure_drop_rule(m,z,t):
    if z == m.z.first():
        return Constraint.Skip

    elif z == m.z.last():
        return m.dPhatdz[z,t] == 0.0

    else:
        Ctot = m.C[z,t]
        MWmix = sum(m.y[s,z,t]*m.MW[s] for s in m.S)
        rho_g = Ctot*MWmix

        viscous_scaled = L/m.Pin*150.0*m.mu*(1.0 - m.epsilon)**2/(m.epsilon**3*m.dp**2)*m.uin*m.uhat[z,t]
        inertial_scaled = L/m.Pin*1.75*rho_g*(1.0 - m.epsilon)/(m.epsilon**3*m.dp)*m.uin**2*m.uhat[z,t]**2

        return -L*m.dPhatdz[z,t] == viscous_scaled + inertial_scaled

model.pressure_drop = Constraint(model.z, model.t, rule=pressure_drop_rule)

# --------------------------------------------------
# Initial composition
# --------------------------------------------------

def composition_initial_rule(m,s,z):
    if z == m.z.first():
        return Constraint.Skip
    return m.y[s,z,m.t.first()] == y_in[s]

model.composition_initial = Constraint(model.S, model.z, rule=composition_initial_rule)

def temperature_initial_rule(m,z):
    if z == m.z.first():
        return Constraint.Skip
    return m.Temp[z,m.t.first()] == m.Tin

model.temperature_initial = Constraint(model.z, rule=temperature_initial_rule)

def velocity_initial_rule(m,z):
    if z == m.z.first():
        return Constraint.Skip
    return m.uhat[z,m.t.first()] == 1.0

model.velocity_initial = Constraint(model.z, rule=velocity_initial_rule)

# --------------------------------------------------
# Inlet composition
# --------------------------------------------------

def composition_inlet_rule(m,s,t):
    return m.y[s,m.z.first(),t] == y_in[s]

model.composition_inlet = Constraint(model.S, model.t, rule=composition_inlet_rule)

def temperature_inlet_rule(m,t):
    return m.Temp[m.z.first(),t] == m.Tin

model.temperature_inlet = Constraint(model.t, rule=temperature_inlet_rule)

def pressure_inlet_rule(m,t):
    return m.Phat[m.z.first(),t] == 1.0

model.pressure_inlet = Constraint(model.t, rule=pressure_inlet_rule)

def velocity_inlet_rule(m,t):
    return m.uhat[m.z.first(),t] == 1.0

model.velocity_inlet = Constraint(model.t, rule=velocity_inlet_rule)

# --------------------------------------------------
# Outlet composition BC
#
# Zero dispersive flux:
# dyi/dz = 0
# --------------------------------------------------

# def composition_outlet_rule(m, s, t):
#     if t == m.t.first():
#         return Constraint.Skip
#     return m.dydz[s,m.z.last(),t] == 0.0

# model.composition_outlet = Constraint(model.S, model.t, rule=composition_outlet_rule)

# --------------------------------------------------
# Outlet temperature BC
# --------------------------------------------------

# def temperature_outlet_rule(m, t):
#     if t == m.t.first():
#         return Constraint.Skip
#     return m.dTdz[m.z.last(),t] == 0.0

# model.temperature_outlet = Constraint(model.t, rule=temperature_outlet_rule)

# --------------------------------------------------
# Discretization
# --------------------------------------------------

def d2y_rule(m,s,z,t):
    zlist = sorted(m.z)
    idx = zlist.index(z)

    if idx == 0 or idx == len(zlist)-1:
        return Constraint.Skip

    zm = zlist[idx-1]
    zp = zlist[idx+1]

    dz_local = float(zp-z)

    return m.d2ydz2[s,z,t] == (m.y[s,zp,t] - 2.0*m.y[s,z,t] + m.y[s,zm,t])/dz_local**2

model.d2y_disc = Constraint(model.S, model.z, model.t, rule=d2y_rule)

def d2C_rule(m,z,t):
    zlist = sorted(m.z)
    idx = zlist.index(z)

    if idx == 0 or idx == len(zlist)-1:
        return Constraint.Skip

    zm = zlist[idx-1]
    zp = zlist[idx+1]

    dz_local = float(zp-z)

    return m.d2Cdz2[z,t] == (m.C[zp,t] - 2.0*m.C[z,t] + m.C[zm,t])/dz_local**2

model.d2C_disc = Constraint(model.z, model.t, rule=d2C_rule)

def d2T_rule(m,z,t):
    zlist = sorted(m.z)
    idx = zlist.index(z)

    if idx == 0 or idx == len(zlist)-1:
        return Constraint.Skip

    zm = zlist[idx-1]
    zp = zlist[idx+1]

    dz_local = float(zp-z)

    return m.d2Tdz2[z,t] == (m.Temp[zp,t] - 2.0*m.Temp[z,t] + m.Temp[zm,t])/dz_local**2

model.d2T_disc = Constraint(model.z, model.t, rule=d2T_rule)

n_z = 40
n_t = 40

TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.z, nfe=n_z - 1, scheme='BACKWARD')


TransformationFactory('dae.finite_difference').apply_to(model, wrt=model.t, nfe=n_t - 1, scheme='BACKWARD')

# --------------------------------------------------
# Initial guesses after discretization
# --------------------------------------------------

pressure_drop_guess = 0.0002

for z in model.z:
    z_norm = float(z)/L

    for t in model.t:
        Phat_guess = 1.0 - pressure_drop_guess*z_norm
        Ctot_guess = P_in*Phat_guess/(R*T_in)

        model.Phat[z,t].set_value(Phat_guess)
        model.uhat[z,t].set_value(1.0)
        model.Temp[z,t].set_value(T_in)
        # model.Ctot[z,t].set_value(Ctot_guess)
        # model.Ntot[z,t].set_value(u_in*Ctot_guess)

        for s in model.S:
            model.y[s,z,t].set_value(y_in[s])
            # model.Ci[s,z,t].set_value(y_in[s]*Ctot_guess)
            model.N[s,z,t].set_value(u_in*y_in[s]*Ctot_guess)
            # model.Jdisp[s,z,t].set_value(0.0)

# --------------------------------------------------
# Solve
# --------------------------------------------------

reaction_factors = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# reaction_factors = [1.0]

solver = SolverFactory('ipopt')
solver.options['max_iter'] = 3000
solver.options['tol'] = 1e-7
solver.options['linear_solver'] = 'ma97'

for factor in reaction_factors:
    print(f"\n{'='*60}")
    print(f"Reaction activation factor = {factor:.4f}")
    print(f"{'='*60}")

    model.reaction_factor.set_value(factor)

    results = solver.solve(model, tee=True)

    print("Termination:", results.solver.termination_condition)

    if results.solver.termination_condition not in [TerminationCondition.optimal, TerminationCondition.locallyOptimal]:
        print(f"Failed at reaction factor = {factor}")
        break

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

    # t = model.t.last()

    # for z in list(model.z)[1:5]:
    #     print("\nz =", float(z))

    #     for s in model.S:
    #         accumulation = value(model.epsilon*(model.C[z,t]*model.dydt[s,z,t] + model.y[s,z,t]*model.dCdt[z,t])/t_final)
    #         convection = value(model.dNdz[s,z,t])
    #         dispersion = value(model.epsilon*model.Dax*(model.d2Cdz2[z,t]*model.y[s,z,t] + 2.0*model.dydz[s,z,t]*model.dCdz[z,t] + model.d2ydz2[s,z,t]*model.C[z,t]))
    #         source = value((1.0-model.epsilon)*model.nu[s]*reaction_rate(model,z,t))

    #         print(s, "acc =", accumulation, "dNdz =", convection, "disp =", dispersion, "source =", source, "res =", accumulation + convection - dispersion - source)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

# --------------------------------------------------
# Common plotting setup
# --------------------------------------------------

times_to_plot = np.linspace(0.0, t_final, 21)

z_values = np.array([float(z) for z in model.z])

# --------------------------------------------------
# Mole fraction profiles
# --------------------------------------------------

cmap = colormaps['Blues']
colors = cmap(np.linspace(0.25, 1.0, len(times_to_plot)))

for s in model.S:
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, t_target in enumerate(times_to_plot):
        tau_target = t_target/t_final
        tau_plot = min(model.t, key=lambda tau: abs(float(tau) - tau_target))
        physical_time = float(tau_plot)*t_final

        y_profile = np.array([value(model.y[s,z,tau_plot]) for z in model.z])

        ax.plot(z_values, y_profile, color=colors[i], linewidth=1.5, label=f'{physical_time:.1f} s')

    ax.set_xlabel('Axial position, z [m]')
    ax.set_ylabel('Mole fraction [-]')
    ax.set_title(f'{s} mole fraction profile')
    ax.legend(title='Time', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Temperature profiles
# --------------------------------------------------

cmap = colormaps['Reds']
colors = cmap(np.linspace(0.25, 1.0, len(times_to_plot)))

fig, ax = plt.subplots(figsize=(7, 5))

for i, t_target in enumerate(times_to_plot):
    tau_target = t_target/t_final
    tau_plot = min(model.t, key=lambda tau: abs(float(tau) - tau_target))
    physical_time = float(tau_plot)*t_final

    T_profile = np.array([value(model.Temp[z,tau_plot]) for z in model.z])

    ax.plot(z_values, T_profile, color=colors[i], linewidth=1.5, label=f'{physical_time:.1f} s')

ax.set_xlabel('Axial position, z [m]')
ax.set_ylabel('Temperature [K]')
ax.set_title('Temperature profile')
ax.legend(title='Time', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# Pressure profiles
# --------------------------------------------------

cmap = colormaps['Greens']
colors = cmap(np.linspace(0.25, 1.0, len(times_to_plot)))

fig, ax = plt.subplots(figsize=(7, 5))

for i, t_target in enumerate(times_to_plot):
    tau_target = t_target/t_final
    tau_plot = min(model.t, key=lambda tau: abs(float(tau) - tau_target))
    physical_time = float(tau_plot)*t_final

    P_profile = np.array([value(model.Pin*model.Phat[z,tau_plot])/1e5 for z in model.z])

    ax.plot(z_values, P_profile, color=colors[i], linewidth=1.5, label=f'{physical_time:.1f} s')

ax.set_xlabel('Axial position, z [m]')
ax.set_ylabel('Pressure [bar]')
ax.set_title('Pressure profile')
ax.legend(title='Time', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# Superficial velocity profiles
# --------------------------------------------------

cmap = colormaps['Purples']
colors = cmap(np.linspace(0.25, 1.0, len(times_to_plot)))

fig, ax = plt.subplots(figsize=(7, 5))

for i, t_target in enumerate(times_to_plot):
    tau_target = t_target/t_final
    tau_plot = min(model.t, key=lambda tau: abs(float(tau) - tau_target))
    physical_time = float(tau_plot)*t_final

    u_profile = np.array([value(model.uin*model.uhat[z,tau_plot]) for z in model.z])

    ax.plot(z_values, u_profile, color=colors[i], linewidth=1.5, label=f'{physical_time:.1f} s')

ax.set_xlabel('Axial position, z [m]')
ax.set_ylabel('Superficial velocity [m/s]')
ax.set_title('Gas velocity profile')
ax.legend(title='Time', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# Sum of mole fractions
# Useful diagnostic
# --------------------------------------------------

cmap = colormaps['Greys']
colors = cmap(np.linspace(0.25, 1.0, len(times_to_plot)))

fig, ax = plt.subplots(figsize=(7, 5))

for i, t_target in enumerate(times_to_plot):
    tau_target = t_target/t_final
    tau_plot = min(model.t, key=lambda tau: abs(float(tau) - tau_target))
    physical_time = float(tau_plot)*t_final

    ysum_profile = np.array([sum(value(model.y[s,z,tau_plot]) for s in model.S) for z in model.z])

    ax.plot(z_values, ysum_profile, color=colors[i], linewidth=1.5, label=f'{physical_time:.1f} s')

ax.set_xlabel('Axial position, z [m]')
ax.set_ylabel('Sum of mole fractions [-]')
ax.set_title('Mole fraction summation check')
ax.axhline(1.0, linestyle='--', linewidth=1.0)
ax.legend(title='Time', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()