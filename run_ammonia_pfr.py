# run_ammonia_pfr.py

from pyomo.environ import (
    ConcreteModel,
    SolverFactory,
    value,
)

from idaes.core import (
    FlowsheetBlock,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
)

from idaes.models.unit_models import PFR

from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)

from idaes.models.properties.modular_properties.base.generic_reaction import (
    GenericReactionParameterBlock,
)

from idaes.core.util.model_statistics import degrees_of_freedom

import ammonia_thermo
import ammonia_reaction


m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)

# -------------------------------------------------------------------------
# Property and reaction packages
# -------------------------------------------------------------------------

m.fs.properties = GenericParameterBlock(
    **ammonia_thermo.configuration
)

m.fs.reactions = GenericReactionParameterBlock(
    property_package=m.fs.properties,
    **ammonia_reaction.configuration
)

# -------------------------------------------------------------------------
# Built-in IDAES PFR
# -------------------------------------------------------------------------

m.fs.R101 = PFR(
    property_package=m.fs.properties,
    reaction_package=m.fs.reactions,

    material_balance_type=MaterialBalanceType.componentTotal,
    energy_balance_type=EnergyBalanceType.enthalpyTotal,
    momentum_balance_type=MomentumBalanceType.pressureTotal,

    has_equilibrium_reactions=False,
    has_heat_of_reaction=True,
    has_heat_transfer=True,
    has_pressure_change=True,

    transformation_method="dae.finite_difference",
    transformation_scheme="BACKWARD",
    finite_elements=30,
)

r = m.fs.R101

# -------------------------------------------------------------------------
# Reactor geometry
# -------------------------------------------------------------------------

r.length.fix(0.2)       # m
r.area.fix(3.14159*0.02**2)        # m2

# IDAES PFR has two geometry DOF. Fix any two of length, area, volume.
# Do not fix volume if length and area are fixed.

# -------------------------------------------------------------------------
# Inlet conditions
# -------------------------------------------------------------------------

r.inlet.flow_mol.fix(0.03)          # mol/s
r.inlet.temperature.fix(673.15)      # K
r.inlet.pressure.fix(3.0e6)          # Pa

r.inlet.mole_frac_comp[0, "N2"].fix(0.25)
r.inlet.mole_frac_comp[0, "H2"].fix(0.75)
r.inlet.mole_frac_comp[0, "NH3"].fix(1e-10)

# -------------------------------------------------------------------------
# Heat transfer and pressure drop
# -------------------------------------------------------------------------

for t in m.fs.time:
    for x in r.control_volume.length_domain:
        # Adiabatic first test
        r.heat_duty[t, x].fix(0.0)

        # No pressure drop first test
        r.deltaP[t, x].fix(0.0)

# -------------------------------------------------------------------------
# Check DOF
# -------------------------------------------------------------------------

print("Degrees of freedom:", degrees_of_freedom(m))

# -------------------------------------------------------------------------
# Initialization and solve
# -------------------------------------------------------------------------

r.initialize()

solver = SolverFactory("ipopt")
res = solver.solve(m, tee=True)

print(res.solver.termination_condition)

# -------------------------------------------------------------------------
# Results
# -------------------------------------------------------------------------

print("Outlet T [K]:", value(r.outlet.temperature[0]))
print("Outlet P [Pa]:", value(r.outlet.pressure[0]))

print("Inlet y_N2:", value(r.inlet.mole_frac_comp[0, "N2"]))
print("Inlet y_H2:", value(r.inlet.mole_frac_comp[0, "H2"]))
print("Inlet y_NH3:", value(r.inlet.mole_frac_comp[0, "NH3"]))

print("Outlet y_N2:", value(r.outlet.mole_frac_comp[0, "N2"]))
print("Outlet y_H2:", value(r.outlet.mole_frac_comp[0, "H2"]))
print("Outlet y_NH3:", value(r.outlet.mole_frac_comp[0, "NH3"]))

print("Inlet velocity:",value(r.inlet.flow_mol[0]/(r.inlet.pressure[0]/8.314/r.inlet.temperature[0])/r.area))
print("GHSV:", value(r.inlet.flow_mol[0]/(r.inlet.pressure[0]/8.314/r.inlet.temperature[0])/r.area/r.length*3600))
print("GHSV (STP):", value(r.inlet.flow_mol[0]/(r.inlet.pressure[0]/8.314/r.inlet.temperature[0])/r.area/r.length*3600*(r.inlet.pressure[0]/r.inlet.temperature[0])*(273.15/1e5)))