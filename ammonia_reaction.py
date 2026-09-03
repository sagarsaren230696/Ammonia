# ammonia_reaction.py

from pyomo.environ import units as pyunits

from idaes.models.properties.modular_properties.base.generic_reaction import (
    ConcentrationForm,
)

from idaes.models.properties.modular_properties.reactions.dh_rxn import (
    constant_dh_rxn,
)

from idaes.models.properties.modular_properties.reactions.rate_constant import (
    arrhenius,
)

from idaes.models.properties.modular_properties.reactions.rate_forms import (
    power_law_rate,
)


configuration = {
    "base_units": {
        "time": pyunits.s,
        "length": pyunits.m,
        "mass": pyunits.kg,
        "amount": pyunits.mol,
        "temperature": pyunits.K,
    },

    "rate_reactions": {
        "R1": {
            "stoichiometry": {
                ("Vap", "N2"): -1,
                ("Vap", "H2"): -3,
                ("Vap", "NH3"): 2,
            },

            "heat_of_reaction": constant_dh_rxn,

            # Built-in Arrhenius:
            # k = A exp(-Ea / RT)
            "rate_constant": arrhenius,

            # Built-in power-law:
            # r = k * product(C_i ** order_i)
            "rate_form": power_law_rate,

            # For a first skeleton, use mole fractions.
            # Better later: use partial pressure or fugacity-based custom rate.
            "concentration_form": ConcentrationForm.moleFraction,

            "parameter_data": {
                # For N2 + 3H2 -> 2NH3
                # Approximate heat of reaction per mol N2 consumed:
                # ΔHrxn ≈ -92.2 kJ/mol-reaction
                "dh_rxn_ref": (-92.2e3, pyunits.J / pyunits.mol),

                # Dummy kinetic parameters.
                # Replace with fitted values.
                #
                # Since concentration_form = moleFraction, the concentration
                # term is dimensionless, so A has units mol/m3/s.
                "arrhenius_const": (
                    1e4,
                    pyunits.mol / pyunits.m**3 / pyunits.s,
                ),

                "energy_activation": (
                    40.0e3,
                    pyunits.J / pyunits.mol,
                ),

                # Explicitly set reaction order.
                # Without this, IDAES may infer elementary orders from stoichiometry.
                "reaction_order": {
                    ("Vap", "N2"): 1.0,
                    ("Vap", "H2"): 3.0,
                    ("Vap", "NH3"): 0.0,
                },
            },
        },
    },
}