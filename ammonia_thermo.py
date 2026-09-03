# ammonia_thermo.py

from pyomo.environ import units as pyunits

from idaes.core import Component, VaporPhase, PhaseType
from idaes.models.properties.modular_properties.state_definitions import FTPx
from idaes.models.properties.modular_properties.eos.ideal import Ideal

import idaes.models.properties.modular_properties.pure.NIST as NIST


configuration = {
    "components": {
        "N2": {
            "type": Component,
            "valid_phase_types": [PhaseType.vaporPhase],
            "elemental_composition": {"N": 2},
            "enth_mol_ig_comp": NIST,
            "entr_mol_ig_comp": NIST,
            "cp_mol_ig_comp": NIST,
            "parameter_data": {
                "mw": (28.0134e-3, pyunits.kg / pyunits.mol),
                "pressure_crit": (3.3958e6, pyunits.Pa),
                "temperature_crit": (126.2, pyunits.K),
                "omega": 0.0372,
                # NIST Shomate coefficients, valid approximately 500–2000 K
                "cp_mol_ig_comp_coeff": {
                    "A": 19.50583,
                    "B": 19.88705,
                    "C": -8.598535,
                    "D": 1.369784,
                    "E": 0.527601,
                    "F": -4.935202,
                    "G": 212.39,
                    "H": 0.0,
                },
            },
        },

        "H2": {
            "type": Component,
            "valid_phase_types": [PhaseType.vaporPhase],
            "elemental_composition": {"H": 2},
            "enth_mol_ig_comp": NIST,
            "entr_mol_ig_comp": NIST,
            "cp_mol_ig_comp": NIST,
            "parameter_data": {
                "mw": (2.01588e-3, pyunits.kg / pyunits.mol),
                "pressure_crit": (1.2964e6, pyunits.Pa),
                "temperature_crit": (33.145, pyunits.K),
                "omega": -0.219,
                # NIST Shomate coefficients, valid approximately 298–1000 K
                "cp_mol_ig_comp_coeff": {
                    "A": 33.066178,
                    "B": -11.363417,
                    "C": 11.432816,
                    "D": -2.772874,
                    "E": -0.158558,
                    "F": -9.980797,
                    "G": 172.707974,
                    "H": 0.0,
                },
            },
        },

        "NH3": {
            "type": Component,
            "valid_phase_types": [PhaseType.vaporPhase],
            "elemental_composition": {"N": 1, "H": 3},
            "enth_mol_ig_comp": NIST,
            "entr_mol_ig_comp": NIST,
            "cp_mol_ig_comp": NIST,
            "parameter_data": {
                "mw": (17.03052e-3, pyunits.kg / pyunits.mol),
                "pressure_crit": (11.333e6, pyunits.Pa),
                "temperature_crit": (405.65, pyunits.K),
                "omega": 0.256,
                # NIST Shomate coefficients, valid approximately 298–1400 K
                "cp_mol_ig_comp_coeff": {
                    "A": 19.99563,
                    "B": 49.77119,
                    "C": -15.37599,
                    "D": 1.921168,
                    "E": 0.189174,
                    "F": -53.30667,
                    "G": 203.8591,
                    "H": -45.89806,
                },
            },
        },
    },

    "phases": {
        "Vap": {
            "type": VaporPhase,
            "equation_of_state": Ideal,
        },
    },

    "base_units": {
        "time": pyunits.s,
        "length": pyunits.m,
        "mass": pyunits.kg,
        "amount": pyunits.mol,
        "temperature": pyunits.K,
    },

    "state_definition": FTPx,

    "state_bounds": {
        "flow_mol": (1e-6, 100.0, 1e5, pyunits.mol / pyunits.s),
        "temperature": (300.0, 700.0, 1200.0, pyunits.K),
        "pressure": (1e5, 10e6, 30e6, pyunits.Pa),
    },

    "pressure_ref": (101325.0, pyunits.Pa),
    "temperature_ref": (298.15, pyunits.K),
}