# ammonia_pfr.py

from pyomo.environ import Var, Constraint, Reference, units as pyunits
from pyomo.common.config import ConfigBlock, ConfigValue, In, Bool, ListOf

from idaes.core import (
    ControlVolume1DBlock,
    UnitModelBlockData,
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    useDefault,
)

from idaes.core.util.config import (
    is_physical_parameter_block,
    is_reaction_parameter_block,
)

from idaes.core.util.misc import add_object_reference


@declare_process_block_class("AmmoniaPFR")
class AmmoniaPFRData(UnitModelBlockData):
    """
    Custom 1D plug-flow reactor for ammonia synthesis.

    Reaction:
        N2 + 3 H2 <-> 2 NH3

    This unit assumes that the property package contains:
        N2, H2, NH3

    and that the reaction package defines one rate reaction, e.g.:
        "R1": N2 + 3 H2 -> 2 NH3
    """

    CONFIG = UnitModelBlockData.CONFIG()

    CONFIG.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.componentTotal,
            domain=In(MaterialBalanceType),
        ),
    )

    CONFIG.declare(
        "energy_balance_type",
        ConfigValue(
            default=EnergyBalanceType.enthalpyTotal,
            domain=In(EnergyBalanceType),
        ),
    )

    CONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
        ),
    )

    CONFIG.declare(
        "has_heat_of_reaction",
        ConfigValue(default=True, domain=Bool),
    )

    CONFIG.declare(
        "has_heat_transfer",
        ConfigValue(default=True, domain=Bool),
    )

    CONFIG.declare(
        "has_pressure_change",
        ConfigValue(default=True, domain=Bool),
    )

    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
        ),
    )

    CONFIG.declare(
        "property_package_args",
        ConfigBlock(implicit=True),
    )

    CONFIG.declare(
        "reaction_package",
        ConfigValue(
            default=None,
            domain=is_reaction_parameter_block,
        ),
    )

    CONFIG.declare(
        "reaction_package_args",
        ConfigBlock(implicit=True),
    )

    CONFIG.declare(
        "length_domain_set",
        ConfigValue(
            default=[0.0, 1.0],
            domain=ListOf(float),
        ),
    )

    CONFIG.declare(
        "transformation_method",
        ConfigValue(default="dae.finite_difference"),
    )

    CONFIG.declare(
        "transformation_scheme",
        ConfigValue(default="BACKWARD"),
    )

    CONFIG.declare(
        "finite_elements",
        ConfigValue(default=20),
    )

    CONFIG.declare(
        "collocation_points",
        ConfigValue(default=3),
    )

    def build(self):
        super().build()

        self.control_volume = ControlVolume1DBlock(
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            reaction_package=self.config.reaction_package,
            reaction_package_args=self.config.reaction_package_args,
            transformation_method=self.config.transformation_method,
            transformation_scheme=self.config.transformation_scheme,
            finite_elements=self.config.finite_elements,
            collocation_points=self.config.collocation_points,
        )

        self.control_volume.add_geometry(
            length_domain_set=self.config.length_domain_set
        )

        self.control_volume.add_state_blocks(
            has_phase_equilibrium=False
        )

        self.control_volume.add_reaction_blocks(
            has_equilibrium=False
        )

        self.control_volume.add_material_balances(
            balance_type=self.config.material_balance_type,
            has_rate_reactions=True,
            has_equilibrium_reactions=False,
            has_phase_equilibrium=False,
        )

        self.control_volume.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_heat_of_reaction=self.config.has_heat_of_reaction,
            has_heat_transfer=self.config.has_heat_transfer,
        )

        self.control_volume.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )

        self.control_volume.apply_transformation()

        self.add_inlet_port()
        self.add_outlet_port()

        add_object_reference(self, "length", self.control_volume.length)
        add_object_reference(self, "area", self.control_volume.area)

        units = self.config.property_package.get_metadata().get_derived_units

        self.volume = Var(
            initialize=1.0,
            units=units("volume"),
            doc="Total reactor volume",
        )

        self.geometry = Constraint(
            expr=self.volume == self.length * self.area
        )

        self.catalyst_density = Var(
            initialize=1000,
            bounds=(1e-6, None),
            units=pyunits.kg / pyunits.m**3,
            doc="Catalyst loading per reactor volume",
        )

        self.effectiveness_factor = Var(
            initialize=1.0,
            bounds=(0, 1),
            doc="Catalyst effectiveness factor",
        )

        @self.Constraint(
            self.flowsheet().time,
            self.control_volume.length_domain,
            self.config.reaction_package.rate_reaction_idx,
            doc="Ammonia PFR reaction extent equation",
        )
        def performance_eqn(b, t, x, r):
            return (
                b.control_volume.rate_reaction_extent[t, x, r]
                ==
                b.control_volume.area
                * b.effectiveness_factor
                * b.catalyst_density
                * b.control_volume.reactions[t, x].reaction_rate[r]
            )

        if self.config.has_heat_transfer:
            self.heat_duty = Reference(self.control_volume.heat[...])

        if self.config.has_pressure_change:
            self.deltaP = Reference(self.control_volume.deltaP[...])

    def _get_performance_contents(self, time_point=0):
        return {
            "vars": {
                "Length": self.length,
                "Area": self.area,
                "Volume": self.volume,
                "Catalyst density": self.catalyst_density,
                "Effectiveness factor": self.effectiveness_factor,
            }
        }