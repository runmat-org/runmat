use super::*;
use crate::builtins::fea::values::{ELECTRIC_CURRENT_CLASS, ELECTRIC_POTENTIAL_CLASS};

define_simple_constraint!(
    electrical_equipotential_builtin,
    FEA_ELECTRICAL_CONSTRAINT_EQUIPOTENTIAL_DESCRIPTOR,
    "fea.ElectricalConstraint.equipotential",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::electrical",
    |constraint_id, target| ElectricalConstraint {
        constraint_id,
        kind: ElectricalConstraintKind::Equipotential { target },
    }
);

define_simple_constraint!(
    electrical_floating_conductor_builtin,
    FEA_ELECTRICAL_CONSTRAINT_FLOATING_CONDUCTOR_DESCRIPTOR,
    "fea.ElectricalConstraint.floatingConductor",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::electrical",
    |constraint_id, target| ElectricalConstraint {
        constraint_id,
        kind: ElectricalConstraintKind::FloatingConductor { target },
    }
);

define_simple_constraint!(
    electrical_potential_reference_builtin,
    FEA_ELECTRICAL_CONSTRAINT_POTENTIAL_REFERENCE_DESCRIPTOR,
    "fea.ElectricalConstraint.potentialReference",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::electrical",
    |constraint_id, target| ElectricalConstraint {
        constraint_id,
        kind: ElectricalConstraintKind::PotentialReference { target },
    }
);

define_quantity_constraint!(
    electrical_prescribed_potential_builtin,
    FEA_ELECTRICAL_CONSTRAINT_PRESCRIBED_POTENTIAL_DESCRIPTOR,
    "fea.ElectricalConstraint.prescribedPotential",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::electrical",
    ELECTRIC_POTENTIAL_CLASS,
    "volts",
    |constraint_id, target, volts| ElectricalConstraint {
        constraint_id,
        kind: ElectricalConstraintKind::PrescribedPotential {
            target,
            value: PrescribedValue::Constant(ElectricPotential::new(volts)),
        },
    }
);

define_quantity_constraint!(
    electrical_integral_current_builtin,
    FEA_ELECTRICAL_CONSTRAINT_INTEGRAL_CURRENT_DESCRIPTOR,
    "fea.ElectricalConstraint.integralCurrent",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::electrical",
    ELECTRIC_CURRENT_CLASS,
    "amperes",
    |constraint_id, target, amperes| ElectricalConstraint {
        constraint_id,
        kind: ElectricalConstraintKind::IntegralCurrent {
            target,
            value: PrescribedValue::Constant(ElectricCurrent::new(amperes)),
        },
    }
);
