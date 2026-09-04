use super::*;
use crate::builtins::fea::values::{ELECTRIC_CURRENT_CLASS, ELECTRIC_POTENTIAL_CLASS};

define_simple_constraint!(
    electrical_equipotential_builtin,
    ELECTRICAL_EQUIPOTENTIAL_DESCRIPTOR,
    ELECTRICAL_EQUIPOTENTIAL_SIGNATURES,
    "fea.ElectricalConstraint.equipotential",
    "constraint = fea.ElectricalConstraint.equipotential(id, region)",
    "Create an electrical equipotential constraint.",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    |constraint_id, target| ElectricalConstraint {
        constraint_id,
        kind: ElectricalConstraintKind::Equipotential { target },
    }
);

define_simple_constraint!(
    electrical_floating_conductor_builtin,
    ELECTRICAL_FLOATING_CONDUCTOR_DESCRIPTOR,
    ELECTRICAL_FLOATING_CONDUCTOR_SIGNATURES,
    "fea.ElectricalConstraint.floatingConductor",
    "constraint = fea.ElectricalConstraint.floatingConductor(id, region)",
    "Create an electrical floating-conductor constraint.",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    |constraint_id, target| ElectricalConstraint {
        constraint_id,
        kind: ElectricalConstraintKind::FloatingConductor { target },
    }
);

define_simple_constraint!(
    electrical_potential_reference_builtin,
    ELECTRICAL_POTENTIAL_REFERENCE_DESCRIPTOR,
    ELECTRICAL_POTENTIAL_REFERENCE_SIGNATURES,
    "fea.ElectricalConstraint.potentialReference",
    "constraint = fea.ElectricalConstraint.potentialReference(id, region)",
    "Create an electrical potential-reference constraint.",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    |constraint_id, target| ElectricalConstraint {
        constraint_id,
        kind: ElectricalConstraintKind::PotentialReference { target },
    }
);

define_quantity_constraint!(
    electrical_prescribed_potential_builtin,
    ELECTRICAL_PRESCRIBED_POTENTIAL_DESCRIPTOR,
    ELECTRICAL_PRESCRIBED_POTENTIAL_SIGNATURES,
    ELECTRICAL_PRESCRIBED_POTENTIAL_INPUTS,
    "fea.ElectricalConstraint.prescribedPotential",
    "constraint = fea.ElectricalConstraint.prescribedPotential(id, region, potential)",
    "Create a prescribed electric-potential constraint.",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    "potential",
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
    ELECTRICAL_INTEGRAL_CURRENT_DESCRIPTOR,
    ELECTRICAL_INTEGRAL_CURRENT_SIGNATURES,
    ELECTRICAL_INTEGRAL_CURRENT_INPUTS,
    "fea.ElectricalConstraint.integralCurrent",
    "constraint = fea.ElectricalConstraint.integralCurrent(id, region, current)",
    "Create an electrical integral-current constraint.",
    FEA_ELECTRICAL_CONSTRAINT_CLASS,
    "current",
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
