use super::*;
use crate::builtins::fea::values::MAGNETIC_VECTOR_POTENTIAL_CLASS;

define_simple_constraint!(
    magnetic_vector_potential_ground_builtin,
    FEA_MAGNETIC_CONSTRAINT_VECTOR_POTENTIAL_GROUND_DESCRIPTOR,
    "fea.MagneticConstraint.vectorPotentialGround",
    FEA_MAGNETIC_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::magnetic",
    |constraint_id, target| MagneticConstraint {
        constraint_id,
        kind: MagneticConstraintKind::VectorPotentialGround { target },
    }
);

define_simple_constraint!(
    magnetic_coulomb_gauge_builtin,
    FEA_MAGNETIC_CONSTRAINT_COULOMB_GAUGE_DESCRIPTOR,
    "fea.MagneticConstraint.coulombGauge",
    FEA_MAGNETIC_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::magnetic",
    |constraint_id, target| MagneticConstraint {
        constraint_id,
        kind: MagneticConstraintKind::CoulombGauge { target },
    }
);

define_quantity_constraint!(
    magnetic_prescribed_vector_potential_builtin,
    FEA_MAGNETIC_CONSTRAINT_PRESCRIBED_VECTOR_POTENTIAL_DESCRIPTOR,
    "fea.MagneticConstraint.prescribedVectorPotential",
    FEA_MAGNETIC_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::magnetic",
    MAGNETIC_VECTOR_POTENTIAL_CLASS,
    "webers_per_meter",
    |constraint_id, target, webers_per_meter| MagneticConstraint {
        constraint_id,
        kind: MagneticConstraintKind::PrescribedVectorPotential {
            target,
            value: PrescribedValue::Constant(MagneticVectorPotential::new(webers_per_meter)),
        },
    }
);
