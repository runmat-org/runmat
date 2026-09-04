use super::*;
use crate::builtins::fea::values::MAGNETIC_VECTOR_POTENTIAL_CLASS;

define_simple_constraint!(
    magnetic_vector_potential_ground_builtin,
    MAGNETIC_VECTOR_POTENTIAL_GROUND_DESCRIPTOR,
    MAGNETIC_VECTOR_POTENTIAL_GROUND_SIGNATURES,
    "fea.MagneticConstraint.vectorPotentialGround",
    "constraint = fea.MagneticConstraint.vectorPotentialGround(id, region)",
    "Create a magnetic vector-potential ground constraint.",
    FEA_MAGNETIC_CONSTRAINT_CLASS,
    |constraint_id, target| MagneticConstraint {
        constraint_id,
        kind: MagneticConstraintKind::VectorPotentialGround { target },
    }
);

define_simple_constraint!(
    magnetic_coulomb_gauge_builtin,
    MAGNETIC_COULOMB_GAUGE_DESCRIPTOR,
    MAGNETIC_COULOMB_GAUGE_SIGNATURES,
    "fea.MagneticConstraint.coulombGauge",
    "constraint = fea.MagneticConstraint.coulombGauge(id, region)",
    "Create a magnetic Coulomb-gauge constraint.",
    FEA_MAGNETIC_CONSTRAINT_CLASS,
    |constraint_id, target| MagneticConstraint {
        constraint_id,
        kind: MagneticConstraintKind::CoulombGauge { target },
    }
);

define_quantity_constraint!(
    magnetic_prescribed_vector_potential_builtin,
    MAGNETIC_PRESCRIBED_VECTOR_POTENTIAL_DESCRIPTOR,
    MAGNETIC_PRESCRIBED_VECTOR_POTENTIAL_SIGNATURES,
    MAGNETIC_PRESCRIBED_VECTOR_POTENTIAL_INPUTS,
    "fea.MagneticConstraint.prescribedVectorPotential",
    "constraint = fea.MagneticConstraint.prescribedVectorPotential(id, region, potential)",
    "Create a prescribed magnetic vector-potential constraint.",
    FEA_MAGNETIC_CONSTRAINT_CLASS,
    "potential",
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
