use super::*;
use crate::builtins::fea::values::DISPLACEMENT_CLASS;

define_simple_constraint!(
    structural_symmetry_builtin,
    STRUCTURAL_SYMMETRY_DESCRIPTOR,
    STRUCTURAL_SYMMETRY_SIGNATURES,
    "fea.StructuralConstraint.symmetry",
    "constraint = fea.StructuralConstraint.symmetry(id, region)",
    "Create a structural symmetry constraint.",
    FEA_STRUCTURAL_CONSTRAINT_CLASS,
    |constraint_id, target| StructuralConstraint {
        constraint_id,
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Symmetry { target },
    }
);

define_simple_constraint!(
    structural_antisymmetry_builtin,
    STRUCTURAL_ANTISYMMETRY_DESCRIPTOR,
    STRUCTURAL_ANTISYMMETRY_SIGNATURES,
    "fea.StructuralConstraint.antisymmetry",
    "constraint = fea.StructuralConstraint.antisymmetry(id, region)",
    "Create a structural antisymmetry constraint.",
    FEA_STRUCTURAL_CONSTRAINT_CLASS,
    |constraint_id, target| StructuralConstraint {
        constraint_id,
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Antisymmetry { target },
    }
);

define_quantity_constraint!(
    structural_normal_displacement_builtin,
    STRUCTURAL_NORMAL_DISPLACEMENT_DESCRIPTOR,
    STRUCTURAL_NORMAL_DISPLACEMENT_SIGNATURES,
    STRUCTURAL_NORMAL_DISPLACEMENT_INPUTS,
    "fea.StructuralConstraint.normalDisplacement",
    "constraint = fea.StructuralConstraint.normalDisplacement(id, region, displacement)",
    "Create a prescribed normal-displacement constraint.",
    FEA_STRUCTURAL_CONSTRAINT_CLASS,
    "displacement",
    DISPLACEMENT_CLASS,
    "meters",
    |constraint_id, target, meters| StructuralConstraint {
        constraint_id,
        frame: CoordinateFrame::BoundaryNormalTangent,
        kind: StructuralConstraintKind::NormalDisplacement {
            target,
            value: PrescribedValue::Constant(Displacement::new(meters)),
        },
    }
);
