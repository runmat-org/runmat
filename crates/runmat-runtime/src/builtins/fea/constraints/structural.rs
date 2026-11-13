use super::support::{
    coordinate_frame_from_value, reference_point_from_value, structural_dof_values_from_value,
    structural_motions_from_value,
};
use super::*;
use crate::builtins::fea::values::DISPLACEMENT_CLASS;
use runmat_analysis_core::{PrescribedStructuralMotion, ReferencePointCoupling};

#[runtime_builtin(
    name = "fea.StructuralConstraint.prescribed",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::constraints::structural"
)]
pub async fn structural_prescribed_builtin(
    id: String,
    region: Value,
    values: Value,
    frame: Value,
) -> BuiltinResult<Value> {
    fixed_outputs(
        &runmat_builtins::FEA_STRUCTURAL_CONSTRAINT_PRESCRIBED_DESCRIPTOR,
        "fea.StructuralConstraint.prescribed",
    )?;
    let (id, geometry_id, geometry_revision, target) =
        bound_region("fea.StructuralConstraint.prescribed", id, &region)?;
    let (translations, rotations) =
        structural_dof_values_from_value("fea.StructuralConstraint.prescribed", &values)?;
    let frame = coordinate_frame_from_value("fea.StructuralConstraint.prescribed", &frame)?;
    bound_constraint_to_object(
        "fea.StructuralConstraint.prescribed",
        FEA_STRUCTURAL_CONSTRAINT_CLASS,
        geometry_id,
        geometry_revision,
        StructuralConstraint {
            constraint_id: id,
            frame,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target,
                translations,
                rotations,
            }),
        },
    )
}

#[runtime_builtin(
    name = "fea.StructuralConstraint.prescribedMotion",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::constraints::structural"
)]
pub async fn structural_prescribed_motion_builtin(
    id: String,
    region: Value,
    motions: Value,
) -> BuiltinResult<Value> {
    let builtin = "fea.StructuralConstraint.prescribedMotion";
    fixed_outputs(
        &runmat_builtins::FEA_STRUCTURAL_CONSTRAINT_PRESCRIBED_MOTION_DESCRIPTOR,
        builtin,
    )?;
    let (id, geometry_id, geometry_revision, target) = bound_region(builtin, id, &region)?;
    let translations = structural_motions_from_value(builtin, &motions)?;
    bound_constraint_to_object(
        builtin,
        FEA_STRUCTURAL_CONSTRAINT_CLASS,
        geometry_id,
        geometry_revision,
        StructuralConstraint {
            constraint_id: id,
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::PrescribedMotion(PrescribedStructuralMotion {
                target,
                translations,
            }),
        },
    )
}

macro_rules! coupling_builtin {
    ($function:ident, $descriptor:ident, $name:literal, $variant:ident) => {
        #[runtime_builtin(
                                    name = $name,
                                    binding_variant = "default",
                                    builtin_path = "crate::builtins::fea::constraints::structural"
                                )]
        pub async fn $function(
            id: String,
            region: Value,
            reference_point: Value,
        ) -> BuiltinResult<Value> {
            fixed_outputs(&runmat_builtins::$descriptor, $name)?;
            let (id, geometry_id, geometry_revision, target) = bound_region($name, id, &region)?;
            let reference_point = reference_point_from_value($name, &reference_point)?;
            bound_constraint_to_object(
                $name,
                FEA_STRUCTURAL_CONSTRAINT_CLASS,
                geometry_id,
                geometry_revision,
                StructuralConstraint {
                    constraint_id: id,
                    frame: CoordinateFrame::Global,
                    kind: StructuralConstraintKind::$variant(ReferencePointCoupling {
                        target,
                        reference_point,
                    }),
                },
            )
        }
    };
}

coupling_builtin!(
    structural_rigid_coupling_builtin,
    FEA_STRUCTURAL_CONSTRAINT_RIGID_COUPLING_DESCRIPTOR,
    "fea.StructuralConstraint.rigidCoupling",
    RigidCoupling
);
coupling_builtin!(
    structural_distributing_coupling_builtin,
    FEA_STRUCTURAL_CONSTRAINT_DISTRIBUTING_COUPLING_DESCRIPTOR,
    "fea.StructuralConstraint.distributingCoupling",
    DistributingCoupling
);

define_simple_constraint!(
    structural_symmetry_builtin,
    FEA_STRUCTURAL_CONSTRAINT_SYMMETRY_DESCRIPTOR,
    "fea.StructuralConstraint.symmetry",
    FEA_STRUCTURAL_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::structural",
    |constraint_id, target| StructuralConstraint {
        constraint_id,
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Symmetry { target },
    }
);

define_simple_constraint!(
    structural_antisymmetry_builtin,
    FEA_STRUCTURAL_CONSTRAINT_ANTISYMMETRY_DESCRIPTOR,
    "fea.StructuralConstraint.antisymmetry",
    FEA_STRUCTURAL_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::structural",
    |constraint_id, target| StructuralConstraint {
        constraint_id,
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Antisymmetry { target },
    }
);

define_quantity_constraint!(
    structural_normal_displacement_builtin,
    FEA_STRUCTURAL_CONSTRAINT_NORMAL_DISPLACEMENT_DESCRIPTOR,
    "fea.StructuralConstraint.normalDisplacement",
    FEA_STRUCTURAL_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::structural",
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
