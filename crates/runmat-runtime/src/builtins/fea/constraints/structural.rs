use super::support::{
    coordinate_frame_from_value, reference_point_from_value, structural_dof_values_from_value,
    structural_motions_from_value,
};
use super::*;
use crate::builtins::fea::values::DISPLACEMENT_CLASS;
use runmat_analysis_core::{PrescribedStructuralMotion, ReferencePointCoupling};

const PRESCRIBED_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Constraint identifier.",
    },
    BuiltinParamDescriptor {
        name: "region",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A geometry.Region selected from the model geometry.",
    },
    BuiltinParamDescriptor {
        name: "values",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One typed structural DOF value or a cell array of them.",
    },
    BuiltinParamDescriptor {
        name: "frame",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.CoordinateFrame value.",
    },
];
const PRESCRIBED_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.StructuralConstraint.prescribed(id, region, values, frame)",
    inputs: &PRESCRIBED_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const PRESCRIBED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PRESCRIBED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const PRESCRIBED_MOTION_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Constraint identifier.",
    },
    BuiltinParamDescriptor {
        name: "region",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A geometry.Region selected from the model geometry.",
    },
    BuiltinParamDescriptor {
        name: "motions",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One typed structural motion or a cell array of component motions.",
    },
];
const PRESCRIBED_MOTION_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "constraint = fea.StructuralConstraint.prescribedMotion(id, region, motions)",
        inputs: &PRESCRIBED_MOTION_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(super) const PRESCRIBED_MOTION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PRESCRIBED_MOTION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const COUPLING_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Constraint identifier.",
    },
    BuiltinParamDescriptor {
        name: "region",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "A geometry.Region selected from the model geometry.",
    },
    BuiltinParamDescriptor {
        name: "referencePoint",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.ReferencePoint value.",
    },
];
const COUPLING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "constraint = fea.StructuralConstraint.coupling(id, region, referencePoint)",
    inputs: &COUPLING_INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const COUPLING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COUPLING_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.StructuralConstraint.prescribed",
    category = "fea",
    summary = "Create a component-wise prescribed structural constraint in a typed frame.",
    keywords = "fea,structural,constraint,dof,frame,typed",
    descriptor(crate::builtins::fea::constraints::structural::PRESCRIBED_DESCRIPTOR),
    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::constraints::structural"
)]
pub async fn structural_prescribed_builtin(
    id: String,
    region: Value,
    values: Value,
    frame: Value,
) -> BuiltinResult<Value> {
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
    category = "fea",
    summary = "Prescribe derivative-consistent translational support motion on a region.",
    keywords = "fea,structural,constraint,support,motion,transient,typed",
    descriptor(crate::builtins::fea::constraints::structural::PRESCRIBED_MOTION_DESCRIPTOR),
    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::constraints::structural"
)]
pub async fn structural_prescribed_motion_builtin(
    id: String,
    region: Value,
    motions: Value,
) -> BuiltinResult<Value> {
    let builtin = "fea.StructuralConstraint.prescribedMotion";
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
    ($function:ident, $name:literal, $variant:ident) => {
        #[runtime_builtin(
                    name = $name,
                    category = "fea",
                    summary = "Create a typed structural reference-point coupling.",
                    keywords = "fea,structural,constraint,coupling,reference,point,typed",
                    descriptor(crate::builtins::fea::constraints::structural::COUPLING_DESCRIPTOR),
                    integer_audit(TYPED_CONSTRAINT_INTEGER_AUDIT),
                    builtin_path = "crate::builtins::fea::constraints::structural"
                )]
        pub async fn $function(
            id: String,
            region: Value,
            reference_point: Value,
        ) -> BuiltinResult<Value> {
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
    "fea.StructuralConstraint.rigidCoupling",
    RigidCoupling
);
coupling_builtin!(
    structural_distributing_coupling_builtin,
    "fea.StructuralConstraint.distributingCoupling",
    DistributingCoupling
);

define_simple_constraint!(
    structural_symmetry_builtin,
    STRUCTURAL_SYMMETRY_DESCRIPTOR,
    STRUCTURAL_SYMMETRY_SIGNATURES,
    "fea.StructuralConstraint.symmetry",
    "constraint = fea.StructuralConstraint.symmetry(id, region)",
    "Create a structural symmetry constraint.",
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
    STRUCTURAL_ANTISYMMETRY_DESCRIPTOR,
    STRUCTURAL_ANTISYMMETRY_SIGNATURES,
    "fea.StructuralConstraint.antisymmetry",
    "constraint = fea.StructuralConstraint.antisymmetry(id, region)",
    "Create a structural antisymmetry constraint.",
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
    STRUCTURAL_NORMAL_DISPLACEMENT_DESCRIPTOR,
    STRUCTURAL_NORMAL_DISPLACEMENT_SIGNATURES,
    STRUCTURAL_NORMAL_DISPLACEMENT_INPUTS,
    "fea.StructuralConstraint.normalDisplacement",
    "constraint = fea.StructuralConstraint.normalDisplacement(id, region, displacement)",
    "Create a prescribed normal-displacement constraint.",
    FEA_STRUCTURAL_CONSTRAINT_CLASS,
    "crate::builtins::fea::constraints::structural",
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
