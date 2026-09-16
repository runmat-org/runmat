//! Strongly typed material-domain targets and material assignment constructors.

use super::*;

const MATERIAL_ASSIGNMENT_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed beam, shell, solid, or cell property-region target.",
    },
    BuiltinParamDescriptor {
        name: "material",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.Material value.",
    },
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Expected-material and confidence options.",
    },
];
const SELECTED_TARGET_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "region",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Typed geometry.Region value.",
}];
const ALL_TARGET_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "geometry",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Typed geometry asset whose compatible domains are selected.",
}];

const MATERIAL_ASSIGNMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "assignment = fea.materialAssignment(target, material, Name, Value, ...)",
        inputs: &MATERIAL_ASSIGNMENT_INPUTS,
        outputs: &OUT_ANY,
    }];

macro_rules! target_descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:ident) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &$inputs,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

pub(super) const MATERIAL_ASSIGNMENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATERIAL_ASSIGNMENT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
target_descriptor!(
    BEAM_TARGET_DESCRIPTOR,
    BEAM_TARGET_SIGNATURES,
    "target = fea.PropertyRegionTarget.beam(region)",
    SELECTED_TARGET_INPUTS
);
target_descriptor!(
    SHELL_TARGET_DESCRIPTOR,
    SHELL_TARGET_SIGNATURES,
    "target = fea.PropertyRegionTarget.shell(region)",
    SELECTED_TARGET_INPUTS
);
target_descriptor!(
    SOLID_TARGET_DESCRIPTOR,
    SOLID_TARGET_SIGNATURES,
    "target = fea.PropertyRegionTarget.solid(region)",
    SELECTED_TARGET_INPUTS
);
target_descriptor!(
    CELL_TARGET_DESCRIPTOR,
    CELL_TARGET_SIGNATURES,
    "target = fea.PropertyRegionTarget.cell(region)",
    SELECTED_TARGET_INPUTS
);
target_descriptor!(
    ALL_BEAMS_TARGET_DESCRIPTOR,
    ALL_BEAMS_TARGET_SIGNATURES,
    "target = fea.PropertyRegionTarget.allBeams(geometry)",
    ALL_TARGET_INPUTS
);
target_descriptor!(
    ALL_SHELLS_TARGET_DESCRIPTOR,
    ALL_SHELLS_TARGET_SIGNATURES,
    "target = fea.PropertyRegionTarget.allShells(geometry)",
    ALL_TARGET_INPUTS
);
target_descriptor!(
    ALL_SOLIDS_TARGET_DESCRIPTOR,
    ALL_SOLIDS_TARGET_SIGNATURES,
    "target = fea.PropertyRegionTarget.allSolids(geometry)",
    ALL_TARGET_INPUTS
);
target_descriptor!(
    ALL_CELLS_TARGET_DESCRIPTOR,
    ALL_CELLS_TARGET_SIGNATURES,
    "target = fea.PropertyRegionTarget.allCells(geometry)",
    ALL_TARGET_INPUTS
);

fn selected_target(
    builtin: &'static str,
    region: &Value,
    constructor: fn(String) -> runmat_analysis_core::PropertyRegionTarget,
) -> BuiltinResult<Value> {
    create_property_region_target_object(builtin, region, constructor)
}

fn all_target(
    builtin: &'static str,
    geometry: &Value,
    target: runmat_analysis_core::PropertyRegionTarget,
) -> BuiltinResult<Value> {
    create_all_property_region_target_object(builtin, geometry, target)
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.beam",
    category = "fea",
    summary = "Bind a geometry region as a beam material domain.",
    keywords = "fea,property,material,target,beam,typed",
    descriptor(crate::builtins::fea::material_assignments::BEAM_TARGET_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_beam_builtin(region: Value) -> BuiltinResult<Value> {
    selected_target(
        "fea.PropertyRegionTarget.beam",
        &region,
        runmat_analysis_core::PropertyRegionTarget::beam,
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.shell",
    category = "fea",
    summary = "Bind a geometry region as a shell material domain.",
    keywords = "fea,property,material,target,shell,typed",
    descriptor(crate::builtins::fea::material_assignments::SHELL_TARGET_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_shell_builtin(region: Value) -> BuiltinResult<Value> {
    selected_target(
        "fea.PropertyRegionTarget.shell",
        &region,
        runmat_analysis_core::PropertyRegionTarget::shell,
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.solid",
    category = "fea",
    summary = "Bind a geometry region as a solid material domain.",
    keywords = "fea,property,material,target,solid,typed",
    descriptor(crate::builtins::fea::material_assignments::SOLID_TARGET_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_solid_builtin(region: Value) -> BuiltinResult<Value> {
    selected_target(
        "fea.PropertyRegionTarget.solid",
        &region,
        runmat_analysis_core::PropertyRegionTarget::solid,
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.cell",
    category = "fea",
    summary = "Bind a geometry region as a finite-volume cell material domain.",
    keywords = "fea,property,material,target,cell,typed",
    descriptor(crate::builtins::fea::material_assignments::CELL_TARGET_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_cell_builtin(region: Value) -> BuiltinResult<Value> {
    selected_target(
        "fea.PropertyRegionTarget.cell",
        &region,
        runmat_analysis_core::PropertyRegionTarget::cell,
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.allBeams",
    category = "fea",
    summary = "Bind every beam domain in one geometry to a material assignment.",
    keywords = "fea,property,material,target,beam,all,typed",
    descriptor(crate::builtins::fea::material_assignments::ALL_BEAMS_TARGET_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_all_beams_builtin(geometry: Value) -> BuiltinResult<Value> {
    all_target(
        "fea.PropertyRegionTarget.allBeams",
        &geometry,
        runmat_analysis_core::PropertyRegionTarget::all_beams(),
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.allShells",
    category = "fea",
    summary = "Bind every shell domain in one geometry to a material assignment.",
    keywords = "fea,property,material,target,shell,all,typed",
    descriptor(crate::builtins::fea::material_assignments::ALL_SHELLS_TARGET_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_all_shells_builtin(
    geometry: Value,
) -> BuiltinResult<Value> {
    all_target(
        "fea.PropertyRegionTarget.allShells",
        &geometry,
        runmat_analysis_core::PropertyRegionTarget::all_shells(),
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.allSolids",
    category = "fea",
    summary = "Bind every solid domain in one geometry to a material assignment.",
    keywords = "fea,property,material,target,solid,all,typed",
    descriptor(crate::builtins::fea::material_assignments::ALL_SOLIDS_TARGET_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_all_solids_builtin(
    geometry: Value,
) -> BuiltinResult<Value> {
    all_target(
        "fea.PropertyRegionTarget.allSolids",
        &geometry,
        runmat_analysis_core::PropertyRegionTarget::all_solids(),
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.allCells",
    category = "fea",
    summary = "Bind every finite-volume cell domain in one geometry to a material assignment.",
    keywords = "fea,property,material,target,cell,all,typed",
    descriptor(crate::builtins::fea::material_assignments::ALL_CELLS_TARGET_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_all_cells_builtin(geometry: Value) -> BuiltinResult<Value> {
    all_target(
        "fea.PropertyRegionTarget.allCells",
        &geometry,
        runmat_analysis_core::PropertyRegionTarget::all_cells(),
    )
}

#[runtime_builtin(
    name = "fea.materialAssignment",
    category = "fea",
    summary = "Create a typed FEA material assignment.",
    keywords = "fea,material,assignment,region",
    descriptor(crate::builtins::fea::material_assignments::MATERIAL_ASSIGNMENT_DESCRIPTOR),
    integer_audit(crate::builtins::fea::FEA_STRUCTURAL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_material_assignment_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    create_material_assignment_object_from_args(args)
}
