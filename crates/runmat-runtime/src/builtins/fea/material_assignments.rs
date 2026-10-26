//! Strongly typed material-domain targets and material assignment constructors.

use super::*;

/// Expose the binding revision with the geometry revision type (`uint32`),
/// like every other geometry-bound FEA model object.
fn with_uint32_revision(builtin: &'static str, value: Value) -> BuiltinResult<Value> {
    let Value::Object(mut object) = value else {
        return Ok(value);
    };
    let revision = match object.properties().get("geometry_revision") {
        Some(Value::Int(revision)) => revision.to_i64(),
        Some(Value::Num(revision)) if revision.fract() == 0.0 => *revision as i64,
        _ => return Ok(Value::Object(object)),
    };
    let revision = u32::try_from(revision).map_err(|_| {
        builtin_error(
            builtin,
            &ERROR_INTERNAL,
            "geometry revision does not fit the geometry revision type",
        )
    })?;
    crate::builtins::fea::object_api::replace(
        builtin,
        &mut object,
        "geometry_revision",
        Value::Int(IntValue::U32(revision)),
    )?;
    Ok(Value::Object(object))
}

fn selected_target(
    builtin: &'static str,
    region: &Value,
    constructor: fn(String) -> runmat_analysis_core::PropertyRegionTarget,
) -> BuiltinResult<Value> {
    with_uint32_revision(
        builtin,
        create_property_region_target_object(builtin, region, constructor)?,
    )
}

fn all_target(
    builtin: &'static str,
    geometry: &Value,
    target: runmat_analysis_core::PropertyRegionTarget,
) -> BuiltinResult<Value> {
    with_uint32_revision(
        builtin,
        create_all_property_region_target_object(builtin, geometry, target)?,
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.beam",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_beam_builtin(region: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PROPERTY_REGION_TARGET_BEAM_DESCRIPTOR,
        "fea.PropertyRegionTarget.beam",
    )?;
    selected_target(
        "fea.PropertyRegionTarget.beam",
        &region,
        runmat_analysis_core::PropertyRegionTarget::beam,
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.shell",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_shell_builtin(region: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PROPERTY_REGION_TARGET_SHELL_DESCRIPTOR,
        "fea.PropertyRegionTarget.shell",
    )?;
    selected_target(
        "fea.PropertyRegionTarget.shell",
        &region,
        runmat_analysis_core::PropertyRegionTarget::shell,
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.solid",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_solid_builtin(region: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PROPERTY_REGION_TARGET_SOLID_DESCRIPTOR,
        "fea.PropertyRegionTarget.solid",
    )?;
    selected_target(
        "fea.PropertyRegionTarget.solid",
        &region,
        runmat_analysis_core::PropertyRegionTarget::solid,
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.cell",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_cell_builtin(region: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PROPERTY_REGION_TARGET_CELL_DESCRIPTOR,
        "fea.PropertyRegionTarget.cell",
    )?;
    selected_target(
        "fea.PropertyRegionTarget.cell",
        &region,
        runmat_analysis_core::PropertyRegionTarget::cell,
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.allBeams",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_all_beams_builtin(geometry: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PROPERTY_REGION_TARGET_ALL_BEAMS_DESCRIPTOR,
        "fea.PropertyRegionTarget.allBeams",
    )?;
    all_target(
        "fea.PropertyRegionTarget.allBeams",
        &geometry,
        runmat_analysis_core::PropertyRegionTarget::all_beams(),
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.allShells",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_all_shells_builtin(
    geometry: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PROPERTY_REGION_TARGET_ALL_SHELLS_DESCRIPTOR,
        "fea.PropertyRegionTarget.allShells",
    )?;
    all_target(
        "fea.PropertyRegionTarget.allShells",
        &geometry,
        runmat_analysis_core::PropertyRegionTarget::all_shells(),
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.allSolids",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_all_solids_builtin(
    geometry: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PROPERTY_REGION_TARGET_ALL_SOLIDS_DESCRIPTOR,
        "fea.PropertyRegionTarget.allSolids",
    )?;
    all_target(
        "fea.PropertyRegionTarget.allSolids",
        &geometry,
        runmat_analysis_core::PropertyRegionTarget::all_solids(),
    )
}

#[runtime_builtin(
    name = "fea.PropertyRegionTarget.allCells",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_property_region_target_all_cells_builtin(geometry: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_PROPERTY_REGION_TARGET_ALL_CELLS_DESCRIPTOR,
        "fea.PropertyRegionTarget.allCells",
    )?;
    all_target(
        "fea.PropertyRegionTarget.allCells",
        &geometry,
        runmat_analysis_core::PropertyRegionTarget::all_cells(),
    )
}

#[runtime_builtin(
    name = "fea.materialAssignment",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::material_assignments"
)]
pub async fn fea_material_assignment_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    create_material_assignment_object_from_args(args)
}
