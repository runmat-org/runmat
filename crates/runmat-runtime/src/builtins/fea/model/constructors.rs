use crate::builtins::fea::contracts::descriptors::{ERROR_INPUT, ERROR_INTERNAL};
use crate::builtins::fea::contracts::identities::{
    FEA_MODEL_CLASS, FEA_PAYLOAD_JSON_PROPERTY, MODEL_NAME,
};
use crate::builtins::fea::errors::builtin_error;
use crate::builtins::fea::geometry::{geometry_asset_from_value, parse_scalar_enum, scalar_string};
use crate::builtins::fea::integer_serialization::serializable_to_object_preserving_integers;
use crate::builtins::fea::model::assembly::build_model_from_parts;
use crate::builtins::fea::model::codec::{
    boundary_condition_vec_from_value, domain_vec_from_value, interface_vec_from_value,
    load_case_vec_from_value, material_assignment_vec_from_value, material_vec_from_value,
    step_vec_from_value,
};
use crate::builtins::fea::options_json::expect_name_value_tail;
use crate::builtins::fea::study::ModelConstructorOptions;
use crate::builtins::fea::value_decode::parse_model_defaults_mode;
use crate::BuiltinResult;
use runmat_value::Value;

pub(in crate::builtins::fea) fn create_model_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            "fea.model requires id and geometry arguments",
        ));
    }
    let model_id = scalar_string(&args[0], MODEL_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value(MODEL_NAME, &args[1])?;
    let options = parse_model_constructor_options(MODEL_NAME, &args[2..])?;
    let profile = options.profile.ok_or_else(|| {
        builtin_error(
            MODEL_NAME,
            &ERROR_INPUT,
            "fea.model requires Profile; choose a physics profile from fea.capabilities().physicsProfiles",
        )
    })?;
    let model = build_model_from_parts(
        MODEL_NAME,
        &geometry,
        model_id,
        profile,
        options.defaults,
        options.frame,
        options.materials,
        options.material_assignments,
        options.boundary_conditions,
        options.loads,
        options.steps,
        options.domains,
        options.interfaces,
    )?;
    serializable_to_object_preserving_integers(
        MODEL_NAME,
        &ERROR_INTERNAL,
        FEA_MODEL_CLASS,
        &model,
        Some(FEA_PAYLOAD_JSON_PROPERTY),
        &[],
        &["geometry_revision", "revision"],
    )
}

pub(in crate::builtins::fea) fn parse_model_constructor_options(
    builtin: &'static str,
    args: &[Value],
) -> BuiltinResult<ModelConstructorOptions> {
    let mut options = ModelConstructorOptions::default();
    for pair in expect_name_value_tail(builtin, args)? {
        match pair.key.as_str() {
            "profile" => {
                let text = scalar_string(pair.value, builtin, &ERROR_INPUT)?;
                options.profile = Some(parse_scalar_enum(&text, "Profile")?);
            }
            "frame" => {
                let text = scalar_string(pair.value, builtin, &ERROR_INPUT)?;
                options.frame = Some(parse_scalar_enum(&text, "Frame")?);
            }
            "defaults" => {
                options.defaults =
                    parse_model_defaults_mode(&scalar_string(pair.value, builtin, &ERROR_INPUT)?)?;
            }
            "materials" => options.materials = material_vec_from_value(builtin, pair.value)?,
            "materialassignments" | "assignments" => {
                options.material_assignments =
                    material_assignment_vec_from_value(builtin, pair.value)?;
            }
            "boundaryconditions" | "bcs" => {
                options.boundary_conditions =
                    boundary_condition_vec_from_value(builtin, pair.value)?;
            }
            "loads" | "loadcases" => options.loads = load_case_vec_from_value(builtin, pair.value)?,
            "steps" => options.steps = step_vec_from_value(builtin, pair.value)?,
            "domains" => options.domains = domain_vec_from_value(builtin, pair.value)?,
            "interfaces" => options.interfaces = interface_vec_from_value(builtin, pair.value)?,
            other => {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("unsupported {builtin} option `{other}`"),
                ));
            }
        }
    }
    Ok(options)
}
