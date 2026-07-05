use runmat_analysis_fea::ComputeBackend;
use runmat_builtins::Value;

use crate::analysis::{
    analysis_author_study_op, AnalysisCreateModelProfile, AnalysisRunKind,
    AnalysisStudyAuthoringIntent,
};
use crate::operations::OperationContext;
use crate::BuiltinResult;

use super::{
    builtin_error, builtin_error_with_source, geometry_asset_from_value_with_builtin,
    json_deserialize, operation_error, option_key, parse_scalar_enum, scalar_string,
    study_to_object, value_to_json, AUTHOR_STUDY_NAME, ERROR_INPUT, ERROR_OPERATION,
};

pub(super) fn create_author_study_object_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() < 3 {
        return Err(builtin_error(
            AUTHOR_STUDY_NAME,
            &ERROR_INPUT,
            "fea.authorStudy requires id, geometry, and mesh authoring summary arguments",
        ));
    }
    if !(args.len() - 3).is_multiple_of(2) {
        return Err(builtin_error(
            AUTHOR_STUDY_NAME,
            &ERROR_INPUT,
            "fea.authorStudy options must be Name, Value pairs",
        ));
    }

    let study_id = scalar_string(&args[0], AUTHOR_STUDY_NAME, &ERROR_INPUT)?;
    let geometry = geometry_asset_from_value_with_builtin(&args[1], AUTHOR_STUDY_NAME)?;
    let mesh_authoring_summary = mesh_authoring_summary_from_value(&args[2])?;
    let mut profile = AnalysisCreateModelProfile::LinearStaticStructural;
    let mut run_kind = AnalysisRunKind::LinearStatic;
    let mut backend = ComputeBackend::Cpu;
    let mut model_id = None::<String>;
    let mut material_region_id = None::<String>;
    let mut fixed_boundary_region_id = None::<String>;
    let mut load_boundary_region_id = None::<String>;
    let mut force_n = None::<[f64; 3]>;

    for pair in args[3..].chunks(2) {
        let key = option_key(&pair[0], AUTHOR_STUDY_NAME)?;
        match key.as_str() {
            "profile" => {
                let text = scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?;
                profile = parse_scalar_enum(&text, "Profile")?;
            }
            "runkind" | "kind" => {
                let text = scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?;
                run_kind = parse_scalar_enum(&text, "RunKind")?;
            }
            "backend" => {
                let text = scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?;
                backend = parse_scalar_enum(&text, "Backend")?;
            }
            "modelid" => {
                model_id = Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "materialregion" | "materialregionid" => {
                material_region_id =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "fixedregion" | "fixedregionid" | "fixedboundaryregion" | "fixedboundaryregionid" => {
                fixed_boundary_region_id =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "loadregion" | "loadregionid" | "loadboundaryregion" | "loadboundaryregionid" => {
                load_boundary_region_id =
                    Some(scalar_string(&pair[1], AUTHOR_STUDY_NAME, &ERROR_INPUT)?);
            }
            "forcen" | "force" | "forcevector" => {
                force_n = Some(vector3_from_value(AUTHOR_STUDY_NAME, &pair[1], "ForceN")?);
            }
            other => {
                return Err(builtin_error(
                    AUTHOR_STUDY_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.authorStudy option `{other}`"),
                ));
            }
        }
    }

    let authored = analysis_author_study_op(
        AnalysisStudyAuthoringIntent {
            study_id,
            model_id,
            geometry,
            mesh_authoring_summary,
            profile,
            run_kind,
            backend,
            material_region_id,
            fixed_boundary_region_id,
            load_boundary_region_id,
            force_n,
        },
        OperationContext::new(None, None),
    )
    .map_err(|err| operation_error(AUTHOR_STUDY_NAME, &ERROR_OPERATION, err))?;

    study_to_object(authored.data.study)
}

fn mesh_authoring_summary_from_value(
    value: &Value,
) -> BuiltinResult<runmat_meshing_evidence::MeshAuthoringSummary> {
    if let Ok(text) = scalar_string(value, AUTHOR_STUDY_NAME, &ERROR_INPUT) {
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
            builtin_error_with_source(AUTHOR_STUDY_NAME, &ERROR_INPUT, err.to_string(), err)
        })?;
        return mesh_authoring_summary_from_json(json);
    }

    let json = value_to_json(AUTHOR_STUDY_NAME, value)?;
    mesh_authoring_summary_from_json(json)
}

fn mesh_authoring_summary_from_json(
    json: serde_json::Value,
) -> BuiltinResult<runmat_meshing_evidence::MeshAuthoringSummary> {
    let mut summary_json = json.get("mesh_authoring_summary").cloned().unwrap_or(json);
    normalize_mesh_authoring_summary_json(&mut summary_json);
    json_deserialize(AUTHOR_STUDY_NAME, summary_json, "mesh authoring summary")
}

fn normalize_mesh_authoring_summary_json(value: &mut serde_json::Value) {
    normalize_integral_json_numbers(value);
    wrap_scalar_string_arrays(value);
}

fn normalize_integral_json_numbers(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                normalize_integral_json_numbers(value);
            }
        }
        serde_json::Value::Object(fields) => {
            for value in fields.values_mut() {
                normalize_integral_json_numbers(value);
            }
        }
        serde_json::Value::Number(number) => {
            let Some(value) = number.as_f64() else {
                return;
            };
            if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
                return;
            }
            if value > u64::MAX as f64 {
                return;
            }
            *number = serde_json::Number::from(value as u64);
        }
        _ => {}
    }
}

fn wrap_scalar_string_arrays(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                wrap_scalar_string_arrays(value);
            }
        }
        serde_json::Value::Object(fields) => {
            for key in [
                "required_material_region_ids",
                "missing_required_material_region_ids",
                "required_boundary_region_ids",
                "missing_required_boundary_region_ids",
            ] {
                if let Some(serde_json::Value::String(region_id)) = fields.get(key).cloned() {
                    fields.insert(
                        key.to_string(),
                        serde_json::Value::Array(vec![serde_json::Value::String(region_id)]),
                    );
                }
            }
            for value in fields.values_mut() {
                wrap_scalar_string_arrays(value);
            }
        }
        _ => {}
    }
}

fn vector3_from_value(builtin: &'static str, value: &Value, key: &str) -> BuiltinResult<[f64; 3]> {
    let values: Vec<f64> =
        serde_json::from_value(value_to_json(builtin, value)?).map_err(|err| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("invalid vector option `{key}`: {err}"),
            )
        })?;
    if values.len() != 3 {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("vector option `{key}` must contain exactly 3 values"),
        ));
    }
    Ok([values[0], values[1], values[2]])
}
