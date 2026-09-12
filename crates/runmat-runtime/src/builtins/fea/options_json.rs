use crate::builtins::common::json::int_value_to_json;
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::fea::contracts::descriptors::{ERROR_INPUT, ERROR_INTERNAL};
use crate::builtins::fea::contracts::identities::{FEA_PAYLOAD_JSON_PROPERTY, MATERIAL_NAME};
use crate::builtins::fea::errors::{builtin_error, builtin_error_with_source};
use crate::builtins::fea::geometry::scalar_string;
use crate::BuiltinResult;
use runmat_value::NumericScalar;
use runmat_value::Value;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub(in crate::builtins::fea) struct NameValuePair<'a> {
    pub(in crate::builtins::fea) name: &'a Value,
    pub(in crate::builtins::fea) key: String,
    pub(in crate::builtins::fea) value: &'a Value,
}

pub(in crate::builtins::fea) fn expect_name_value_tail<'a>(
    builtin: &'static str,
    args: &'a [Value],
) -> BuiltinResult<Vec<NameValuePair<'a>>> {
    if !args.len().is_multiple_of(2) {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{builtin} options must be Name, Value pairs"),
        ));
    }
    args.chunks(2)
        .map(|pair| {
            let key = option_key(&pair[0], builtin)?;
            Ok(NameValuePair {
                name: &pair[0],
                key,
                value: &pair[1],
            })
        })
        .collect()
}

pub(in crate::builtins::fea) fn json_fields_from_name_values(
    builtin: &'static str,
    args: &[Value],
) -> BuiltinResult<serde_json::Map<String, serde_json::Value>> {
    let mut fields = serde_json::Map::new();
    for pair in expect_name_value_tail(builtin, args)? {
        let raw = scalar_string(pair.name, builtin, &ERROR_INPUT)?;
        let key = canonical_field_name(&raw);
        if fields
            .insert(key.clone(), value_to_json(builtin, pair.value)?)
            .is_some()
        {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("duplicate {builtin} option `{key}`"),
            ));
        }
    }
    Ok(fields)
}

pub(in crate::builtins::fea) fn option_key(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<String> {
    Ok(normalize_token(&scalar_string(
        value,
        builtin,
        &ERROR_INPUT,
    )?))
}

pub(in crate::builtins::fea) fn normalize_token(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

pub(in crate::builtins::fea) fn canonical_field_name(text: &str) -> String {
    let mut out = String::new();
    let mut previous_lower_or_digit = false;
    for ch in text.chars() {
        if ch == '-' || ch == ' ' {
            if !out.ends_with('_') && !out.is_empty() {
                out.push('_');
            }
            previous_lower_or_digit = false;
            continue;
        }
        if ch == '_' {
            if !out.ends_with('_') && !out.is_empty() {
                out.push('_');
            }
            previous_lower_or_digit = false;
            continue;
        }
        if ch.is_ascii_uppercase() {
            if previous_lower_or_digit && !out.ends_with('_') {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            previous_lower_or_digit = false;
        } else if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            previous_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }
    match normalize_token(&out).as_str() {
        "youngsmoduluspa" => "youngs_modulus_pa".to_string(),
        "poissonratio" => "poisson_ratio".to_string(),
        "density" | "densitykgperm3" => "density_kg_per_m3".to_string(),
        "magnitude" | "magnitudepa" => "magnitude_pa".to_string(),
        "current" | "currenta" => "current_a".to_string(),
        "phase" | "phaserad" => "phase_rad".to_string(),
        "specificimpedancepasperm" => "specific_impedance_pa_s_per_m".to_string(),
        "temperaturek" => "temperature_k".to_string(),
        "heatfluxwperm2" => "heat_flux_w_per_m2".to_string(),
        "ambienttemperaturek" => "ambient_temperature_k".to_string(),
        "coefficientwperm2k" => "coefficient_w_per_m2k".to_string(),
        "velocitympers" => "velocity_m_per_s".to_string(),
        "pressurepa" => "pressure_pa".to_string(),
        "amplitudescale" => "amplitude_scale".to_string(),
        "conductivitywpermk" => "conductivity_w_per_mk".to_string(),
        "specificheatjperkgk" => "specific_heat_j_per_kgk".to_string(),
        "conductivitysperm" => "conductivity_s_per_m".to_string(),
        "speedofsoundmpers" => "speed_of_sound_m_per_s".to_string(),
        "volumetricwperm3" => "volumetric_w_per_m3".to_string(),
        "inletvelocitympers" => "inlet_velocity_m_per_s".to_string(),
        "thermalconductancewperm2k" => "thermal_conductance_w_per_m2k".to_string(),
        "contactresistancem2kperw" => "contact_resistance_m2k_per_w".to_string(),
        "deterministicmode" => "deterministic_mode".to_string(),
        "precisionmode" => "precision_mode".to_string(),
        "preconditionermode" => "preconditioner_mode".to_string(),
        "qualitypolicy" => "quality_policy".to_string(),
        "prepcalibrationprofile" => "prep_calibration_profile".to_string(),
        "prepartifactid" => "prep_artifact_id".to_string(),
        "sweepfrequencyhz" => "sweep_frequency_hz".to_string(),
        "sweepenabled" => "sweep_enabled".to_string(),
        _ => out.trim_matches('_').to_string(),
    }
}

pub(in crate::builtins::fea) fn value_to_json(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<serde_json::Value> {
    match value {
        Value::Num(n) => json_number(builtin, *n),
        Value::Int(i) => Ok(int_value_to_json(i)),
        Value::Bool(b) => Ok(serde_json::Value::Bool(*b)),
        Value::String(s) => Ok(serde_json::Value::String(s.clone())),
        Value::CharArray(chars) if chars.rows == 1 => {
            Ok(serde_json::Value::String(chars.data.iter().collect()))
        }
        Value::StringArray(array) if array.data.len() == 1 => {
            Ok(serde_json::Value::String(array.data[0].clone()))
        }
        Value::StringArray(array) => Ok(serde_json::Value::Array(
            array
                .data
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect(),
        )),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => numeric_scalar_to_json(
            builtin,
            tensor
                .numeric_value_at(0)
                .expect("validated scalar tensor storage"),
        ),
        Value::Tensor(tensor) => Ok(serde_json::Value::Array(
            (0..tensor.len())
                .map(|index| {
                    numeric_scalar_to_json(
                        builtin,
                        tensor
                            .numeric_value_at(index)
                            .expect("validated tensor storage"),
                    )
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
        )),
        Value::Cell(cell) => Ok(serde_json::Value::Array(
            cell.data
                .iter()
                .map(|item| value_to_json(builtin, item))
                .collect::<BuiltinResult<Vec<_>>>()?,
        )),
        Value::Struct(fields) => {
            let mut object = serde_json::Map::new();
            for (key, value) in &fields.fields {
                object.insert(canonical_field_name(key), value_to_json(builtin, value)?);
            }
            Ok(serde_json::Value::Object(object))
        }
        Value::Object(object) => {
            if let Some(Value::String(json)) = object.properties.get(FEA_PAYLOAD_JSON_PROPERTY) {
                serde_json::from_str(json).map_err(|err| {
                    builtin_error_with_source(builtin, &ERROR_INPUT, err.to_string(), err)
                })
            } else {
                let mut object_json = serde_json::Map::new();
                for (key, value) in &object.properties {
                    if key.starts_with("__runmat_") {
                        continue;
                    }
                    object_json.insert(canonical_field_name(key), value_to_json(builtin, value)?);
                }
                Ok(serde_json::Value::Object(object_json))
            }
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("cannot convert value to FEA JSON payload: {other:?}"),
        )),
    }
}

pub(in crate::builtins::fea) fn numeric_scalar_to_json(
    builtin: &'static str,
    value: NumericScalar,
) -> BuiltinResult<serde_json::Value> {
    match value {
        NumericScalar::F64(value) => json_number(builtin, value),
        NumericScalar::F32(value) => json_number(builtin, f64::from(value)),
        value => Ok(int_value_to_json(
            &value
                .into_int_value()
                .expect("non-floating numeric scalar is integer"),
        )),
    }
}

pub(in crate::builtins::fea) fn json_number(
    builtin: &'static str,
    value: f64,
) -> BuiltinResult<serde_json::Value> {
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                "FEA numeric option values must be finite JSON numbers",
            )
        })
}

pub(in crate::builtins::fea) fn typed_json_with_overrides<T: Serialize + DeserializeOwned>(
    builtin: &'static str,
    default: T,
    fields: serde_json::Map<String, serde_json::Value>,
    label: &str,
) -> BuiltinResult<serde_json::Value> {
    let base = serde_json::to_value(default)
        .map_err(|err| builtin_error(builtin, &ERROR_INTERNAL, err.to_string()))?;
    let merged = json_with_overrides(builtin, base, fields, label)?;
    let typed: T = json_deserialize(builtin, merged, label)?;
    serde_json::to_value(typed)
        .map_err(|err| builtin_error_with_source(builtin, &ERROR_INTERNAL, err.to_string(), err))
}

pub(in crate::builtins::fea) fn json_with_overrides(
    builtin: &'static str,
    mut base: serde_json::Value,
    fields: serde_json::Map<String, serde_json::Value>,
    label: &str,
) -> BuiltinResult<serde_json::Value> {
    let Some(object) = base.as_object_mut() else {
        return Err(builtin_error(
            builtin,
            &ERROR_INTERNAL,
            format!("{label} default payload is not an object"),
        ));
    };
    for (key, value) in fields {
        if !object.contains_key(&key) {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("unsupported {label} option `{key}`"),
            ));
        }
        object.insert(key, value);
    }
    Ok(base)
}

pub(in crate::builtins::fea) fn json_deserialize<T: DeserializeOwned>(
    builtin: &'static str,
    value: serde_json::Value,
    label: &str,
) -> BuiltinResult<T> {
    serde_json::from_value(value)
        .map_err(|err| builtin_error(builtin, &ERROR_INPUT, format!("invalid {label}: {err}")))
}

pub(in crate::builtins::fea) fn typed_domain_data<T: DeserializeOwned + Serialize>(
    builtin: &'static str,
    label: &str,
    value: serde_json::Value,
) -> BuiltinResult<serde_json::Value> {
    let typed: T = json_deserialize(builtin, value, label)?;
    serde_json::to_value(typed)
        .map_err(|err| builtin_error_with_source(builtin, &ERROR_INTERNAL, err.to_string(), err))
}

pub(in crate::builtins::fea) fn json_to_string(value: serde_json::Value) -> BuiltinResult<String> {
    serde_json::from_value(value).map_err(|err| {
        builtin_error(
            MATERIAL_NAME,
            &ERROR_INPUT,
            format!("invalid string option: {err}"),
        )
    })
}

pub(in crate::builtins::fea) fn remove_required_f64(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    builtin: &'static str,
    key: &str,
) -> BuiltinResult<f64> {
    let Some(value) = fields.remove(key) else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("missing required option `{key}`"),
        ));
    };
    serde_json::from_value(value).map_err(|err| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("invalid numeric option `{key}`: {err}"),
        )
    })
}

pub(in crate::builtins::fea) fn remove_optional_f64(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    builtin: &'static str,
    key: &str,
) -> BuiltinResult<Option<f64>> {
    fields
        .remove(key)
        .map(|value| {
            serde_json::from_value(value).map_err(|err| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("invalid numeric option `{key}`: {err}"),
                )
            })
        })
        .transpose()
}

pub(in crate::builtins::fea) fn remove_required_vector3(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    builtin: &'static str,
    key: &str,
) -> BuiltinResult<[f64; 3]> {
    let Some(value) = fields.remove(key) else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("missing required vector option `{key}`"),
        ));
    };
    let values: Vec<f64> = serde_json::from_value(value).map_err(|err| {
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

pub(in crate::builtins::fea) fn move_known_fields(
    source: &mut serde_json::Map<String, serde_json::Value>,
    target: &mut serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> bool {
    let mut moved = false;
    for key in keys {
        if let Some(value) = source.remove(*key) {
            target.insert((*key).to_string(), value);
            moved = true;
        }
    }
    moved
}

pub(in crate::builtins::fea) fn reject_unknown_fields(
    builtin: &'static str,
    fields: serde_json::Map<String, serde_json::Value>,
) -> BuiltinResult<()> {
    if fields.is_empty() {
        return Ok(());
    }
    let keys = fields.keys().cloned().collect::<Vec<_>>().join(", ");
    Err(builtin_error(
        builtin,
        &ERROR_INPUT,
        format!("unsupported option field(s): {keys}"),
    ))
}
