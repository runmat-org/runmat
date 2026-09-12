use crate::analysis::AnalysisCreateModelProfile;
use crate::analysis::AnalysisRunKind;
use crate::builtins::fea::contracts::descriptors::ERROR_INPUT;
use crate::builtins::fea::contracts::identities::STUDY_NAME;
use crate::builtins::fea::errors::{builtin_error, builtin_error_with_source};
use crate::builtins::fea::study::StudyConstructorOptions;
use crate::builtins::geometry::GEOMETRY_ASSET_CLASS;
use crate::builtins::geometry::GEOMETRY_ASSET_JSON_PROPERTY;
use crate::BuiltinResult;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_geometry_core::GeometryAsset;
use runmat_value::ObjectInstance;
use runmat_value::Value;
use serde::de::DeserializeOwned;

pub(in crate::builtins::fea) fn geometry_asset_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<GeometryAsset> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{builtin} geometry must be {GEOMETRY_ASSET_CLASS}"),
        ));
    };
    if object.class_name != GEOMETRY_ASSET_CLASS {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "{builtin} geometry must be {GEOMETRY_ASSET_CLASS}, got {}",
                object.class_name
            ),
        ));
    }
    object_json_property(builtin, object, GEOMETRY_ASSET_JSON_PROPERTY, &ERROR_INPUT)
}

pub(in crate::builtins::fea) fn geometry_asset_from_value_with_builtin(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<GeometryAsset> {
    geometry_asset_from_value(builtin, value)
}

pub(in crate::builtins::fea) fn object_json_property<T: DeserializeOwned>(
    builtin: &'static str,
    object: &ObjectInstance,
    property: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<T> {
    let Some(Value::String(json)) = object.properties.get(property) else {
        return Err(builtin_error(
            builtin,
            error,
            format!(
                "{} is missing required runtime payload property `{property}`",
                object.class_name
            ),
        ));
    };
    serde_json::from_str(json)
        .map_err(|err| builtin_error_with_source(builtin, error, err.to_string(), err))
}

pub(in crate::builtins::fea) fn scalar_string(
    value: &Value,
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        _ => Err(builtin_error(
            builtin,
            error,
            format!("{builtin} expected a text scalar"),
        )),
    }
}

pub(in crate::builtins::fea) fn parse_scalar_enum<T: DeserializeOwned>(
    text: &str,
    label: &str,
) -> BuiltinResult<T> {
    parse_scalar_enum_for_builtin(STUDY_NAME, text, label)
}

pub(in crate::builtins::fea) fn parse_scalar_enum_for_builtin<T: DeserializeOwned>(
    builtin: &'static str,
    text: &str,
    label: &str,
) -> BuiltinResult<T> {
    serde_yaml::from_str::<T>(&text.to_ascii_lowercase()).map_err(|err| {
        builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("invalid {label} value `{text}`: {err}"),
        )
    })
}

pub(in crate::builtins::fea) fn resolve_study_profile_and_run_kind(
    options: &StudyConstructorOptions,
) -> BuiltinResult<(AnalysisCreateModelProfile, AnalysisRunKind)> {
    let profile = options.profile.ok_or_else(|| {
        builtin_error(
            STUDY_NAME,
            &ERROR_INPUT,
            "fea.study requires Profile; choose a physics profile from fea.capabilities().physicsProfiles",
        )
    })?;
    let run_kind = profile.derived_run_kind();
    if let Some(explicit_run_kind) = options.run_kind {
        if explicit_run_kind != run_kind {
            return Err(builtin_error(
                STUDY_NAME,
                &ERROR_INPUT,
                format!(
                    "explicit solver {} does not match Profile {}; omit RunKind or choose a matching Profile",
                    explicit_run_kind.as_snake_case(),
                    profile.as_snake_case()
                ),
            ));
        }
    }
    Ok((profile, run_kind))
}
