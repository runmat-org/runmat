//! Properties of registered colorbars, backed by their shared scene owner.
use super::super::state::ColorBarHandleState;
use super::*;

pub(super) fn get(
    bar: &ColorBarHandleState,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match property.map(canonical_property_name).as_deref() {
        None => {
            let mut value = runmat_value::UnboundedStructValueBuilder::new();
            value.insert("Type", Value::String("colorbar".into()));
            value.insert("Parent", Value::Num(bar.figure.as_u32() as f64));
            value.insert("Children", handles_value(Vec::new()));
            value.insert("Visible", Value::Bool(bar.visible));
            Ok(Value::Struct(value.finish()))
        }
        Some("type") => Ok(Value::String("colorbar".into())),
        Some("parent") => Ok(Value::Num(bar.figure.as_u32() as f64)),
        Some("children") => Ok(handles_value(Vec::new())),
        Some("visible") => Ok(Value::Bool(bar.visible)),
        Some(other) => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported colorbar property `{other}`"),
        )),
    }
}

pub(super) fn set(
    authority: &super::super::state::PropertyMutationTarget,
    handle: f64,
    key: &str,
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match key {
        "visible" => {
            let visible = value_as_bool(value).ok_or_else(|| {
                plotting_error(
                    builtin,
                    format!("{builtin}: Visible expects an on/off logical value"),
                )
            })?;
            super::super::state::set_colorbar_visible_for_property(authority, handle, visible)
                .map_err(|error| map_figure_error(builtin, error))
        }
        other => Err(plotting_error(
            builtin,
            format!("{builtin}: unsupported or read-only colorbar property `{other}`"),
        )),
    }
}
