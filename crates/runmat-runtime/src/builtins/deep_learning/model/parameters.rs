//! One canonical network parameter authority in `Layers`.
//!
//! The public Learnables table is a derived projection. Typed keys, borrowed
//! admission, presentation, and atomic replacement live in bounded modules.
use super::*;

mod projection;
mod precision;
mod schema;
mod tape_values;
mod update;
mod validation;
mod view;

pub(in crate::builtins::deep_learning) use precision::floating_precision;
pub(in crate::builtins::deep_learning) use schema::{
    ParameterKey, ParameterRole, ParameterShape, ParameterView,
};
pub(in crate::builtins::deep_learning) use tape_values::install_annotated;
pub(in crate::builtins::deep_learning) use validation::numeric_payload_shape;
use validation::{empty_parameter, parameter_data, validate_parameter};
pub(in crate::builtins::deep_learning) use view::{view, view_layers};

pub(in crate::builtins) use projection::project;
pub(super) use projection::public_value;
pub(in crate::builtins) use update::apply;
pub(in crate::builtins::deep_learning) use update::{apply_updates, ParameterUpdate};
pub(in crate::builtins) use view::validate_matching_networks;

pub(in crate::builtins::deep_learning) fn validate_initialized_parameter(
    value: &Value,
    expected: &ParameterShape,
    role: ParameterRole,
    function: &'static str,
) -> BuiltinResult<()> {
    validate_parameter(value, expected, role, function)
}

pub(super) fn all_filled(network: &ObjectInstance, function: &'static str) -> BuiltinResult<bool> {
    for parameter in view(network, function)? {
        let Some(value) = parameter.value else {
            return Ok(false);
        };
        if empty_parameter(parameter_data(value, function)?) {
            return Ok(false);
        }
    }
    Ok(true)
}
