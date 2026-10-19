//! Text-start, text-end, or either-boundary pattern construction.

use runmat_builtins::{TEXT_BOUNDARY_DESCRIPTOR, TEXT_BOUNDARY_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::core::compat::{compat_error, pattern_object, scalar_text};
use crate::{gather_if_needed_async, BuiltinResult};

#[runtime_builtin(
    name = "textBoundary",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::text_boundary"
)]
pub(crate) async fn text_boundary_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &TEXT_BOUNDARY_DESCRIPTOR,
        &TEXT_BOUNDARY_ERROR_TOO_MANY_OUTPUTS,
        "textBoundary",
    )?;
    let boundary_type = match rest.as_slice() {
        [] => "either".to_string(),
        [value] => {
            let value = gather_if_needed_async(value)
                .await
                .map_err(|err| map_control_flow_with_builtin(err, "textBoundary"))?;
            scalar_text(&value, "textBoundary")?
        }
        _ => {
            return Err(compat_error(
                "textBoundary",
                "textBoundary: expected zero inputs or one boundary type",
            ))
        }
    };
    let regex = match boundary_type.to_ascii_lowercase().as_str() {
        "either" => r"(?:^|$)",
        "start" => r"^",
        "end" => r"$",
        other => {
            return Err(compat_error(
                "textBoundary",
                format!("textBoundary: unsupported boundary type '{other}'"),
            ))
        }
    };
    Ok(pattern_object(regex))
}
