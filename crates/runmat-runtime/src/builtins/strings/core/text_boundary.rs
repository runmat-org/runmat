//! Text-start, text-end, or either-boundary pattern construction.

use runmat_builtins::{TEXT_BOUNDARY_DESCRIPTOR, TEXT_BOUNDARY_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::common::{scalar_text_input, text_input_admission, TextInput};
use crate::builtins::strings::core::compat::{compat_error, pattern_object_in_context};
use crate::{gather_if_needed_async, BuiltinResult};

#[runtime_builtin(
    name = "textBoundary",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::core::text_boundary"
)]
pub(crate) async fn text_boundary_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &TEXT_BOUNDARY_DESCRIPTOR,
        &TEXT_BOUNDARY_ERROR_TOO_MANY_OUTPUTS,
        "textBoundary",
    )?;
    let boundary_owner = match (rest.len(), rest.first()) {
        (0, _) => None,
        (1, Some(value)) => Some(
            gather_if_needed_async(value)
                .await
                .map_err(|err| map_control_flow_with_builtin(err, "textBoundary"))?,
        ),
        _ => {
            return Err(compat_error(
                "textBoundary",
                "textBoundary: expected zero inputs or one boundary type",
            ))
        }
    };
    let boundary_type = if let Some(value) = boundary_owner.as_ref() {
        scalar_text_input(value, runtime.as_ref(), |runtime, cause| {
            text_input_admission("textBoundary", runtime, cause)
        })?
        .ok_or_else(|| compat_error("textBoundary", "textBoundary: expected a text scalar"))?
    } else {
        TextInput::Borrowed("either")
    };
    let regex = if boundary_type.as_str().eq_ignore_ascii_case("either") {
        r"(?:^|$)"
    } else if boundary_type.as_str().eq_ignore_ascii_case("start") {
        r"^"
    } else if boundary_type.as_str().eq_ignore_ascii_case("end") {
        r"$"
    } else {
        return Err(compat_error(
            "textBoundary",
            format!(
                "textBoundary: unsupported boundary type '{}'",
                boundary_type.as_str()
            ),
        ));
    };
    pattern_object_in_context(runtime.as_ref(), regex)
}
