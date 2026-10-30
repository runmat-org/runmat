//! Character-property classification, including exact numeric character codes.

use runmat_builtins::{
    CharacterPropertyCategory, ISSTRPROP_DESCRIPTOR, ISSTRPROP_ERROR_TOO_MANY_OUTPUTS,
    ISSTRPROP_RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::common::{scalar_text_input, text_input_admission};
use crate::builtins::strings::core::classification::{
    char_matches_prop, classify_text_or_numeric_value, parse_force_cell_output,
};
use crate::builtins::strings::core::compat::compat_error;
use crate::{gather_if_needed_async, make_cell_with_shape, BuiltinResult};

#[runtime_builtin(
    name = "isstrprop",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::core::isstrprop"
)]
pub(crate) async fn isstrprop_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    text: Value,
    prop: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &ISSTRPROP_DESCRIPTOR,
        &ISSTRPROP_ERROR_TOO_MANY_OUTPUTS,
        "isstrprop",
    )?;
    if !matches!(
        prop,
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_)
    ) {
        return Err(compat_error(
            "isstrprop",
            "isstrprop: category must be a character vector or string scalar",
        ));
    }
    if matches!(&text, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ISSTRPROP_RESIDENT_INPUT_EXTENSION,
            "isstrprop",
        )?;
    }
    let prop = gather_if_needed_async(&prop)
        .await
        .map_err(|err| map_control_flow_with_builtin(err, "isstrprop"))?;
    let prop = scalar_text_input(&prop, runtime.as_ref(), |runtime, cause| {
        text_input_admission("isstrprop", runtime, cause)
    })?
    .ok_or_else(|| compat_error("isstrprop", "isstrprop: expected a text category"))?;
    let category = CharacterPropertyCategory::parse(prop.as_str()).ok_or_else(|| {
        compat_error(
            "isstrprop",
            format!("isstrprop: unknown character category '{}'", prop.as_str()),
        )
    })?;
    let force_cell_output = parse_force_cell_output(runtime.as_ref(), rest.view())?;
    // Category and option values are host-only. Reject known invalid controls
    // before a selected resident input incurs provider work.
    let text = match &text {
        Value::GpuTensor(handle) => {
            let readback =
                runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
                    let error = compat_error(
                        "isstrprop",
                        "isstrprop: no provider owns the resident input",
                    );
                    crate::runtime_error::with_provider_execution_source(error, source.into())
                })?;
            crate::builtins::common::gpu_helpers::download_readback_value_in_context(
                runtime.as_ref(),
                &readback,
            )
            .await?
        }
        _ => text,
    };
    let result =
        classify_text_or_numeric_value(text, "isstrprop", |ch| char_matches_prop(ch, category))?;
    if force_cell_output && !matches!(result, Value::Cell(_)) {
        make_cell_with_shape(vec![result], vec![1, 1]).map_err(|e| compat_error("isstrprop", e))
    } else {
        Ok(result)
    }
}

#[cfg(test)]
#[path = "isstrprop/pre_effect_tests.rs"]
mod pre_effect_tests;

#[cfg(test)]
#[path = "isstrprop/flag_carrier_tests.rs"]
mod flag_carrier_tests;
