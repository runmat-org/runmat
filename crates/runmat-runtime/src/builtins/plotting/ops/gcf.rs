//! Current figure lookup, ensuring the shared figure state exists.
use super::state::{current_figure_handle, select_figure};
use runmat_builtins::{GCF_DESCRIPTOR, GCF_ERROR_TOO_MANY_OUTPUTS};

#[runmat_macros::runtime_builtin(
    name = "gcf",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::gcf"
)]
pub fn gcf_builtin() -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &GCF_DESCRIPTOR,
        &GCF_ERROR_TOO_MANY_OUTPUTS,
        "gcf",
    )?;
    let handle = current_figure_handle();
    // Shared selection creates missing state and emits Created once, without
    // bypassing host observers or forcing pixel presentation.
    select_figure(handle);
    Ok(handle.as_u32() as f64)
}

#[cfg(test)]
#[path = "gcf/tests.rs"]
mod tests;
