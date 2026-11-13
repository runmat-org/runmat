//! MATLAB-compatible `groot` builtin.

use runmat_builtins::{GROOT_DESCRIPTOR, GROOT_ERROR_TOO_MANY_OUTPUTS};

use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "groot",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::groot"
)]
pub fn groot_builtin() -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &GROOT_DESCRIPTOR,
        &GROOT_ERROR_TOO_MANY_OUTPUTS,
        "groot",
    )
    .map_err(super::coordinate_admission::terminal)?;
    Ok(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::GROOT_DESCRIPTOR;

    #[test]
    fn groot_descriptor_signature_present() {
        assert_eq!(GROOT_DESCRIPTOR.signatures.len(), 1);
        assert_eq!(GROOT_DESCRIPTOR.signatures[0].label, "root = groot()");
    }

    #[test]
    fn groot_returns_root_handle_zero() {
        assert_eq!(groot_builtin().unwrap(), 0.0);
    }
}
