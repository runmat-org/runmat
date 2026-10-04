//! One platform boundary for the native TCP transport used by this family.

use crate::BuiltinResult;

pub(super) fn require_native_tcp(builtin: &'static str) -> BuiltinResult<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = builtin;
        Ok(())
    }
    #[cfg(target_arch = "wasm32")]
    {
        let error = &runmat_builtins::NET_ERROR_UNSUPPORTED_TARGET;
        Err(
            crate::build_runtime_error(format!("{builtin}: {}", error.message))
                .with_builtin(builtin)
                .with_identifier(error.identifier.expect("declared TCP target identifier"))
                .build(),
        )
    }
}
