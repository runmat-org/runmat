//! One platform boundary for the native TCP transport used by this family.

use crate::{
    context::{RuntimeContext, RuntimeTcpService},
    BuiltinResult,
};
use std::rc::Rc;

/// Resolve the installed session authority before an operand download or a
/// socket effect. A browser target retains its typed unsupported diagnostic.
pub(super) fn service(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
) -> BuiltinResult<Rc<dyn RuntimeTcpService>> {
    require_native_tcp(builtin)?;
    runtime
        .ok_or_else(|| {
            crate::context::RuntimeCapabilityError::new(
                crate::context::RuntimeCapability::Tcp,
                builtin,
            )
        })
        .map_err(crate::context::RuntimeCapabilityError::into_runtime_error)?
        .service_ports()
        .require_tcp(builtin)
        .cloned()
        .map_err(crate::context::RuntimeCapabilityError::into_runtime_error)
}

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
