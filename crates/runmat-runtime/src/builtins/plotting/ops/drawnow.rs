//! MATLAB-compatible `drawnow` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use crate::builtins::plotting::type_resolvers::bool_type;
use crate::BuiltinResult;

const DRAWNOW_OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when pending graphics updates were flushed (or no-op completed).",
}];

const DRAWNOW_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const DRAWNOW_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ok = drawnow()",
    inputs: &DRAWNOW_INPUTS_NONE,
    outputs: &DRAWNOW_OUTPUT_STATUS,
}];

const DRAWNOW_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DRAWNOW.INTERNAL",
    identifier: Some("RunMat:drawnow:Internal"),
    when: "Rendering flush fails in the active plotting backend.",
    message: "drawnow: internal operation failed",
};

const DRAWNOW_ERRORS: [BuiltinErrorDescriptor; 1] = [DRAWNOW_ERROR_INTERNAL];

pub const DRAWNOW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DRAWNOW_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DRAWNOW_ERRORS,
};

/// Flush pending figure updates to any bound plot surfaces.
///
/// On Web/WASM/RunMat Desktop, this presents the current figure revision to any bound surfaces and yields so
/// the browser can process rendering work. On other targets, this is a no-op.
#[runtime_builtin(
    name = "drawnow",
    category = "plotting",
    summary = "Flush pending graphics updates.",
    keywords = "drawnow,graphics,flush,plot",
    sink = true,
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::drawnow::DRAWNOW_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::drawnow"
)]
pub async fn drawnow_builtin() -> BuiltinResult<bool> {
    #[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
    {
        use crate::builtins::plotting;
        let handle = plotting::current_figure_handle();
        plotting::render_current_scene(handle.as_u32()).map_err(|e| {
            crate::build_runtime_error(format!("drawnow: {e}"))
                .with_builtin("drawnow")
                .build()
        })?;
        Ok(true)
    }

    #[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
    {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn drawnow_descriptor_signature_present() {
        assert_eq!(DRAWNOW_DESCRIPTOR.signatures.len(), 1);
        assert_eq!(DRAWNOW_DESCRIPTOR.signatures[0].label, "ok = drawnow()");
    }

    #[test]
    fn drawnow_returns_true() {
        let result = block_on(drawnow_builtin()).unwrap();
        assert!(result);
    }
}
