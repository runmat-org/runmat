//! MATLAB-compatible MEX target introspection.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "extension",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Platform-specific MEX filename extension without a leading dot.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "extension = mexext",
    inputs: &[],
    outputs: &OUTPUTS,
}];

const ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEX.UNAVAILABLE_TARGET",
    identifier: Some("RunMat:MEX:UnsupportedTarget"),
    when: "The current target cannot load native MEX modules.",
    message: "MEX modules are unavailable on this target.",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_UNAVAILABLE];

pub const MEXEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const MEXEXT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "mexext accepts no inputs and returns a platform extension string.",
};

#[runtime_builtin(
    name = "mexext",
    category = "interop/mex",
    summary = "Return the MEX filename extension for the current native platform.",
    keywords = "mexext,mex,native,extension,platform",
    descriptor(crate::builtins::interop::mex::MEXEXT_DESCRIPTOR),
    integer_audit(crate::builtins::interop::mex::MEXEXT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::mex"
)]
fn mexext_builtin() -> BuiltinResult<Value> {
    runmat_mex::mex_suffix().map(Value::from).ok_or_else(|| {
        build_runtime_error(ERROR_UNAVAILABLE.message)
            .with_builtin("mexext")
            .with_identifier(
                ERROR_UNAVAILABLE
                    .identifier
                    .expect("MEX unavailable error has an identifier"),
            )
            .build()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mexext_matches_the_adapter_target_suffix() {
        match runmat_mex::mex_suffix() {
            Some(expected) => assert_eq!(mexext_builtin().unwrap(), Value::from(expected)),
            None => assert_eq!(
                mexext_builtin()
                    .unwrap_err()
                    .identifier()
                    .expect("identifier"),
                "RunMat:MEX:UnsupportedTarget"
            ),
        }
    }
}
