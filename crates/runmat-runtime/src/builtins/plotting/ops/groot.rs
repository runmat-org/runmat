//! MATLAB-compatible `groot` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const GROOT_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "root",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Root graphics object handle.",
}];

const GROOT_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const GROOT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "root = groot()",
    inputs: &GROOT_INPUTS_NONE,
    outputs: &GROOT_OUTPUT_HANDLE,
}];

const GROOT_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const GROOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GROOT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GROOT_ERRORS,
};

#[runtime_builtin(
    name = "groot",
    category = "plotting",
    summary = "Return the root graphics object handle.",
    keywords = "groot,root,graphics,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::groot::GROOT_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::groot"
)]
pub fn groot_builtin() -> crate::BuiltinResult<f64> {
    Ok(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

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
