//! MATLAB-compatible `gcf` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

use super::state::current_figure_handle;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

const GCF_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current figure handle.",
}];

const GCF_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const GCF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "fig = gcf()",
    inputs: &GCF_INPUTS_NONE,
    outputs: &GCF_OUTPUT_HANDLE,
}];

const GCF_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const GCF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GCF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GCF_ERRORS,
};

#[runtime_builtin(
    name = "gcf",
    category = "plotting",
    summary = "Return the handle of the current figure.",
    keywords = "gcf,figure,plotting",
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::gcf::GCF_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::gcf"
)]
pub fn gcf_builtin() -> crate::BuiltinResult<f64> {
    Ok(current_figure_handle().as_u32() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcf_descriptor_signature_present() {
        assert_eq!(GCF_DESCRIPTOR.signatures.len(), 1);
        assert_eq!(GCF_DESCRIPTOR.signatures[0].label, "fig = gcf()");
    }
}
