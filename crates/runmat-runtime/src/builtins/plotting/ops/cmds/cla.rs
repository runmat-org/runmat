use crate::builtins::plotting::state::clear_current_axes;
use crate::builtins::plotting::type_resolvers::bool_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;
const CLA_OUTPUT_OK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when current axes are cleared.",
}];
const CLA_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const CLA_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ok = cla()",
    inputs: &CLA_INPUTS_NONE,
    outputs: &CLA_OUTPUT_OK,
}];
const CLA_ERRORS: [BuiltinErrorDescriptor; 0] = [];
pub const CLA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLA_ERRORS,
};

#[runtime_builtin(
    name = "cla",
    category = "plotting",
    summary = "Clear the current axes.",
    keywords = "cla,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::cmds::CLA_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn cla_builtin(_args: Vec<Value>) -> crate::BuiltinResult<bool> {
    clear_current_axes();
    Ok(true)
}
