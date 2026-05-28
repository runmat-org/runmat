//! MATLAB-compatible `clc` builtin for clearing the host-visible console.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, console, BuiltinResult};

const CLC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ans",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty matrix placeholder returned by sink invocation.",
}];
const CLC_INPUTS_EMPTY: [BuiltinParamDescriptor; 0] = [];
const CLC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "clc()",
    inputs: &CLC_INPUTS_EMPTY,
    outputs: &CLC_OUTPUT,
}];
const CLC_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLC.ARG_COUNT",
    identifier: None,
    when: "One or more input arguments are passed to clc.",
    message: "clc: expected no input arguments",
};
const CLC_ERRORS: [BuiltinErrorDescriptor; 1] = [CLC_ERROR_ARG_COUNT];
pub const CLC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLC_ERRORS,
};

fn clc_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin("clc");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "clc",
    category = "io",
    summary = "Clear the Command Window display.",
    keywords = "clc,console,clear screen",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::io::clc::CLC_DESCRIPTOR),
    builtin_path = "crate::builtins::io::clc"
)]
async fn clc_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(clc_error(&CLC_ERROR_ARG_COUNT));
    }

    console::record_clear_screen();
    Ok(empty_return_value())
}

fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clc_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CLC_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"clc()"));
    }
}
