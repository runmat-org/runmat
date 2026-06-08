use crate::builtins::plotting::type_resolvers::bool_type;
use runmat_builtins::Value;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

const ISHANDLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input resolves to a valid plotting handle.",
}];

const ISHANDLE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const ISHANDLE_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Candidate plotting handle value.",
}];

const ISHANDLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = ishandle()",
        inputs: &ISHANDLE_INPUTS_NONE,
        outputs: &ISHANDLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = ishandle(h)",
        inputs: &ISHANDLE_INPUTS_HANDLE,
        outputs: &ISHANDLE_OUTPUT,
    },
];

const ISHANDLE_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const ISHANDLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISHANDLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISHANDLE_ERRORS,
};

#[runtime_builtin(
    name = "ishandle",
    category = "plotting",
    summary = "Return true if the input is a valid plotting handle.",
    keywords = "ishandle,plotting,handle",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::ishandle::ISHANDLE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::ishandle"
)]
pub fn ishandle_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    let Some(value) = args.first() else {
        return Ok(false);
    };
    if let Some(v) = match value {
        Value::Num(v) => Some(*v),
        _ => None,
    } {
        if !v.is_finite() || v <= 0.0 {
            return Ok(false);
        }
    }
    Ok(crate::builtins::plotting::properties::resolve_plot_handle(value, "ishandle").is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ishandle_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ISHANDLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"tf = ishandle()"));
        assert!(labels.contains(&"tf = ishandle(h)"));
    }
}
