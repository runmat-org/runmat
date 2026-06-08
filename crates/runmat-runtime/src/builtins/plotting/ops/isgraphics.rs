use crate::builtins::plotting::type_resolvers::bool_type;
use runmat_builtins::Value;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

const ISGRAPHICS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input resolves to a valid plotting graphics handle.",
}];

const ISGRAPHICS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const ISGRAPHICS_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Candidate plotting handle value.",
}];

const ISGRAPHICS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = isgraphics()",
        inputs: &ISGRAPHICS_INPUTS_NONE,
        outputs: &ISGRAPHICS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = isgraphics(h)",
        inputs: &ISGRAPHICS_INPUTS_HANDLE,
        outputs: &ISGRAPHICS_OUTPUT,
    },
];

const ISGRAPHICS_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const ISGRAPHICS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISGRAPHICS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISGRAPHICS_ERRORS,
};

#[runtime_builtin(
    name = "isgraphics",
    category = "plotting",
    summary = "Return true if the input is a valid plotting graphics handle.",
    keywords = "isgraphics,plotting,handle",
    suppress_auto_output = true,
    type_resolver(bool_type),
    descriptor(crate::builtins::plotting::isgraphics::ISGRAPHICS_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::isgraphics"
)]
pub fn isgraphics_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
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
    Ok(crate::builtins::plotting::properties::resolve_plot_handle(value, "isgraphics").is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isgraphics_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ISGRAPHICS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"tf = isgraphics()"));
        assert!(labels.contains(&"tf = isgraphics(h)"));
    }
}
