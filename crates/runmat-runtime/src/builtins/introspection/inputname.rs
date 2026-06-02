use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};

const INPUTNAME_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Caller argument variable name, or empty text when unavailable.",
}];

const INPUTNAME_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "argNumber",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based caller argument index.",
}];

const INPUTNAME_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "name = inputname(argNumber)",
    inputs: &INPUTNAME_INPUTS,
    outputs: &INPUTNAME_OUTPUT,
}];

pub const INPUTNAME_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTNAME.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:NotEnoughInputs"),
    when: "No argument index is provided.",
    message: "inputname: not enough input arguments",
};

pub const INPUTNAME_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTNAME.TOO_MANY_INPUTS",
    identifier: Some("RunMat:TooManyInputs"),
    when: "More than one argument index is provided.",
    message: "inputname: too many input arguments",
};

pub const INPUTNAME_ERROR_ARGUMENT_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTNAME.ARGUMENT_INVALID",
    identifier: Some("RunMat:InputnameArgumentInvalid"),
    when: "The argument index is not a positive integer scalar.",
    message: "inputname: argument index must be a positive integer scalar",
};

pub const INPUTNAME_ERRORS: [BuiltinErrorDescriptor; 3] = [
    INPUTNAME_ERROR_NOT_ENOUGH_INPUTS,
    INPUTNAME_ERROR_TOO_MANY_INPUTS,
    INPUTNAME_ERROR_ARGUMENT_INVALID,
];

pub const INPUTNAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INPUTNAME_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INPUTNAME_ERRORS,
};

fn descriptor_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    crate::runtime_descriptor_error("inputname", error)
}

fn numeric_index(value: &Value) -> Option<usize> {
    let n = match value {
        Value::Int(value) => value.to_f64(),
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        _ => return None,
    };
    if !n.is_finite() || n < 1.0 || n.fract() != 0.0 || n > usize::MAX as f64 {
        return None;
    }
    Some(n as usize)
}

fn is_simple_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

pub(crate) fn dispatch_inputname(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match args.len() {
        0 => return Err(descriptor_error(&INPUTNAME_ERROR_NOT_ENOUGH_INPUTS)),
        1 => {}
        _ => return Err(descriptor_error(&INPUTNAME_ERROR_TOO_MANY_INPUTS)),
    }
    let index = numeric_index(&args[0])
        .ok_or_else(|| descriptor_error(&INPUTNAME_ERROR_ARGUMENT_INVALID))?;
    let text = crate::callsite::function_input_arg_text(index - 1)
        .map(|text| text.trim().to_string())
        .filter(|text| is_simple_identifier(text))
        .unwrap_or_default();
    Ok(Value::String(text))
}

#[runmat_macros::runtime_builtin(
    name = "inputname",
    category = "introspection",
    summary = "Return the caller argument variable name for a function input.",
    descriptor(self::INPUTNAME_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::inputname"
)]
pub fn inputname_builtin_registered(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_inputname(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_hir::{SourceId, Span};

    fn span_of(source: &str, needle: &str) -> Span {
        let start = source.find(needle).expect("needle present");
        Span {
            start,
            end: start + needle.len(),
        }
    }

    #[test]
    fn inputname_reads_simple_caller_argument_name() {
        let source = "out = probe(alpha, alpha + 1, 7);";
        let _catalog_guard = crate::source_context::replace_source_catalog(vec![(
            SourceId(7),
            "/tmp/caller.m".to_string(),
            source.to_string(),
        )]);
        let _callsite_guard = crate::callsite::push_function_input_callsite(
            Some(SourceId(7)),
            Some(vec![
                span_of(source, "alpha"),
                span_of(source, "alpha + 1"),
                span_of(source, "7"),
            ]),
        );

        let name = dispatch_inputname(vec![Value::Num(1.0)]).expect("inputname succeeds");
        assert_eq!(name, Value::String("alpha".to_string()));
    }

    #[test]
    fn inputname_returns_empty_for_expression_literal_and_missing_context() {
        let source = "out = probe(alpha, alpha + 1, 7);";
        let _catalog_guard = crate::source_context::replace_source_catalog(vec![(
            SourceId(8),
            "/tmp/caller.m".to_string(),
            source.to_string(),
        )]);
        let _callsite_guard = crate::callsite::push_function_input_callsite(
            Some(SourceId(8)),
            Some(vec![
                span_of(source, "alpha"),
                span_of(source, "alpha + 1"),
                span_of(source, "7"),
            ]),
        );

        let expr = dispatch_inputname(vec![Value::Num(2.0)]).expect("inputname succeeds");
        let literal = dispatch_inputname(vec![Value::Num(3.0)]).expect("inputname succeeds");
        let missing = dispatch_inputname(vec![Value::Num(4.0)]).expect("inputname succeeds");
        assert_eq!(expr, Value::String(String::new()));
        assert_eq!(literal, Value::String(String::new()));
        assert_eq!(missing, Value::String(String::new()));
    }
}
