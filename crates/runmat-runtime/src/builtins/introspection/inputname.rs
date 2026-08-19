use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_value::CharArray;
use runmat_value::Value;

use crate::builtins::common::tensor;

const INPUTNAME_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::Any,
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

pub const INPUTNAME_ERROR_ARGUMENT_EXCEEDS_INPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.INPUTNAME.ARGUMENT_EXCEEDS_INPUTS",
        identifier: Some("RunMat:InputnameArgumentExceedsInputs"),
        when: "The requested argument number exceeds the caller's input count or no function-input callsite exists.",
        message: "inputname: argument number exceeds number of function inputs",
    };

pub const INPUTNAME_ERRORS: [BuiltinErrorDescriptor; 4] = [
    INPUTNAME_ERROR_NOT_ENOUGH_INPUTS,
    INPUTNAME_ERROR_TOO_MANY_INPUTS,
    INPUTNAME_ERROR_ARGUMENT_INVALID,
    INPUTNAME_ERROR_ARGUMENT_EXCEEDS_INPUTS,
];

const INPUTNAME_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "argNumber",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes, single, and double are documented for the scalar real positive integer-valued caller-argument index; logical is not admitted.",
    }];
pub const INPUTNAME_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "s = inputname(integer_argNumber)",
        inputs: &INPUTNAME_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The index is read exactly before caller-frame lookup. The result is a character row vector for a named argument and an empty character array for an unnamed expression.",
    }];

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
    if let Value::Int(value) = value {
        let index = value.try_to_usize()?;
        return (index >= 1).then_some(index);
    }
    if let Value::Tensor(tensor) = value {
        if !tensor::is_scalar_tensor(tensor) {
            return None;
        }
        if let Some(storage) = tensor.integer_storage() {
            let index = storage.value_at(0)?.try_to_usize()?;
            return (index >= 1).then_some(index);
        }
    }
    let n = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        _ => return None,
    };
    if !n.is_finite()
        || n < 1.0
        || n.fract() != 0.0
        || n > usize::MAX as f64
        || (usize::BITS == 64 && n == usize::MAX as f64)
    {
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

fn contains_comma_list_indexing(text: &str) -> bool {
    let chars = text.char_indices().collect::<Vec<_>>();
    let mut quote = None;
    let mut previous_non_whitespace = None;
    let mut index = 0;
    while index < chars.len() {
        let (_, ch) = chars[index];
        if let Some(active_quote) = quote {
            if ch == active_quote {
                if index + 1 < chars.len() && chars[index + 1].1 == active_quote {
                    index += 2;
                    continue;
                }
                quote = None;
            }
            index += 1;
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            index += 1;
            continue;
        }
        if ch == '{'
            && previous_non_whitespace.is_some_and(|previous: char| {
                previous == '_'
                    || previous.is_ascii_alphanumeric()
                    || matches!(previous, ')' | ']' | '}')
            })
        {
            return true;
        }
        if ch == '.' {
            let remaining = chars[index + 1..]
                .iter()
                .map(|(_, ch)| *ch)
                .collect::<Vec<_>>();
            let next = remaining.iter().copied().find(|ch| !ch.is_whitespace());
            if previous_non_whitespace.is_some_and(|previous: char| {
                previous == '_'
                    || previous.is_ascii_alphabetic()
                    || matches!(previous, ')' | ']' | '}')
            }) && next == Some('(')
            {
                return true;
            }
            if previous_non_whitespace.is_some_and(|previous: char| {
                previous == '_'
                    || previous.is_ascii_alphabetic()
                    || matches!(previous, ')' | ']' | '}')
            }) && next.is_some_and(|next| next == '_' || next.is_ascii_alphabetic())
            {
                let after_member = remaining
                    .iter()
                    .skip_while(|ch| ch.is_whitespace())
                    .skip_while(|ch| **ch == '_' || ch.is_ascii_alphanumeric())
                    .copied()
                    .find(|ch| !ch.is_whitespace());
                if after_member != Some('(') {
                    return true;
                }
            }
        }
        if !ch.is_whitespace() {
            previous_non_whitespace = Some(ch);
        }
        index += 1;
    }
    false
}

pub(crate) fn dispatch_inputname(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match args.len() {
        0 => return Err(descriptor_error(&INPUTNAME_ERROR_NOT_ENOUGH_INPUTS)),
        1 => {}
        _ => return Err(descriptor_error(&INPUTNAME_ERROR_TOO_MANY_INPUTS)),
    }
    let index = numeric_index(&args[0])
        .ok_or_else(|| descriptor_error(&INPUTNAME_ERROR_ARGUMENT_INVALID))?;
    let callsite = crate::callsite::current_function_input_callsite()
        .ok_or_else(|| descriptor_error(&INPUTNAME_ERROR_ARGUMENT_EXCEEDS_INPUTS))?;
    let actual_inputs =
        super::arity_check::current_input_count().unwrap_or(callsite.arg_spans.len());
    if index > actual_inputs {
        return Err(descriptor_error(&INPUTNAME_ERROR_ARGUMENT_EXCEEDS_INPUTS));
    }
    let follows_comma_list = (0..index).any(|arg_index| {
        crate::callsite::function_input_arg_text(arg_index)
            .is_some_and(|text| contains_comma_list_indexing(text.trim()))
    });
    let text = crate::callsite::function_input_arg_text(index - 1)
        .map(|text| text.trim().to_string())
        .filter(|text| !follows_comma_list && is_simple_identifier(text))
        .unwrap_or_default();
    if text.is_empty() {
        Ok(Value::CharArray(
            CharArray::new(Vec::new(), 0, 0).expect("valid empty character array"),
        ))
    } else {
        Ok(Value::CharArray(CharArray::new_row(&text)))
    }
}

#[runmat_macros::runtime_builtin(
    name = "inputname",
    category = "introspection",
    summary = "Return the caller argument variable name for a function input.",
    integer_capabilities(self::INPUTNAME_INTEGER_CAPABILITIES),
    descriptor(self::INPUTNAME_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::inputname"
)]
pub fn inputname_builtin_registered(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_inputname(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{SourceId, Span};
    use runmat_value::{IntegerStorage, Tensor};

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
        assert_eq!(name, Value::CharArray(CharArray::new_row("alpha")));
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
        let empty = Value::CharArray(CharArray::new(Vec::new(), 0, 0).unwrap());
        assert_eq!(expr, empty);
        assert_eq!(literal, empty);
        let missing = dispatch_inputname(vec![Value::Num(4.0)]).unwrap_err();
        assert_eq!(
            missing.identifier(),
            INPUTNAME_ERROR_ARGUMENT_EXCEEDS_INPUTS.identifier
        );
    }

    #[test]
    fn inputname_reads_typed_integer_tensor_index_exactly() {
        let source = "out = probe(alpha, beta);";
        let _catalog_guard = crate::source_context::replace_source_catalog(vec![(
            SourceId(9),
            "/tmp/caller.m".to_string(),
            source.to_string(),
        )]);
        let _callsite_guard = crate::callsite::push_function_input_callsite(
            Some(SourceId(9)),
            Some(vec![span_of(source, "alpha"), span_of(source, "beta")]),
        );
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("integer tensor");

        let name = dispatch_inputname(vec![Value::Tensor(tensor)]).expect("inputname succeeds");
        assert_eq!(name, Value::CharArray(CharArray::new_row("beta")));
    }

    #[test]
    fn inputname_propagates_dot_and_cell_indexing_to_later_arguments_only() {
        let empty = Value::CharArray(CharArray::new(Vec::new(), 0, 0).unwrap());
        for (source_id, source, indexed) in [
            (
                SourceId(11),
                "out = probe(alpha, object.field, beta);",
                "object.field",
            ),
            (
                SourceId(12),
                "out = probe(alpha, cells{2}, beta);",
                "cells{2}",
            ),
        ] {
            let _catalog_guard = crate::source_context::replace_source_catalog(vec![(
                source_id,
                "/tmp/caller.m".to_string(),
                source.to_string(),
            )]);
            let _callsite_guard = crate::callsite::push_function_input_callsite(
                Some(source_id),
                Some(vec![
                    span_of(source, "alpha"),
                    span_of(source, indexed),
                    span_of(source, "beta"),
                ]),
            );
            assert_eq!(
                dispatch_inputname(vec![Value::Num(1.0)]).unwrap(),
                Value::CharArray(CharArray::new_row("alpha"))
            );
            assert_eq!(dispatch_inputname(vec![Value::Num(2.0)]).unwrap(), empty);
            assert_eq!(dispatch_inputname(vec![Value::Num(3.0)]).unwrap(), empty);
        }
    }

    #[test]
    fn inputname_does_not_treat_decimal_or_string_dots_as_comma_lists() {
        assert!(!contains_comma_list_indexing("1.25"));
        assert!(!contains_comma_list_indexing("'example.name'"));
        assert!(!contains_comma_list_indexing("alpha + 1"));
        assert!(!contains_comma_list_indexing("pkg.func(alpha)"));
        assert!(contains_comma_list_indexing("object.field"));
        assert!(contains_comma_list_indexing("cells { 2 }"));
    }

    #[test]
    fn inputname_rejects_unrepresentable_double_index_before_cast() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };

        assert_eq!(numeric_index(&Value::Num(boundary)), None);
    }

    #[test]
    fn inputname_accepts_all_integer_scalar_classes_and_rejects_logical() {
        let source = "out = probe(alpha);";
        let _catalog_guard = crate::source_context::replace_source_catalog(vec![(
            SourceId(10),
            "/tmp/caller.m".to_string(),
            source.to_string(),
        )]);
        let _callsite_guard = crate::callsite::push_function_input_callsite(
            Some(SourceId(10)),
            Some(vec![span_of(source, "alpha")]),
        );
        for index in [
            runmat_value::IntValue::I8(1),
            runmat_value::IntValue::I16(1),
            runmat_value::IntValue::I32(1),
            runmat_value::IntValue::I64(1),
            runmat_value::IntValue::U8(1),
            runmat_value::IntValue::U16(1),
            runmat_value::IntValue::U32(1),
            runmat_value::IntValue::U64(1),
        ] {
            assert_eq!(
                dispatch_inputname(vec![Value::Int(index)]).unwrap(),
                Value::CharArray(CharArray::new_row("alpha"))
            );
        }
        assert!(dispatch_inputname(vec![Value::Bool(true)]).is_err());
        assert!(dispatch_inputname(vec![Value::LogicalArray(
            runmat_value::LogicalArray::new(vec![1], vec![1, 1]).unwrap()
        )])
        .is_err());
    }

    #[test]
    fn inputname_rejects_resident_index_without_provider_access() {
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
            descriptor: Default::default(),
        });
        let error = dispatch_inputname(vec![resident]).unwrap_err();
        assert_eq!(
            error.identifier(),
            INPUTNAME_ERROR_ARGUMENT_INVALID.identifier
        );
    }
}
