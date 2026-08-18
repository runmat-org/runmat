//! MATLAB-compatible `contains` builtin for RunMat.

use regex::RegexBuilder;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{BuiltinFusionSpec, ConstantStrategy, ShapeRequirements};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};

use super::text_utils::{logical_result, parse_ignore_case, TextCollection, TextElement};
use crate::builtins::strings::core::compat::pattern_regex;
use crate::builtins::strings::type_resolvers::logical_text_match_type;

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::search::contains")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "contains",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text operation; not eligible for fusion and materialises host logical results.",
};

const BUILTIN_NAME: &str = "contains";

const CONTAINS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result indicating whether each text element contains the pattern.",
}];

const CONTAINS_INPUTS_BASE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "pat",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (string/char/cell/string array).",
    },
];

const CONTAINS_INPUTS_IGNORE_CASE_POSITIONAL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "pat",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "ignoreCase",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: Some("false"),
        description: "Logical flag controlling case-sensitive matching.",
    },
];

const CONTAINS_INPUTS_IGNORE_CASE_PAIR: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "pat",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"IgnoreCase\""),
        description: "Option name (`\"IgnoreCase\"`).",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option value for `\"IgnoreCase\"`.",
    },
];

const CONTAINS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "tf = contains(str, pat)",
        inputs: &CONTAINS_INPUTS_BASE,
        outputs: &CONTAINS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = contains(str, pat, ignoreCase)",
        inputs: &CONTAINS_INPUTS_IGNORE_CASE_POSITIONAL,
        outputs: &CONTAINS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = contains(str, pat, \"IgnoreCase\", value)",
        inputs: &CONTAINS_INPUTS_IGNORE_CASE_PAIR,
        outputs: &CONTAINS_OUTPUT,
    },
];

const CONTAINS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINS.INVALID_INPUT",
    identifier: Some("RunMat:contains:InvalidInput"),
    when: "Text or pattern input is not a supported text container.",
    message: "contains: text and pattern inputs must be text values",
};

const CONTAINS_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINS.INVALID_OPTION",
    identifier: Some("RunMat:contains:InvalidOption"),
    when: "IgnoreCase option arguments are invalid or malformed.",
    message: "contains: invalid option arguments",
};

const CONTAINS_ERROR_SHAPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINS.SHAPE_MISMATCH",
    identifier: Some("RunMat:contains:ShapeMismatch"),
    when: "Text and pattern inputs are not broadcast-compatible.",
    message: "contains: input sizes are not broadcast-compatible",
};

const CONTAINS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINS.INTERNAL",
    identifier: Some("RunMat:contains:InternalError"),
    when: "Internal logical result assembly failed.",
    message: "contains: internal error",
};

const CONTAINS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    CONTAINS_ERROR_INVALID_INPUT,
    CONTAINS_ERROR_INVALID_OPTION,
    CONTAINS_ERROR_SHAPE_MISMATCH,
    CONTAINS_ERROR_INTERNAL,
];

pub const CONTAINS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTAINS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTAINS_ERRORS,
};

const CONTAINS_POSITIONAL_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "contains-positional-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "contains with a positional IgnoreCase flag is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ContainsPositionalIgnoreCaseExtension"),
    };

const CONTAINS_NUMERIC_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "contains-numeric-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "contains with a numeric IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ContainsNumericIgnoreCaseExtension"),
    };

const CONTAINS_TEXT_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "contains-text-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "contains with a textual IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ContainsTextIgnoreCaseExtension"),
    };

pub const CONTAINS_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    CONTAINS_POSITIONAL_IGNORE_CASE_EXTENSION,
    CONTAINS_NUMERIC_IGNORE_CASE_EXTENSION,
    CONTAINS_TEXT_IGNORE_CASE_EXTENSION,
];

const CONTAINS_INTEGER_IGNORE_CASE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IgnoreCase",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode accepts a scalar integer zero as false and any nonzero integer as true; MATLAB-compatible mode requires the documented logical name-value form.",
    }];

pub const CONTAINS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = contains(str, pat, 'IgnoreCase', integer_value)",
        inputs: &CONTAINS_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Integer flags are a compatibility-gated RunMat convenience. Text matching remains host-only and returns logical output; numeric subject or pattern data is never treated as text.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = contains(str, pat, integer_ignore_case)",
        inputs: &CONTAINS_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The positional integer flag requires both the positional and numeric RunMat extension gates. Zero is false and any nonzero exact integer is true; subject and pattern inputs remain text-only.",
    },
];

fn contains_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_contains_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "contains",
    category = "strings/search",
    summary = "Test whether text inputs contain patterns.",
    keywords = "contains,substring,text,ignorecase,search",
    accel = "sink",
    type_resolver(logical_text_match_type),
    descriptor(crate::builtins::strings::search::contains::CONTAINS_DESCRIPTOR),
    extensions(crate::builtins::strings::search::contains::CONTAINS_EXTENSIONS),
    integer_capabilities(
        crate::builtins::strings::search::contains::CONTAINS_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::strings::search::contains"
)]
async fn contains_builtin(
    text: Value,
    pattern: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let ignore_case = validate_contains_options(&rest)?;
    reject_resident_numeric_text(&text)?;
    reject_resident_numeric_text(&pattern)?;
    let text = gather_if_needed_async(&text)
        .await
        .map_err(remap_contains_flow)?;
    let pattern = gather_if_needed_async(&pattern)
        .await
        .map_err(remap_contains_flow)?;
    let subject = TextCollection::from_subject(BUILTIN_NAME, text).map_err(|err| {
        contains_error_with_message(err.message().to_string(), &CONTAINS_ERROR_INVALID_INPUT)
    })?;
    if matches!(pattern, Value::Object(_)) {
        let regex = pattern_regex(&pattern, BUILTIN_NAME).map_err(|err| {
            contains_error_with_message(err.message().to_string(), &CONTAINS_ERROR_INVALID_INPUT)
        })?;
        return evaluate_contains_regex(&subject, &regex, ignore_case);
    }
    let patterns = TextCollection::from_pattern(BUILTIN_NAME, pattern).map_err(|err| {
        contains_error_with_message(err.message().to_string(), &CONTAINS_ERROR_INVALID_INPUT)
    })?;
    evaluate_contains(&subject, &patterns, ignore_case)
}

fn validate_contains_options(rest: &[Value]) -> BuiltinResult<bool> {
    if rest.len() > 2 {
        return Err(contains_error_with_message(
            "contains: expected at most one 'IgnoreCase' name-value pair",
            &CONTAINS_ERROR_INVALID_OPTION,
        ));
    }
    let ignore_case = parse_ignore_case(BUILTIN_NAME, rest).map_err(|err| {
        contains_error_with_message(err.message().to_string(), &CONTAINS_ERROR_INVALID_OPTION)
    })?;
    if rest.len() == 1 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONTAINS_POSITIONAL_IGNORE_CASE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let value = match rest {
        [value] => Some(value),
        [_, value] => Some(value),
        _ => None,
    };
    if value.is_some_and(is_numeric_ignore_case_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONTAINS_NUMERIC_IGNORE_CASE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if value.is_some_and(is_text_ignore_case_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CONTAINS_TEXT_IGNORE_CASE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(ignore_case)
}

fn is_numeric_ignore_case_value(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_))
}

fn is_text_ignore_case_value(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn reject_resident_numeric_text(value: &Value) -> BuiltinResult<()> {
    if crate::dispatcher::value_contains_gpu(value) {
        return Err(contains_error_with_message(
            CONTAINS_ERROR_INVALID_INPUT.message,
            &CONTAINS_ERROR_INVALID_INPUT,
        ));
    }
    Ok(())
}

fn evaluate_contains_regex(
    subject: &TextCollection,
    pattern: &str,
    ignore_case: bool,
) -> BuiltinResult<Value> {
    let regex = RegexBuilder::new(pattern)
        .case_insensitive(ignore_case)
        .build()
        .map_err(|err| {
            contains_error_with_message(err.to_string(), &CONTAINS_ERROR_INVALID_INPUT)
        })?;
    let mut data = Vec::with_capacity(subject.elements.len());
    for element in &subject.elements {
        let value = match element {
            TextElement::Missing => false,
            TextElement::Text(text) => regex.is_match(text),
        };
        data.push(u8::from(value));
    }
    logical_result(BUILTIN_NAME, data, subject.shape.clone()).map_err(|err| {
        contains_error_with_message(err.message().to_string(), &CONTAINS_ERROR_INTERNAL)
    })
}

fn evaluate_contains(
    subject: &TextCollection,
    patterns: &TextCollection,
    ignore_case: bool,
) -> BuiltinResult<Value> {
    let output_shape = broadcast_shapes(BUILTIN_NAME, &subject.shape, &patterns.shape)
        .map_err(|err| contains_error_with_message(err, &CONTAINS_ERROR_SHAPE_MISMATCH))?;
    let total = tensor::element_count(&output_shape);
    if total == 0 {
        return logical_result(BUILTIN_NAME, Vec::new(), output_shape).map_err(|err| {
            contains_error_with_message(err.message().to_string(), &CONTAINS_ERROR_INTERNAL)
        });
    }

    let subject_strides = compute_strides(&subject.shape);
    let pattern_strides = compute_strides(&patterns.shape);
    let subject_lower = if ignore_case {
        Some(subject.lowercased())
    } else {
        None
    };
    let pattern_lower = if ignore_case {
        Some(patterns.lowercased())
    } else {
        None
    };

    let mut data = Vec::with_capacity(total);
    for linear in 0..total {
        let subject_idx = broadcast_index(linear, &output_shape, &subject.shape, &subject_strides);
        let pattern_idx = broadcast_index(linear, &output_shape, &patterns.shape, &pattern_strides);
        let value = match (
            &subject.elements[subject_idx],
            &patterns.elements[pattern_idx],
        ) {
            (TextElement::Missing, _) => false,
            (_, TextElement::Missing) => false,
            (TextElement::Text(text), TextElement::Text(pattern)) => {
                if pattern.is_empty() {
                    true
                } else if ignore_case {
                    let lowered_subject = subject_lower
                        .as_ref()
                        .and_then(|vec| vec[subject_idx].as_deref())
                        .expect("lowercase subject available");
                    let lowered_pattern = pattern_lower
                        .as_ref()
                        .and_then(|vec| vec[pattern_idx].as_deref())
                        .expect("lowercase pattern available");
                    lowered_subject.contains(lowered_pattern)
                } else {
                    text.contains(pattern.as_str())
                }
            }
        };
        data.push(if value { 1 } else { 0 });
    }
    logical_result(BUILTIN_NAME, data, output_shape).map_err(|err| {
        contains_error_with_message(err.message().to_string(), &CONTAINS_ERROR_INTERNAL)
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{
        CellArray, CharArray, IntValue, IntegerStorage, LogicalArray, ResolveContext, StringArray,
        Tensor, Type,
    };

    fn run_contains(text: Value, pattern: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(contains_builtin(text, pattern, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_string_scalar_true() {
        let result = run_contains(
            Value::String("RunMat".into()),
            Value::String("Run".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_string_scalar_false() {
        let result = run_contains(
            Value::String("RunMat".into()),
            Value::String("forge".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_option() {
        let result = run_contains(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into()), Value::Bool(true)],
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_string_array_scalar_pattern() {
        let array = StringArray::new(
            vec!["alpha".into(), "beta".into(), "gamma".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = run_contains(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("contains");
        let expected = LogicalArray::new(vec![1, 1, 1], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_elementwise_patterns() {
        let subjects = StringArray::new(
            vec!["hydrogen".into(), "helium".into(), "lithium".into()],
            vec![3, 1],
        )
        .unwrap();
        let patterns =
            StringArray::new(vec!["gen".into(), "ium".into(), "iron".into()], vec![3, 1]).unwrap();
        let result = run_contains(
            Value::StringArray(subjects),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("contains");
        let expected = LogicalArray::new(vec![1, 1, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_broadcast_pattern_column_vector() {
        let patterns = CharArray::new(vec!['s', 'n', 'x'], 3, 1).unwrap();
        let result = run_contains(
            Value::String("saturn".into()),
            Value::CharArray(patterns),
            Vec::new(),
        )
        .expect("contains char");
        let expected = LogicalArray::new(vec![1, 1, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_cell_array_patterns() {
        let cell = CellArray::new(
            vec![
                Value::from("Mercury"),
                Value::from("Venus"),
                Value::from("Mars"),
            ],
            1,
            3,
        )
        .unwrap();
        let result = run_contains(Value::Cell(cell), Value::String("us".into()), Vec::new())
            .expect("contains");
        let expected = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_missing_strings_false() {
        let array = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = run_contains(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_empty_pattern_true() {
        let result = run_contains(
            Value::String("foo".into()),
            Value::String("".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_invalid_option_name() {
        let err = run_contains(
            Value::String("foo".into()),
            Value::String("f".into()),
            vec![Value::String("IgnoreCases".into()), Value::Bool(true)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_string_flag() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run_contains(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("on".into()),
            ],
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_numeric_flag() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for value in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(u64::MAX),
        ] {
            let result = run_contains(
                Value::String("RunMat".into()),
                Value::String("run".into()),
                vec![Value::String("IgnoreCase".into()), Value::Int(value)],
            )
            .expect("RunMat numeric IgnoreCase extension");
            assert_eq!(result, Value::Bool(true));
        }
        for (flag, expected) in [
            (Value::Int(IntValue::I8(-1)), true),
            (Value::Int(IntValue::U8(0)), false),
            (
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                        .expect("integer scalar"),
                ),
                true,
            ),
        ] {
            let result = run_contains(
                Value::String("RunMat".into()),
                Value::String("run".into()),
                vec![Value::String("IgnoreCase".into()), flag],
            )
            .expect("numeric predicate flag");
            assert_eq!(result, Value::Bool(expected));
        }
    }

    #[test]
    fn contains_numeric_ignore_case_is_gated_in_matlab_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = run_contains(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::Int(IntValue::U64(u64::MAX)),
            ],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:ContainsNumericIgnoreCaseExtension")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_invalid_value() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = run_contains(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("maybe".into()),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("invalid value"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_missing_value() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = run_contains(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into())],
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("expected a value after 'IgnoreCase'"));
    }

    #[test]
    fn contains_validates_malformed_options_before_extension_gates() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let cases = [
            vec![Value::String("Bogus".into()), Value::Int(IntValue::U8(1))],
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("maybe".into()),
            ],
            vec![
                Value::String("IgnoreCase".into()),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2])
                        .expect("integer vector"),
                ),
            ],
            vec![Value::String("IgnoreCase".into()), Value::Num(f64::NAN)],
            vec![
                Value::String("IgnoreCase".into()),
                Value::Num(f64::INFINITY),
            ],
        ];
        for rest in cases {
            let err = run_contains(
                Value::String("RunMat".into()),
                Value::String("run".into()),
                rest,
            )
            .unwrap_err();
            assert_eq!(err.identifier(), Some("RunMat:contains:InvalidOption"));
        }
        assert_eq!(
            run_contains(
                Value::String("RunMat".into()),
                Value::String("run".into()),
                vec![Value::String("IgnoreCase".into()), Value::Bool(true)],
            )
            .unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn contains_distinguishes_positional_and_numeric_extension_gates() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let positional = run_contains(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::Int(IntValue::I8(-1))],
        )
        .unwrap_err();
        assert_eq!(
            positional.identifier(),
            Some("RunMat:compatibility:ContainsPositionalIgnoreCaseExtension")
        );
        let name_value = run_contains(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::Int(IntValue::I8(-1)),
            ],
        )
        .unwrap_err();
        assert_eq!(
            name_value.identifier(),
            Some("RunMat:compatibility:ContainsNumericIgnoreCaseExtension")
        );
        let textual = run_contains(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("on".into()),
            ],
        )
        .unwrap_err();
        assert_eq!(
            textual.identifier(),
            Some("RunMat:compatibility:ContainsTextIgnoreCaseExtension")
        );
    }

    #[test]
    fn contains_rejects_nested_resident_numeric_text_before_provider_access() {
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
            descriptor: Default::default(),
        });
        let nested = Value::Cell(CellArray::new(vec![resident], 1, 1).expect("cell"));
        for (text, pattern) in [
            (nested.clone(), Value::String("x".into())),
            (Value::String("x".into()), nested),
        ] {
            let err = run_contains(text, pattern, Vec::new()).unwrap_err();
            assert_eq!(err.identifier(), Some("RunMat:contains:InvalidInput"));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_mismatched_shapes_error() {
        let text = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let pattern =
            StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = run_contains(
            Value::StringArray(text),
            Value::StringArray(pattern),
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("size mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_invalid_subject_type() {
        let err = run_contains(Value::Num(1.0), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("first argument must be text"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_invalid_pattern_type() {
        let err =
            run_contains(Value::String("foo".into()), Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("pattern must be text"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_cell_invalid_element_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err =
            run_contains(Value::Cell(cell), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("cell array elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_zero_sized_inputs() {
        let subjects = StringArray::new(Vec::<String>::new(), vec![0, 1]).unwrap();
        let result = run_contains(
            Value::StringArray(subjects),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("contains");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![0, 1]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_missing_pattern_false() {
        let result = run_contains(
            Value::String("alpha".into()),
            Value::String("<missing>".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn contains_accepts_pattern_object() {
        let pattern = crate::builtins::strings::core::compat::pattern_object(r"\d+");
        let result =
            run_contains(Value::String("run42".into()), pattern, Vec::new()).expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn contains_type_is_logical_match() {
        assert_eq!(
            logical_text_match_type(
                &[Type::String, Type::String],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Bool
        );
    }
}
