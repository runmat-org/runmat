//! MATLAB-compatible `endsWith` builtin for RunMat.

use regex::RegexBuilder;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::text_utils::{logical_result, parse_ignore_case, TextCollection, TextElement};
use crate::builtins::strings::core::compat::pattern_regex;
use crate::builtins::strings::type_resolvers::logical_text_match_type;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::search::endswith")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "endsWith",
    op_kind: GpuOpKind::Custom("string-search"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The builtin owns resident arguments so numeric text roles reject and option compatibility gates run before provider access; admitted controls gather explicitly and outputs remain host logical values.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::search::endswith")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "endsWith",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text operation; not eligible for fusion and materialises host logical results.",
};

const BUILTIN_NAME: &str = "endsWith";

const ENDSWITH_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result indicating whether each text element ends with the pattern.",
}];

const ENDSWITH_INPUTS_BASE: [BuiltinParamDescriptor; 2] = [
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

const ENDSWITH_INPUTS_IGNORE_CASE_POSITIONAL: [BuiltinParamDescriptor; 3] = [
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

const ENDSWITH_INPUTS_IGNORE_CASE_PAIR: [BuiltinParamDescriptor; 4] = [
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

const ENDSWITH_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "tf = endsWith(str, pat)",
        inputs: &ENDSWITH_INPUTS_BASE,
        outputs: &ENDSWITH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = endsWith(str, pat, ignoreCase)",
        inputs: &ENDSWITH_INPUTS_IGNORE_CASE_POSITIONAL,
        outputs: &ENDSWITH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = endsWith(str, pat, \"IgnoreCase\", value)",
        inputs: &ENDSWITH_INPUTS_IGNORE_CASE_PAIR,
        outputs: &ENDSWITH_OUTPUT,
    },
];

const ENDSWITH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENDSWITH.INVALID_INPUT",
    identifier: Some("RunMat:endsWith:InvalidInput"),
    when: "Text or pattern input is not a supported text container.",
    message: "endsWith: text and pattern inputs must be text values",
};

const ENDSWITH_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENDSWITH.INVALID_OPTION",
    identifier: Some("RunMat:endsWith:InvalidOption"),
    when: "IgnoreCase option arguments are invalid or malformed.",
    message: "endsWith: invalid option arguments",
};

const ENDSWITH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENDSWITH.INTERNAL",
    identifier: Some("RunMat:endsWith:InternalError"),
    when: "Internal logical result assembly failed.",
    message: "endsWith: internal error",
};

const ENDSWITH_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ENDSWITH_ERROR_INVALID_INPUT,
    ENDSWITH_ERROR_INVALID_OPTION,
    ENDSWITH_ERROR_INTERNAL,
];

pub const ENDSWITH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ENDSWITH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ENDSWITH_ERRORS,
};

const ENDSWITH_POSITIONAL_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "endswith-positional-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "endsWith with a positional IgnoreCase flag is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EndsWithPositionalIgnoreCaseExtension"),
    };

const ENDSWITH_NUMERIC_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "endswith-numeric-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "endsWith with a numeric IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EndsWithNumericIgnoreCaseExtension"),
    };

const ENDSWITH_TEXT_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "endswith-text-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "endsWith with a textual IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EndsWithTextIgnoreCaseExtension"),
    };

const ENDSWITH_RESIDENT_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "endswith-resident-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "endsWith with a resident IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EndsWithResidentIgnoreCaseExtension"),
    };

pub const ENDSWITH_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    ENDSWITH_POSITIONAL_IGNORE_CASE_EXTENSION,
    ENDSWITH_NUMERIC_IGNORE_CASE_EXTENSION,
    ENDSWITH_TEXT_IGNORE_CASE_EXTENSION,
    ENDSWITH_RESIDENT_IGNORE_CASE_EXTENSION,
];

const ENDSWITH_INTEGER_IGNORE_CASE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IgnoreCase",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode accepts exact scalar integer zero as false and every nonzero integer as true; MATLAB-compatible mode requires the documented logical name-value form.",
    }];

pub const ENDSWITH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = endsWith(str, pat, 'IgnoreCase', integer_value)",
        inputs: &ENDSWITH_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Integer flags are compatibility-gated controls. Subject and pattern remain host text, multiple patterns are alternatives, and output has the subject shape.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = endsWith(str, pat, integer_ignore_case)",
        inputs: &ENDSWITH_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The positional integer flag requires both positional and numeric RunMat extension gates; a resident flag additionally requires the resident-control gate before gather.",
    },
];

fn endswith_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_endswith_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "endsWith",
    category = "strings/search",
    summary = "Test whether text inputs end with patterns.",
    keywords = "endswith,suffix,text,ignorecase,search",
    accel = "sink",
    type_resolver(logical_text_match_type),
    descriptor(crate::builtins::strings::search::endswith::ENDSWITH_DESCRIPTOR),
    extensions(crate::builtins::strings::search::endswith::ENDSWITH_EXTENSIONS),
    integer_capabilities(
        crate::builtins::strings::search::endswith::ENDSWITH_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::strings::search::endswith"
)]
async fn endswith_builtin(
    text: Value,
    pattern: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    reject_resident_numeric_text(&text)?;
    reject_resident_numeric_text(&pattern)?;
    let subject = TextCollection::from_subject(BUILTIN_NAME, text).map_err(|err| {
        endswith_error_with_message(err.message().to_string(), &ENDSWITH_ERROR_INVALID_INPUT)
    })?;
    let ignore_case = validate_endswith_options(rest).await?;
    if matches!(pattern, Value::Object(_)) {
        let regex = pattern_regex(&pattern, BUILTIN_NAME).map_err(|err| {
            endswith_error_with_message(err.message().to_string(), &ENDSWITH_ERROR_INVALID_INPUT)
        })?;
        return evaluate_endswith_regex(&subject, &regex, ignore_case);
    }
    let patterns = TextCollection::from_pattern(BUILTIN_NAME, pattern).map_err(|err| {
        endswith_error_with_message(err.message().to_string(), &ENDSWITH_ERROR_INVALID_INPUT)
    })?;
    evaluate_endswith(&subject, &patterns, ignore_case)
}

async fn validate_endswith_options(rest: Vec<Value>) -> BuiltinResult<bool> {
    if rest.len() > 2 {
        return Err(endswith_error_with_message(
            "endsWith: expected at most one 'IgnoreCase' name-value pair",
            &ENDSWITH_ERROR_INVALID_OPTION,
        ));
    }
    if rest.len() == 2 {
        let option_name = super::text_utils::value_to_owned_string(&rest[0]);
        if !option_name.is_some_and(|name| name.eq_ignore_ascii_case("IgnoreCase")) {
            return Err(endswith_error_with_message(
                "endsWith: unknown option; supported option is 'IgnoreCase'",
                &ENDSWITH_ERROR_INVALID_OPTION,
            ));
        }
    }
    if let Some(value) = option_value(&rest) {
        validate_ignore_case_shape(value)?;
    }
    let has_resident = rest
        .iter()
        .any(|value| crate::dispatcher::value_contains_gpu(value));
    let parsed_host = if has_resident {
        None
    } else {
        Some(parse_ignore_case(BUILTIN_NAME, &rest).map_err(|err| {
            endswith_error_with_message(err.message().to_string(), &ENDSWITH_ERROR_INVALID_OPTION)
        })?)
    };

    if rest.len() == 1 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ENDSWITH_POSITIONAL_IGNORE_CASE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let value = option_value(&rest);
    if value.is_some_and(is_numeric_ignore_case_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ENDSWITH_NUMERIC_IGNORE_CASE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if value.is_some_and(is_text_ignore_case_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ENDSWITH_TEXT_IGNORE_CASE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if has_resident {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ENDSWITH_RESIDENT_IGNORE_CASE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    if let Some(ignore_case) = parsed_host {
        return Ok(ignore_case);
    }

    let mut host = Vec::with_capacity(rest.len());
    for value in rest {
        host.push(
            gather_if_needed_async(&value)
                .await
                .map_err(remap_endswith_flow)?,
        );
    }
    parse_ignore_case(BUILTIN_NAME, &host).map_err(|err| {
        endswith_error_with_message(err.message().to_string(), &ENDSWITH_ERROR_INVALID_OPTION)
    })
}

fn option_value(rest: &[Value]) -> Option<&Value> {
    match rest {
        [value] => Some(value),
        [_, value] => Some(value),
        _ => None,
    }
}

fn validate_ignore_case_shape(value: &Value) -> BuiltinResult<()> {
    let element_count = match value {
        Value::GpuTensor(handle) => tensor::element_count(&handle.shape),
        Value::Tensor(tensor) => tensor.len(),
        Value::LogicalArray(array) => array.data.len(),
        _ => 1,
    };
    if element_count != 1 {
        let message = if matches!(value, Value::LogicalArray(_)) {
            "endsWith: option values must be scalar logicals"
        } else {
            "endsWith: IgnoreCase must be a scalar"
        };
        return Err(endswith_error_with_message(
            message,
            &ENDSWITH_ERROR_INVALID_OPTION,
        ));
    }
    Ok(())
}

fn is_numeric_ignore_case_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    )
}

fn is_text_ignore_case_value(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn reject_resident_numeric_text(value: &Value) -> BuiltinResult<()> {
    if crate::dispatcher::value_contains_gpu(value) {
        return Err(endswith_error_with_message(
            ENDSWITH_ERROR_INVALID_INPUT.message,
            &ENDSWITH_ERROR_INVALID_INPUT,
        ));
    }
    Ok(())
}

fn evaluate_endswith_regex(
    subject: &TextCollection,
    pattern: &str,
    ignore_case: bool,
) -> BuiltinResult<Value> {
    let regex = RegexBuilder::new(&format!("(?:{pattern})$"))
        .case_insensitive(ignore_case)
        .build()
        .map_err(|err| {
            endswith_error_with_message(err.to_string(), &ENDSWITH_ERROR_INVALID_INPUT)
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
        endswith_error_with_message(err.message().to_string(), &ENDSWITH_ERROR_INTERNAL)
    })
}

fn evaluate_endswith(
    subject: &TextCollection,
    patterns: &TextCollection,
    ignore_case: bool,
) -> BuiltinResult<Value> {
    let output_shape = subject.shape.clone();
    let total = subject.elements.len();
    if total == 0 {
        return logical_result(BUILTIN_NAME, Vec::new(), output_shape).map_err(|err| {
            endswith_error_with_message(err.message().to_string(), &ENDSWITH_ERROR_INTERNAL)
        });
    }

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
    for subject_idx in 0..total {
        let value = match &subject.elements[subject_idx] {
            TextElement::Missing => false,
            TextElement::Text(text) => {
                patterns
                    .elements
                    .iter()
                    .enumerate()
                    .any(|(pattern_idx, element)| match element {
                        TextElement::Missing => false,
                        TextElement::Text(pattern) if pattern.is_empty() => true,
                        TextElement::Text(_pattern) if ignore_case => {
                            let lowered_subject = subject_lower
                                .as_ref()
                                .and_then(|values| values[subject_idx].as_deref())
                                .expect("lowercase subject available");
                            let lowered_pattern = pattern_lower
                                .as_ref()
                                .and_then(|values| values[pattern_idx].as_deref())
                                .expect("lowercase pattern available");
                            lowered_subject.ends_with(lowered_pattern)
                        }
                        TextElement::Text(pattern) => text.ends_with(pattern.as_str()),
                    })
            }
        };
        data.push(if value { 1 } else { 0 });
    }
    logical_result(BUILTIN_NAME, data, output_shape).map_err(|err| {
        endswith_error_with_message(err.message().to_string(), &ENDSWITH_ERROR_INTERNAL)
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{CellArray, CharArray, IntValue, LogicalArray, StringArray, Tensor};

    fn run_endswith(text: Value, pattern: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(endswith_builtin(text, pattern, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_string_scalar_true() {
        let result = run_endswith(
            Value::String("RunMat".into()),
            Value::String("Mat".into()),
            Vec::new(),
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_string_scalar_false() {
        let result = run_endswith(
            Value::String("RunMat".into()),
            Value::String("Run".into()),
            Vec::new(),
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_ignore_case_option() {
        let result = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::String("IgnoreCase".into()), Value::Bool(true)],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_string_array_scalar_pattern() {
        let array = StringArray::new(
            vec!["alpha".into(), "beta".into(), "gamma".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = run_endswith(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("endsWith");
        let expected = LogicalArray::new(vec![1, 1, 1], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_elementwise_patterns() {
        let subjects = StringArray::new(
            vec!["hydrogen".into(), "helium".into(), "lithium".into()],
            vec![3, 1],
        )
        .unwrap();
        let patterns =
            StringArray::new(vec!["gen".into(), "ium".into(), "ium".into()], vec![3, 1]).unwrap();
        let result = run_endswith(
            Value::StringArray(subjects),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("endsWith");
        let expected = LogicalArray::new(vec![1, 1, 1], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_multiple_patterns_are_alternatives_for_scalar_subject() {
        let patterns = CharArray::new(vec!['n', 'x', 'r'], 3, 1).unwrap();
        let result = run_endswith(
            Value::String("saturn".into()),
            Value::CharArray(patterns),
            Vec::new(),
        )
        .expect("endsWith char");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_cell_array_patterns() {
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
        let result = run_endswith(Value::Cell(cell), Value::String("s".into()), Vec::new())
            .expect("endsWith");
        let expected = LogicalArray::new(vec![0, 1, 1], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_missing_strings_false() {
        let array = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = run_endswith(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_empty_pattern_true() {
        let result = run_endswith(
            Value::String("foo".into()),
            Value::String("".into()),
            Vec::new(),
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_invalid_option_name() {
        let err = run_endswith(
            Value::String("foo".into()),
            Value::String("o".into()),
            vec![Value::String("IgnoreCases".into()), Value::Bool(true)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_ignore_case_string_flag() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("on".into()),
            ],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_ignore_case_numeric_flag() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::Int(IntValue::I32(0)),
            ],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_ignore_case_positional_value() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::Bool(true)],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_ignore_case_logical_array_value() {
        let logical = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let result = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::LogicalArray(logical),
            ],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_ignore_case_tensor_value() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::String("IgnoreCase".into()), Value::Tensor(tensor)],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_ignore_case_gpu_tensor_flag() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let data = [1.0];
            let shape = [1usize, 1usize];
            let handle = provider
                .upload(&HostTensorView {
                    data: &data,
                    shape: &shape,
                })
                .expect("upload");
            let result = run_endswith(
                Value::String("RunMat".into()),
                Value::String("mat".into()),
                vec![
                    Value::String("IgnoreCase".into()),
                    Value::GpuTensor(handle.clone()),
                ],
            )
            .expect("endsWith");
            assert_eq!(result, Value::Bool(true));
            provider.free(&handle).expect("free gpu flag");
        });
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_ignore_case_gpu_tensor_flag_wgpu() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        if register_wgpu_provider(WgpuProviderOptions::default()).is_err() {
            // Skip when wgpu backend cannot initialise on this machine.
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let data = [1.0];
        let shape = [1usize, 1usize];
        let handle = provider
            .upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload");
        let result = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::GpuTensor(handle.clone()),
            ],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
        let _ = provider.free(&handle);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_ignore_case_invalid_value() {
        let err = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
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
    fn endswith_ignore_case_logical_array_invalid_size() {
        let logical = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        let err = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::LogicalArray(logical),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("scalar logicals"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_ignore_case_numeric_nan_invalid() {
        let err = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::Num(f64::NAN)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("finite scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_ignore_case_missing_value() {
        let err = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::String("IgnoreCase".into())],
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("expected a value after 'IgnoreCase'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_official_multiple_pattern_example_preserves_subject_shape() {
        let text = StringArray::new(
            vec![
                "data.tar.gz".into(),
                "mycode.m".into(),
                "outputs.xlsx".into(),
                "results.pptx".into(),
            ],
            vec![1, 4],
        )
        .unwrap();
        let pattern = StringArray::new(
            vec![".docx".into(), ".xlsx".into(), ".gz".into()],
            vec![1, 3],
        )
        .unwrap();
        let result = run_endswith(
            Value::StringArray(text),
            Value::StringArray(pattern),
            Vec::new(),
        )
        .expect("endsWith");
        assert_eq!(
            result,
            Value::LogicalArray(LogicalArray::new(vec![1, 0, 1, 0], vec![1, 4]).unwrap())
        );
    }

    #[test]
    fn endswith_integer_ignore_case_all_classes_are_exact_runmat_extensions() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for flag in [
            IntValue::I8(-1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(i64::MAX),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(u64::MAX),
        ] {
            assert_eq!(
                run_endswith(
                    Value::String("RunMat".into()),
                    Value::String("mat".into()),
                    vec![Value::String("IgnoreCase".into()), Value::Int(flag)],
                )
                .unwrap(),
                Value::Bool(true)
            );
        }
    }

    #[test]
    fn endswith_integer_and_positional_flags_are_gated_in_matlab_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let numeric = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::Int(IntValue::U64(u64::MAX)),
            ],
        )
        .unwrap_err();
        assert_eq!(
            numeric.identifier(),
            Some("RunMat:compatibility:EndsWithNumericIgnoreCaseExtension")
        );
        let positional = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::Bool(true)],
        )
        .unwrap_err();
        assert_eq!(
            positional.identifier(),
            Some("RunMat:compatibility:EndsWithPositionalIgnoreCaseExtension")
        );
    }

    #[test]
    fn endswith_rejects_nested_resident_numeric_text_before_provider_access() {
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let nested = Value::Cell(CellArray::new(vec![resident], 1, 1).unwrap());
        for (text, pattern) in [
            (nested.clone(), Value::String("x".into())),
            (Value::String("x".into()), nested),
        ] {
            let err = run_endswith(text, pattern, Vec::new()).unwrap_err();
            assert_eq!(err.identifier(), Some("RunMat:endsWith:InvalidInput"));
        }
    }

    #[test]
    fn endswith_resident_control_rejects_before_provider_access_in_matlab_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let err = run_endswith(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::String("IgnoreCase".into()), resident],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:EndsWithNumericIgnoreCaseExtension")
        );
    }

    #[test]
    fn endswith_dispatch_preserves_residency_until_builtin_preflight() {
        assert_eq!(GPU_SPEC.residency, ResidencyPolicy::NewHandle);
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 1,
        });
        let prepared = futures::executor::block_on(runmat_accelerate::prepare_builtin_args(
            "endsWith",
            &[resident],
        ))
        .expect("dispatcher must retain resident argument");
        assert!(matches!(prepared.as_slice(), [Value::GpuTensor(_)]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_invalid_subject_type() {
        let err = run_endswith(Value::Num(1.0), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("first argument must be text"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_invalid_pattern_type() {
        let err =
            run_endswith(Value::String("foo".into()), Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(
            err.to_string().contains("pattern must be text"),
            "expected pattern type error, got: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_cell_invalid_element_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err =
            run_endswith(Value::Cell(cell), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("cell array elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn endswith_zero_sized_inputs() {
        let subjects = StringArray::new(Vec::<String>::new(), vec![0, 1]).unwrap();
        let result = run_endswith(
            Value::StringArray(subjects),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("endsWith");
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
    fn endswith_missing_pattern_false() {
        let result = run_endswith(
            Value::String("alpha".into()),
            Value::String("<missing>".into()),
            Vec::new(),
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn endswith_accepts_pattern_object() {
        let pattern = crate::builtins::strings::core::compat::pattern_object(r"\d+");
        let result =
            run_endswith(Value::String("run42".into()), pattern, Vec::new()).expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn endswith_type_is_logical_match() {
        assert_eq!(
            logical_text_match_type(
                &[Type::String, Type::String],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Bool
        );
    }
}
