//! MATLAB-compatible `startsWith` builtin for RunMat.

use regex::RegexBuilder;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
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

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};

use super::text_utils::{logical_result, parse_ignore_case, TextCollection, TextElement};
use crate::builtins::strings::core::compat::pattern_regex;
use crate::builtins::strings::type_resolvers::logical_text_match_type;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::search::startswith")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "startsWith",
    op_kind: GpuOpKind::Custom("string-search"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes entirely on the host; inputs are gathered from the GPU before evaluating prefix checks.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::search::startswith"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "startsWith",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text operation; not eligible for fusion and materialises host logical results.",
};

const BUILTIN_NAME: &str = "startsWith";

const STARTSWITH_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result indicating whether each text element starts with the pattern.",
}];

const STARTSWITH_INPUTS_BASE: [BuiltinParamDescriptor; 2] = [
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

const STARTSWITH_INPUTS_IGNORE_CASE_POSITIONAL: [BuiltinParamDescriptor; 3] = [
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

const STARTSWITH_INPUTS_IGNORE_CASE_PAIR: [BuiltinParamDescriptor; 4] = [
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

const STARTSWITH_INPUTS_OPTION_PAIRS: [BuiltinParamDescriptor; 3] = [
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
        name: "nameValuePairs...",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value option pairs (`\"IgnoreCase\"`, value).",
    },
];

const STARTSWITH_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "tf = startsWith(str, pat)",
        inputs: &STARTSWITH_INPUTS_BASE,
        outputs: &STARTSWITH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = startsWith(str, pat, ignoreCase)",
        inputs: &STARTSWITH_INPUTS_IGNORE_CASE_POSITIONAL,
        outputs: &STARTSWITH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = startsWith(str, pat, \"IgnoreCase\", value)",
        inputs: &STARTSWITH_INPUTS_IGNORE_CASE_PAIR,
        outputs: &STARTSWITH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = startsWith(str, pat, nameValuePairs...)",
        inputs: &STARTSWITH_INPUTS_OPTION_PAIRS,
        outputs: &STARTSWITH_OUTPUT,
    },
];

const STARTSWITH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STARTSWITH.INVALID_INPUT",
    identifier: Some("RunMat:startsWith:InvalidInput"),
    when: "Text or pattern input is not a supported text container.",
    message: "startsWith: text and pattern inputs must be text values",
};

const STARTSWITH_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STARTSWITH.INVALID_OPTION",
    identifier: Some("RunMat:startsWith:InvalidOption"),
    when: "IgnoreCase option arguments are invalid or malformed.",
    message: "startsWith: invalid option arguments",
};

const STARTSWITH_ERROR_SHAPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STARTSWITH.SHAPE_MISMATCH",
    identifier: Some("RunMat:startsWith:ShapeMismatch"),
    when: "Text and pattern inputs are not broadcast-compatible.",
    message: "startsWith: input sizes are not broadcast-compatible",
};

const STARTSWITH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STARTSWITH.INTERNAL",
    identifier: Some("RunMat:startsWith:InternalError"),
    when: "Internal logical result assembly failed.",
    message: "startsWith: internal error",
};

const STARTSWITH_ERRORS: [BuiltinErrorDescriptor; 4] = [
    STARTSWITH_ERROR_INVALID_INPUT,
    STARTSWITH_ERROR_INVALID_OPTION,
    STARTSWITH_ERROR_SHAPE_MISMATCH,
    STARTSWITH_ERROR_INTERNAL,
];

pub const STARTSWITH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STARTSWITH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STARTSWITH_ERRORS,
};

const STARTSWITH_POSITIONAL_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "startswith-positional-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "startsWith with a positional IgnoreCase flag is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StartsWithPositionalIgnoreCaseExtension"),
    };

const STARTSWITH_NUMERIC_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "startswith-numeric-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "startsWith with a numeric IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StartsWithNumericIgnoreCaseExtension"),
    };

const STARTSWITH_TEXT_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "startswith-text-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "startsWith with a textual IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StartsWithTextIgnoreCaseExtension"),
    };

const STARTSWITH_RESIDENT_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "startswith-resident-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "startsWith with a resident IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StartsWithResidentIgnoreCaseExtension"),
    };

pub const STARTSWITH_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    STARTSWITH_POSITIONAL_IGNORE_CASE_EXTENSION,
    STARTSWITH_NUMERIC_IGNORE_CASE_EXTENSION,
    STARTSWITH_TEXT_IGNORE_CASE_EXTENSION,
    STARTSWITH_RESIDENT_IGNORE_CASE_EXTENSION,
];

const STARTSWITH_INTEGER_IGNORE_CASE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IgnoreCase",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode accepts exact scalar integer zero as false and every nonzero integer as true; MATLAB-compatible mode requires the documented logical name-value form.",
    }];

pub const STARTSWITH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = startsWith(str, pat, 'IgnoreCase', integer_value)",
        inputs: &STARTSWITH_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Integer flags are compatibility-gated controls. Subject and pattern remain host text, multiple patterns are alternatives, and output has the subject shape.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = startsWith(str, pat, integer_ignore_case)",
        inputs: &STARTSWITH_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The positional integer flag requires both positional and numeric RunMat extension gates; a resident flag additionally requires the resident-control gate before gather.",
    },
];

fn startswith_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_startswith_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "startsWith",
    category = "strings/search",
    summary = "Test whether text inputs begin with specified prefix patterns.",
    keywords = "startswith,prefix,text,ignorecase,search",
    accel = "sink",
    type_resolver(logical_text_match_type),
    descriptor(crate::builtins::strings::search::startswith::STARTSWITH_DESCRIPTOR),
    extensions(crate::builtins::strings::search::startswith::STARTSWITH_EXTENSIONS),
    integer_capabilities(
        crate::builtins::strings::search::startswith::STARTSWITH_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::strings::search::startswith"
)]
async fn startswith_builtin(
    text: Value,
    pattern: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    reject_resident_numeric_text(&text)?;
    reject_resident_numeric_text(&pattern)?;
    let subject = TextCollection::from_subject(BUILTIN_NAME, text).map_err(|err| {
        startswith_error_with_message(err.message().to_string(), &STARTSWITH_ERROR_INVALID_INPUT)
    })?;
    let ignore_case = validate_startswith_options(rest).await?;
    if matches!(pattern, Value::Object(_)) {
        let regex = pattern_regex(&pattern, BUILTIN_NAME).map_err(|err| {
            startswith_error_with_message(
                err.message().to_string(),
                &STARTSWITH_ERROR_INVALID_INPUT,
            )
        })?;
        return evaluate_startswith_regex(&subject, &regex, ignore_case);
    }
    let patterns = TextCollection::from_pattern(BUILTIN_NAME, pattern).map_err(|err| {
        startswith_error_with_message(err.message().to_string(), &STARTSWITH_ERROR_INVALID_INPUT)
    })?;
    evaluate_startswith(&subject, &patterns, ignore_case)
}

async fn validate_startswith_options(rest: Vec<Value>) -> BuiltinResult<bool> {
    if rest.len() > 2 {
        return Err(startswith_error_with_message(
            "startsWith: expected at most one 'IgnoreCase' name-value pair",
            &STARTSWITH_ERROR_INVALID_OPTION,
        ));
    }
    if rest.len() == 2 {
        let option_name = super::text_utils::value_to_owned_string(&rest[0]);
        if !option_name.is_some_and(|name| name.eq_ignore_ascii_case("IgnoreCase")) {
            return Err(startswith_error_with_message(
                "startsWith: unknown option; supported option is 'IgnoreCase'",
                &STARTSWITH_ERROR_INVALID_OPTION,
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
            startswith_error_with_message(
                err.message().to_string(),
                &STARTSWITH_ERROR_INVALID_OPTION,
            )
        })?)
    };

    if rest.len() == 1 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STARTSWITH_POSITIONAL_IGNORE_CASE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let value = option_value(&rest);
    if value.is_some_and(is_numeric_ignore_case_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STARTSWITH_NUMERIC_IGNORE_CASE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if value.is_some_and(is_text_ignore_case_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STARTSWITH_TEXT_IGNORE_CASE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if has_resident {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STARTSWITH_RESIDENT_IGNORE_CASE_EXTENSION,
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
                .map_err(remap_startswith_flow)?,
        );
    }
    parse_ignore_case(BUILTIN_NAME, &host).map_err(|err| {
        startswith_error_with_message(err.message().to_string(), &STARTSWITH_ERROR_INVALID_OPTION)
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
            "startsWith: option values must be scalar logicals"
        } else {
            "startsWith: IgnoreCase must be a scalar"
        };
        return Err(startswith_error_with_message(
            message,
            &STARTSWITH_ERROR_INVALID_OPTION,
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
        return Err(startswith_error_with_message(
            STARTSWITH_ERROR_INVALID_INPUT.message,
            &STARTSWITH_ERROR_INVALID_INPUT,
        ));
    }
    Ok(())
}

fn evaluate_startswith_regex(
    subject: &TextCollection,
    pattern: &str,
    ignore_case: bool,
) -> BuiltinResult<Value> {
    let regex = RegexBuilder::new(&format!("^(?:{pattern})"))
        .case_insensitive(ignore_case)
        .build()
        .map_err(|err| {
            startswith_error_with_message(err.to_string(), &STARTSWITH_ERROR_INVALID_INPUT)
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
        startswith_error_with_message(err.message().to_string(), &STARTSWITH_ERROR_INTERNAL)
    })
}

fn evaluate_startswith(
    subject: &TextCollection,
    patterns: &TextCollection,
    ignore_case: bool,
) -> BuiltinResult<Value> {
    let output_shape = broadcast_shapes(BUILTIN_NAME, &subject.shape, &patterns.shape)
        .map_err(|err| startswith_error_with_message(err, &STARTSWITH_ERROR_SHAPE_MISMATCH))?;
    let total = tensor::element_count(&output_shape);
    if total == 0 {
        return logical_result(BUILTIN_NAME, Vec::new(), output_shape).map_err(|err| {
            startswith_error_with_message(err.message().to_string(), &STARTSWITH_ERROR_INTERNAL)
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
                    lowered_subject.starts_with(lowered_pattern)
                } else {
                    text.starts_with(pattern.as_str())
                }
            }
        };
        data.push(if value { 1 } else { 0 });
    }
    logical_result(BUILTIN_NAME, data, output_shape).map_err(|err| {
        startswith_error_with_message(err.message().to_string(), &STARTSWITH_ERROR_INTERNAL)
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{
        CellArray, CharArray, IntValue, IntegerStorage, LogicalArray, StringArray, Tensor,
    };

    fn run_startswith(text: Value, pattern: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(startswith_builtin(text, pattern, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_string_scalar_true() {
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("Run".into()),
            Vec::new(),
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_string_scalar_false() {
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("Mat".into()),
            Vec::new(),
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_option() {
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into()), Value::Bool(true)],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_string_array_scalar_pattern() {
        let array = StringArray::new(
            vec!["alpha".into(), "beta".into(), "gamma".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = run_startswith(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("startsWith");
        let expected = LogicalArray::new(vec![1, 0, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_elementwise_patterns() {
        let subjects = StringArray::new(
            vec!["hydrogen".into(), "helium".into(), "lithium".into()],
            vec![3, 1],
        )
        .unwrap();
        let patterns =
            StringArray::new(vec!["hyd".into(), "hel".into(), "lit".into()], vec![3, 1]).unwrap();
        let result = run_startswith(
            Value::StringArray(subjects),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("startsWith");
        let expected = LogicalArray::new(vec![1, 1, 1], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_broadcast_pattern_column_vector() {
        let patterns = CharArray::new(vec!['s', 'n', 'x'], 3, 1).unwrap();
        let result = run_startswith(
            Value::String("saturn".into()),
            Value::CharArray(patterns),
            Vec::new(),
        )
        .expect("startsWith char");
        let expected = LogicalArray::new(vec![1, 0, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_cell_array_patterns() {
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
        let result = run_startswith(Value::Cell(cell), Value::String("M".into()), Vec::new())
            .expect("startsWith");
        let expected = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_missing_strings_false() {
        let array = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = run_startswith(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_empty_pattern_true() {
        let result = run_startswith(
            Value::String("foo".into()),
            Value::String("".into()),
            Vec::new(),
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_invalid_option_name() {
        let err = run_startswith(
            Value::String("foo".into()),
            Value::String("f".into()),
            vec![Value::String("IgnoreCases".into()), Value::Bool(true)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_string_flag() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("on".into()),
            ],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_numeric_flag() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::Int(IntValue::I32(0)),
            ],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_positional_value() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::Bool(true)],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_logical_array_value() {
        let logical = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::LogicalArray(logical),
            ],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_tensor_value() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into()), Value::Tensor(tensor)],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_typed_integer_tensor_reads_exact_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor =
            Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("integer tensor");
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into()), Value::Tensor(tensor)],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_invalid_value() {
        let err = run_startswith(
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
    fn startswith_ignore_case_logical_array_invalid_size() {
        let logical = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        let err = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
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
    fn startswith_ignore_case_numeric_nan_invalid() {
        let err = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::Num(f64::NAN)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("finite scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_missing_value() {
        let err = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into())],
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("expected a value after 'IgnoreCase'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_mismatched_shapes_error() {
        let text = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let pattern =
            StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = run_startswith(
            Value::StringArray(text),
            Value::StringArray(pattern),
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("size mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_invalid_subject_type() {
        let err =
            run_startswith(Value::Num(1.0), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("first argument must be text"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_invalid_pattern_type() {
        let err =
            run_startswith(Value::String("foo".into()), Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(
            err.to_string().contains("pattern must be text"),
            "expected pattern type error, got: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_cell_invalid_element_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err =
            run_startswith(Value::Cell(cell), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("cell array elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_zero_sized_inputs() {
        let subjects = StringArray::new(Vec::<String>::new(), vec![0, 1]).unwrap();
        let result = run_startswith(
            Value::StringArray(subjects),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("startsWith");
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
    fn startswith_missing_pattern_false() {
        let result = run_startswith(
            Value::String("alpha".into()),
            Value::String("<missing>".into()),
            Vec::new(),
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn startswith_accepts_pattern_object() {
        let pattern = crate::builtins::strings::core::compat::pattern_object(r"run\d+");
        let result =
            run_startswith(Value::String("run42".into()), pattern, Vec::new()).expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn startswith_type_is_logical_match() {
        assert_eq!(
            logical_text_match_type(
                &[Type::String, Type::String],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Bool
        );
    }
}
