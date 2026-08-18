//! Integer capability metadata and bounded-count parsing for string-pattern constructors.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, NumericScalar, Value,
};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

const PATTERN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pat",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host pattern object.",
}];

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const COUNT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exact nonnegative number of characters to match.",
}];

const RANGE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "minCharacters",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum nonnegative number of characters to match.",
    },
    BuiltinParamDescriptor {
        name: "maxCharacters",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum nonnegative count, or positive infinity for no upper bound.",
    },
];

const LETTERS_PATTERN_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "pat = lettersPattern",
        inputs: &NO_INPUTS,
        outputs: &PATTERN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "pat = lettersPattern(N)",
        inputs: &COUNT_INPUTS,
        outputs: &PATTERN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "pat = lettersPattern(minCharacters, maxCharacters)",
        inputs: &RANGE_INPUTS,
        outputs: &PATTERN_OUTPUT,
    },
];

const WILDCARD_PATTERN_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "pat = wildcardPattern",
        inputs: &NO_INPUTS,
        outputs: &PATTERN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "pat = wildcardPattern(N)",
        inputs: &COUNT_INPUTS,
        outputs: &PATTERN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "pat = wildcardPattern(minCharacters, maxCharacters)",
        inputs: &RANGE_INPUTS,
        outputs: &PATTERN_OUTPUT,
    },
];

const LETTERS_PATTERN_ERRORS: [BuiltinErrorDescriptor; 3] = pattern_errors(
    "RM.LETTERS_PATTERN.ARGUMENT_COUNT",
    "RunMat:lettersPattern:ArgumentCount",
    "RM.LETTERS_PATTERN.INVALID_COUNT",
    "RunMat:lettersPattern:InvalidCount",
    "RM.LETTERS_PATTERN.INVALID_RANGE",
    "RunMat:lettersPattern:InvalidRange",
    "lettersPattern: expected zero, one, or two input arguments",
    "lettersPattern: invalid character count",
    "lettersPattern: minCharacters must not exceed maxCharacters",
);

const WILDCARD_PATTERN_ERRORS: [BuiltinErrorDescriptor; 3] = pattern_errors(
    "RM.WILDCARD_PATTERN.ARGUMENT_COUNT",
    "RunMat:wildcardPattern:ArgumentCount",
    "RM.WILDCARD_PATTERN.INVALID_COUNT",
    "RunMat:wildcardPattern:InvalidCount",
    "RM.WILDCARD_PATTERN.INVALID_RANGE",
    "RunMat:wildcardPattern:InvalidRange",
    "wildcardPattern: expected zero, one, or two count arguments",
    "wildcardPattern: invalid character count",
    "wildcardPattern: minCharacters must not exceed maxCharacters",
);

const fn pattern_errors(
    count_code: &'static str,
    count_identifier: &'static str,
    value_code: &'static str,
    value_identifier: &'static str,
    range_code: &'static str,
    range_identifier: &'static str,
    count_message: &'static str,
    value_message: &'static str,
    range_message: &'static str,
) -> [BuiltinErrorDescriptor; 3] {
    [
        BuiltinErrorDescriptor {
            code: count_code,
            identifier: Some(count_identifier),
            when: "More than two count arguments are supplied.",
            message: count_message,
        },
        BuiltinErrorDescriptor {
            code: value_code,
            identifier: Some(value_identifier),
            when: "A count is not a host or automatically resident nonnegative numeric integer scalar, or positive infinity in the maximum position.",
            message: value_message,
        },
        BuiltinErrorDescriptor {
            code: range_code,
            identifier: Some(range_identifier),
            when: "minCharacters exceeds maxCharacters.",
            message: range_message,
        },
    ]
}

pub const LETTERS_PATTERN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LETTERS_PATTERN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LETTERS_PATTERN_ERRORS,
};

pub const WILDCARD_PATTERN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WILDCARD_PATTERN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WILDCARD_PATTERN_ERRORS,
};

pub const PATTERN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "pattern constructs a literal pattern only from text. Numeric and integer inputs reject without implicit text conversion or provider access.",
};

pub const REGEXP_PATTERN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "regexpPattern constructs a regular-expression pattern only from text. Numeric and integer inputs reject without implicit text conversion or provider access.",
    };

const COUNT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "N",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every built-in integer class plus integer-valued host single or double can specify the exact nonnegative character count.",
    }];

const RANGE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "minCharacters",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The inclusive lower bound is read exactly from every built-in integer class.",
    },
    BuiltinIntegerInputCapability {
        name: "maxCharacters",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The inclusive upper bound is read exactly from every built-in integer class; positive host floating infinity separately denotes an unbounded maximum.",
    },
];

const fn bounded_capabilities(
    exact_form: &'static str,
    range_form: &'static str,
    lazy: bool,
) -> [BuiltinIntegerCapabilityDescriptor; 2] {
    [
        BuiltinIntegerCapabilityDescriptor {
            form: exact_form,
            inputs: &COUNT_INTEGER_INPUTS,
            computation_domain: BuiltinIntegerComputationDomain::Structural,
            output_class: BuiltinIntegerOutputClassRule::NotApplicable,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::HostOnly,
            overload: BuiltinIntegerOverloadKind::StructuralParameter,
            notes: "N controls only the exact repetition bound of the returned host pattern object; explicit interactive GPU input is not documented, while automatic residency may gather transparently.",
        },
        BuiltinIntegerCapabilityDescriptor {
            form: range_form,
            inputs: &RANGE_INTEGER_INPUTS,
            computation_domain: BuiltinIntegerComputationDomain::Structural,
            output_class: BuiltinIntegerOutputClassRule::NotApplicable,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::HostOnly,
            overload: BuiltinIntegerOverloadKind::StructuralParameter,
            notes: if lazy {
                "The inclusive bounds are structural controls, min must not exceed max, and wildcard matching is lazy toward the minimum bound."
            } else {
                "The inclusive bounds are structural controls, min must not exceed max, and letter matching is greedy toward the maximum bound."
            },
        },
    ]
}

pub const LETTERS_PATTERN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] =
    bounded_capabilities(
        "pat = lettersPattern(integer_N)",
        "pat = lettersPattern(integer_minCharacters, integer_maxCharacters)",
        false,
    );

pub const WILDCARD_PATTERN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] =
    bounded_capabilities(
        "pat = wildcardPattern(integer_N)",
        "pat = wildcardPattern(integer_minCharacters, integer_maxCharacters)",
        true,
    );

pub(crate) async fn bounded_regex(
    rest: Vec<Value>,
    atom: &str,
    name: &'static str,
    lazy: bool,
) -> BuiltinResult<String> {
    let errors = errors_for(name);
    if rest.len() > 2 {
        return Err(pattern_error(name, &errors[0], errors[0].message));
    }
    if rest.iter().any(is_explicit_resident) {
        return Err(pattern_error(
            name,
            &errors[1],
            format!("{name}: explicit interactive GPU count input is not supported"),
        ));
    }
    let mut gathered = Vec::with_capacity(rest.len());
    for value in rest {
        gathered.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|error| map_control_flow_with_builtin(error, name))?,
        );
    }
    match gathered.as_slice() {
        [] if lazy => Ok(format!("{atom}*?")),
        [] => Ok(format!("{atom}+")),
        [count] => Ok(format!(
            "{atom}{{{}}}{}",
            parse_count(count, name)?,
            if lazy { "?" } else { "" }
        )),
        [minimum, maximum] => {
            let minimum = parse_count(minimum, name)?;
            let suffix = if lazy { "?" } else { "" };
            if is_positive_infinity(maximum) {
                Ok(format!("{atom}{{{minimum},}}{suffix}"))
            } else {
                let maximum = parse_count(maximum, name)?;
                if minimum > maximum {
                    return Err(pattern_error(name, &errors[2], errors[2].message));
                }
                Ok(format!("{atom}{{{minimum},{maximum}}}{suffix}"))
            }
        }
        _ => unreachable!("arity was validated"),
    }
}

fn parse_count(value: &Value, name: &'static str) -> BuiltinResult<usize> {
    let parsed = match value {
        Value::Num(value) => nonnegative_usize(*value),
        Value::Int(value) => value.try_to_usize(),
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            if let Some(storage) = value.integer_storage() {
                storage.value_at(0).and_then(|value| value.try_to_usize())
            } else {
                value.numeric_value_at(0).and_then(|value| match value {
                    NumericScalar::F64(value) => nonnegative_usize(value),
                    NumericScalar::F32(value) => nonnegative_usize(f64::from(value)),
                    _ => None,
                })
            }
        }
        _ => None,
    };
    parsed.ok_or_else(|| {
        let errors = errors_for(name);
        pattern_error(name, &errors[1], errors[1].message)
    })
}

fn nonnegative_usize(value: f64) -> Option<usize> {
    if value.is_finite()
        && value >= 0.0
        && value.fract() == 0.0
        && (value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64))
    {
        Some(value as usize)
    } else {
        None
    }
}

fn is_positive_infinity(value: &Value) -> bool {
    match value {
        Value::Num(value) => value.is_infinite() && value.is_sign_positive(),
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            value.numeric_value_at(0).is_some_and(|value| match value {
                NumericScalar::F64(value) => value.is_infinite() && value.is_sign_positive(),
                NumericScalar::F32(value) => value.is_infinite() && value.is_sign_positive(),
                _ => false,
            })
        }
        _ => false,
    }
}

fn is_explicit_resident(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
}

fn errors_for(name: &str) -> &'static [BuiltinErrorDescriptor; 3] {
    if name == "lettersPattern" {
        &LETTERS_PATTERN_ERRORS
    } else {
        &WILDCARD_PATTERN_ERRORS
    }
}

fn pattern_error(
    name: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(name);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_builtins::{IntValue, IntegerStorage, Tensor};

    #[test]
    fn bounded_patterns_read_all_integer_classes_exactly() {
        for storage in [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ] {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
            assert_eq!(
                block_on(bounded_regex(
                    vec![value],
                    r"\p{Alphabetic}",
                    "lettersPattern",
                    false
                ))
                .unwrap(),
                r"\p{Alphabetic}{2}"
            );
        }
        assert_eq!(
            block_on(bounded_regex(
                vec![Value::Int(IntValue::U8(2)), Value::Int(IntValue::U64(4))],
                ".",
                "wildcardPattern",
                true,
            ))
            .unwrap(),
            ".{2,4}?"
        );
    }

    #[test]
    fn bounded_patterns_validate_ranges_and_infinity() {
        assert_eq!(
            block_on(bounded_regex(Vec::new(), ".", "wildcardPattern", true)).unwrap(),
            ".*?"
        );
        assert_eq!(
            block_on(bounded_regex(
                Vec::new(),
                r"\p{Alphabetic}",
                "lettersPattern",
                false,
            ))
            .unwrap(),
            r"\p{Alphabetic}+"
        );
        assert_eq!(
            block_on(bounded_regex(
                vec![Value::Int(IntValue::U8(3)), Value::Num(f64::INFINITY)],
                r"\p{Alphabetic}",
                "lettersPattern",
                false,
            ))
            .unwrap(),
            r"\p{Alphabetic}{3,}"
        );
        assert!(block_on(bounded_regex(
            vec![Value::Int(IntValue::U8(4)), Value::Int(IntValue::U8(3))],
            ".",
            "wildcardPattern",
            true,
        ))
        .is_err());
        assert!(block_on(bounded_regex(
            vec![Value::Num(usize::MAX as f64)],
            ".",
            "wildcardPattern",
            true,
        ))
        .is_err());
    }

    #[test]
    fn bounded_patterns_gather_automatic_integer_counts_but_reject_explicit_counts() {
        test_support::with_test_provider(|provider| {
            let values = [3_u64];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &[1, 1],
                })
                .expect("automatic integer count");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            assert_eq!(
                block_on(bounded_regex(
                    vec![Value::GpuTensor(handle.clone())],
                    r"\p{Alphabetic}",
                    "lettersPattern",
                    false,
                ))
                .expect("automatic residency is transparent"),
                r"\p{Alphabetic}{3}"
            );

            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let error = block_on(bounded_regex(
                vec![Value::GpuTensor(handle.clone())],
                r"\p{Alphabetic}",
                "lettersPattern",
                false,
            ))
            .expect_err("explicit gpuArray count is unsupported");
            assert_eq!(
                error.identifier(),
                Some("RunMat:lettersPattern:InvalidCount")
            );
            provider.free(&handle).expect("free count");
        });
    }
}
