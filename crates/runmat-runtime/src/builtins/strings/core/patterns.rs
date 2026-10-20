//! Shared bounded-count parsing for Unicode-letter and wildcard patterns.

use runmat_builtins::{
    BuiltinErrorDescriptor, LETTERS_PATTERN_ERROR_ARGUMENT_COUNT,
    LETTERS_PATTERN_ERROR_INVALID_COUNT, LETTERS_PATTERN_ERROR_INVALID_RANGE,
    WILDCARD_PATTERN_ERROR_ARGUMENT_COUNT, WILDCARD_PATTERN_ERROR_INVALID_COUNT,
    WILDCARD_PATTERN_ERROR_INVALID_RANGE,
};
use runmat_value::{NumericScalar, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::tensor;
use crate::builtins::strings::core::numeric_length::nonnegative_platform_usize;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

const LETTERS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    LETTERS_PATTERN_ERROR_ARGUMENT_COUNT,
    LETTERS_PATTERN_ERROR_INVALID_COUNT,
    LETTERS_PATTERN_ERROR_INVALID_RANGE,
];
const WILDCARD_ERRORS: [BuiltinErrorDescriptor; 3] = [
    WILDCARD_PATTERN_ERROR_ARGUMENT_COUNT,
    WILDCARD_PATTERN_ERROR_INVALID_COUNT,
    WILDCARD_PATTERN_ERROR_INVALID_RANGE,
];

#[derive(Clone, Copy)]
pub(crate) enum BoundedPatternKind {
    Letters,
    Wildcard,
}

impl BoundedPatternKind {
    fn name(self) -> &'static str {
        match self {
            Self::Letters => "lettersPattern",
            Self::Wildcard => "wildcardPattern",
        }
    }

    fn atom(self) -> &'static str {
        match self {
            Self::Letters => r"\p{Alphabetic}",
            Self::Wildcard => ".",
        }
    }

    fn lazy(self) -> bool {
        matches!(self, Self::Wildcard)
    }

    fn errors(self) -> &'static [BuiltinErrorDescriptor; 3] {
        match self {
            Self::Letters => &LETTERS_ERRORS,
            Self::Wildcard => &WILDCARD_ERRORS,
        }
    }
}

pub(crate) async fn bounded_regex(
    rest: crate::call::BuiltinArguments,
    kind: BoundedPatternKind,
) -> BuiltinResult<String> {
    let name = kind.name();
    let atom = kind.atom();
    let lazy = kind.lazy();
    let errors = kind.errors();
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
            parse_count(count, kind)?,
            if lazy { "?" } else { "" }
        )),
        [minimum, maximum] => {
            let minimum = parse_count(minimum, kind)?;
            let suffix = if lazy { "?" } else { "" };
            if is_positive_infinity(maximum) {
                Ok(format!("{atom}{{{minimum},}}{suffix}"))
            } else {
                let maximum = parse_count(maximum, kind)?;
                if minimum > maximum {
                    return Err(pattern_error(name, &errors[2], errors[2].message));
                }
                Ok(format!("{atom}{{{minimum},{maximum}}}{suffix}"))
            }
        }
        _ => unreachable!("arity was validated"),
    }
}

fn parse_count(value: &Value, kind: BoundedPatternKind) -> BuiltinResult<usize> {
    let parsed = match value {
        Value::Num(value) => nonnegative_platform_usize(*value),
        Value::Int(value) => value.try_to_usize(),
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            if let Some(storage) = value.integer_storage() {
                storage.value_at(0).and_then(|value| value.try_to_usize())
            } else {
                value.numeric_value_at(0).and_then(|value| match value {
                    NumericScalar::F64(value) => nonnegative_platform_usize(value),
                    NumericScalar::F32(value) => nonnegative_platform_usize(f64::from(value)),
                    _ => None,
                })
            }
        }
        _ => None,
    };
    parsed.ok_or_else(|| {
        let errors = kind.errors();
        pattern_error(kind.name(), &errors[1], errors[1].message)
    })
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
    use runmat_value::{IntValue, IntegerStorage, Tensor};

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
                    crate::builtins::common::test_support::builtin_arguments(vec![value]),
                    BoundedPatternKind::Letters,
                ))
                .unwrap(),
                r"\p{Alphabetic}{2}"
            );
        }
        assert_eq!(
            block_on(bounded_regex(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Int(IntValue::U8(2)),
                    Value::Int(IntValue::U64(4))
                ]),
                BoundedPatternKind::Wildcard,
            ))
            .unwrap(),
            ".{2,4}?"
        );
    }

    #[test]
    fn bounded_patterns_validate_ranges_and_infinity() {
        assert_eq!(
            block_on(bounded_regex(
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
                BoundedPatternKind::Wildcard
            ))
            .unwrap(),
            ".*?"
        );
        assert_eq!(
            block_on(bounded_regex(
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
                BoundedPatternKind::Letters,
            ))
            .unwrap(),
            r"\p{Alphabetic}+"
        );
        assert_eq!(
            block_on(bounded_regex(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Int(IntValue::U8(3)),
                    Value::Num(f64::INFINITY)
                ]),
                BoundedPatternKind::Letters,
            ))
            .unwrap(),
            r"\p{Alphabetic}{3,}"
        );
        assert!(block_on(bounded_regex(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(IntValue::U8(4)),
                Value::Int(IntValue::U8(3))
            ]),
            BoundedPatternKind::Wildcard,
        ))
        .is_err());
        assert!(block_on(bounded_regex(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(
                usize::MAX as f64
            )]),
            BoundedPatternKind::Wildcard,
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
            let handle = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("admit provenance view");
            assert_eq!(
                block_on(bounded_regex(
                    crate::builtins::common::test_support::builtin_arguments(vec![
                        Value::GpuTensor(handle.clone())
                    ]),
                    BoundedPatternKind::Letters,
                ))
                .expect("automatic residency is transparent"),
                r"\p{Alphabetic}{3}"
            );

            let handle = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("admit provenance view");
            let error = block_on(bounded_regex(
                crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                    handle.clone(),
                )]),
                BoundedPatternKind::Letters,
            ))
            .expect_err("explicit gpuArray count is unsupported");
            assert_eq!(
                error.identifier(),
                Some("RunMat:lettersPattern:InvalidCount")
            );
            handle.release().expect("release fixture handle");
        });
    }
}
