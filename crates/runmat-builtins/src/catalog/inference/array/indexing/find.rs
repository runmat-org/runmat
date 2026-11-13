use super::super::super::argument_error;
use super::common::{double, finish};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallContract, CallInference, CallRequest, DynamicReason, LiteralContext, LiteralValue,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer_find(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FIND-ARITY",
            "find requires X and at most two option arguments",
            request.arguments.len().min(3),
        ));
    }
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::Numeric(_)
                | ValueKindFact::Logical
                | ValueKindFact::Character
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FIND-INPUT",
                "find requires numeric, logical, or character array input",
                0,
            ));
        }
    }
    if let Some(control) = request.arguments.get(1) {
        let direction_only = request.arguments.len() == 2
            && matches!(
                control.kind,
                ValueKindFact::String | ValueKindFact::Character
            );
        if !direction_only
            && !matches!(
                control.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FIND-LIMIT-TYPE",
                "find limit must be a numeric scalar",
                1,
            ));
        }
        if !direction_only && control.shape.is_definitely_nonscalar() {
            diagnostics.push(argument_error(
                "RM-CATALOG-FIND-LIMIT-SHAPE",
                "find limit must be a scalar",
                1,
            ));
        }
        if let Some(literal) = request.literals.literal_args.get(1) {
            let invalid = match literal {
                LiteralValue::Integer { .. } => literal
                    .integer_value_i128()
                    .is_some_and(|value| value < 1 || usize::try_from(value).is_err()),
                LiteralValue::Number(_) | LiteralValue::Real { .. } => {
                    request.literals.numeric_at(1).is_some_and(|limit| {
                        !limit.is_finite()
                            || limit < 1.0
                            || limit.fract() != 0.0
                            || limit > usize::MAX.saturating_sub(1) as f64
                    })
                }
                _ => false,
            };
            if invalid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FIND-LIMIT",
                    "find limit must be a positive platform integer",
                    1,
                ));
            }
        }
    }
    let direction_index = if request.arguments.len() == 3 {
        Some(2)
    } else if request.arguments.len() == 2
        && request.arguments.get(1).is_some_and(|argument| {
            matches!(
                argument.kind,
                ValueKindFact::String | ValueKindFact::Character
            )
        })
    {
        Some(1)
    } else {
        None
    };
    if let Some(direction_index) = direction_index {
        if request
            .arguments
            .get(direction_index)
            .is_some_and(|argument| {
                !matches!(
                    argument.kind,
                    ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                )
            })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FIND-DIRECTION-TYPE",
                "find direction must be text",
                direction_index,
            ));
        }
        if let Some(direction) =
            request
                .literals
                .literal_args
                .get(direction_index)
                .and_then(|value| match value {
                    LiteralValue::String(text)
                    | LiteralValue::Character(text)
                    | LiteralValue::Keyword(text) => Some(text.as_str()),
                    _ => None,
                })
        {
            if direction != "first" && direction != "last" {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FIND-DIRECTION",
                    "find direction must be first or last",
                    direction_index,
                ));
            }
        }
    }
    let input = request.arguments.first();
    let dims = input.and_then(|value| value.shape.known_dims());
    let row = matches!(dims.as_ref(), Some(dims) if dims.len() == 2 && dims[0] == Some(1));
    let scalar =
        matches!(dims.as_ref(), Some(dims) if dims.len() == 2 && dims == &[Some(1), Some(1)]);
    let known_scalar_nonzero = scalar
        && request
            .literals
            .literal_args
            .first()
            .is_some_and(literal_nonzero);
    let known_scalar_zero = scalar
        && request
            .literals
            .literal_args
            .first()
            .is_some_and(literal_zero);
    let one_shape = if known_scalar_zero
        || input.is_some_and(|value| value.shape.known_dims() == Some(vec![Some(0), Some(0)]))
    {
        ShapeFact::from(vec![Some(0), Some(0)])
    } else if known_scalar_nonzero {
        ShapeFact::from(vec![Some(1), Some(1)])
    } else if scalar {
        ShapeFact::from(vec![None, None])
    } else if input.is_some_and(|value| value.shape.element_count() == Some(0)) {
        ShapeFact::from(if row {
            vec![Some(1), Some(0)]
        } else {
            vec![Some(0), Some(1)]
        })
    } else if row && input.is_some_and(|value| value.shape.is_definitely_nonscalar()) {
        ShapeFact::from(vec![Some(1), None])
    } else if input.is_some_and(|value| definitely_column(&value.shape)) {
        ShapeFact::from(vec![None, Some(1)])
    } else {
        ShapeFact::from(vec![None, None])
    };
    let one = double(one_shape, input);
    let column = double(ShapeFact::from(vec![None, Some(1)]), input);
    let mut values = input
        .map(|value| {
            let mut result = ValueFact::proven(
                if matches!(value.kind, ValueKindFact::Character) {
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    })
                } else {
                    value.kind.clone()
                },
                ShapeFact::from(vec![None, Some(1)]),
                match value.kind {
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real | NumericDomain::Complex,
                    })
                    | ValueKindFact::Numeric(NumericFact {
                        class:
                            NumericClass::Int8
                            | NumericClass::Int16
                            | NumericClass::Int32
                            | NumericClass::Int64
                            | NumericClass::UInt8
                            | NumericClass::UInt16
                            | NumericClass::UInt32
                            | NumericClass::UInt64,
                        domain: NumericDomain::Real,
                    })
                    | ValueKindFact::Character => StorageFact::Unknown,
                    ValueKindFact::Unknown => StorageFact::Unknown,
                    _ => StorageFact::Dense,
                },
            );
            if !matches!(value.residency, ResidencyFact::Host) {
                result.residency = ResidencyFact::Unknown;
                result.storage = StorageFact::Unknown;
            }
            result
        })
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    if matches!(values.kind, ValueKindFact::Unknown) {
        values = ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    let count = request.outputs.requested.known_count();
    let outputs = match count {
        Some(0 | 1) => vec![one],
        Some(_) => vec![column.clone(), column, values],
        None => {
            let mut joined = one.clone();
            if one.shape != column.shape {
                joined.shape = ShapeFact::from(vec![None, None]);
            }
            vec![joined, column, values]
        }
    };
    let mut contract = CallContract::fixed(outputs);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)));
    finish(entry, request, contract, diagnostics)
}

fn literal_zero(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Bool(false) => true,
        LiteralValue::Integer { .. } => value.integer_value_i128() == Some(0),
        _ => LiteralContext::numeric_from_literal(value) == Some(0.0),
    }
}

fn literal_nonzero(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Bool(true) => true,
        LiteralValue::Integer { .. } => value.integer_value_i128().is_some_and(|n| n != 0),
        _ => LiteralContext::numeric_from_literal(value).is_some_and(|n| n != 0.0),
    }
}

fn definitely_column(shape: &ShapeFact) -> bool {
    matches!(shape.known_dims(), Some(dims) if dims.len() >= 2 && dims[0].is_some_and(|rows| rows != 1))
}
