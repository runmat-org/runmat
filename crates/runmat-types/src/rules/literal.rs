use crate::{
    DynamicReason, FactInference, InferenceDiagnostic, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

/// Source-known scalar arithmetic using the host's represented precision rules.
/// Unsupported classes/operations and non-finite results retain no exact value.
pub fn infer_binary_literal(
    operator: crate::OperatorKind,
    left: &LiteralValue,
    right: &LiteralValue,
) -> LiteralValue {
    let left_fact = infer_literal(left).fact;
    let right_fact = infer_literal(right).fact;
    if !left_fact.is_scalar() || !right_fact.is_scalar() {
        return LiteralValue::Unknown;
    }
    let operator = match operator {
        crate::OperatorKind::Mrdivide => crate::OperatorKind::ElementwiseDivide,
        crate::OperatorKind::Mldivide => crate::OperatorKind::ElementwiseLeftDivide,
        other => other,
    };
    let Some(recipe) = crate::RealFloatingRecipe::for_facts(operator, &left_fact, &right_fact)
    else {
        return LiteralValue::Unknown;
    };
    let Some((left, right)) = crate::LiteralContext::numeric_from_literal(left)
        .zip(crate::LiteralContext::numeric_from_literal(right))
        .filter(|(left, right)| left.is_finite() && right.is_finite())
    else {
        return LiteralValue::Unknown;
    };
    recipe
        .evaluate(left, right)
        .filter(|value| value.is_finite())
        .map_or(LiteralValue::Unknown, |value| LiteralValue::Real {
            text: value.to_string(),
            class: recipe.result_class(),
        })
}

pub fn infer_unary_literal(operator: crate::OperatorKind, operand: &LiteralValue) -> LiteralValue {
    match (operator, operand) {
        (crate::OperatorKind::UnaryPlus, _) => operand.clone(),
        (crate::OperatorKind::UnaryMinus, LiteralValue::Number(value)) if value.is_finite() => {
            LiteralValue::Number(-value)
        }
        (crate::OperatorKind::UnaryMinus, LiteralValue::Real { class, .. })
            if matches!(class, NumericClass::Double | NumericClass::Single) =>
        {
            crate::LiteralContext::numeric_from_literal(operand)
                .filter(|value| value.is_finite())
                .map_or(LiteralValue::Unknown, |value| LiteralValue::Real {
                    text: (-value).to_string(),
                    class: *class,
                })
        }
        _ => LiteralValue::Unknown,
    }
}

pub fn infer_literal(literal: &LiteralValue) -> FactInference {
    let fact = match literal {
        LiteralValue::Number(_) => numeric_scalar(NumericClass::Double, NumericDomain::Real),
        LiteralValue::Real { class, .. } => numeric_scalar(*class, NumericDomain::Real),
        LiteralValue::Integer { class, .. } => numeric_scalar(*class, NumericDomain::Real),
        LiteralValue::Complex { class, .. } => numeric_scalar(*class, NumericDomain::Complex),
        LiteralValue::Bool(_) => ValueFact::scalar(ValueKindFact::Logical),
        LiteralValue::Character(value) => ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(value.encode_utf16().count())]),
            StorageFact::Dense,
        ),
        LiteralValue::String(_) | LiteralValue::Keyword(_) => {
            ValueFact::scalar(ValueKindFact::String)
        }
        LiteralValue::Symbolic(_) => ValueFact::scalar(ValueKindFact::Symbolic),
        LiteralValue::Empty => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(0), Some(0)]),
            StorageFact::Dense,
        ),
        LiteralValue::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
        LiteralValue::Vector(values) => return infer_vector(values),
        LiteralValue::Matrix(rows) => return infer_matrix(rows),
    };
    FactInference::exact(fact)
}

fn infer_vector(values: &[LiteralValue]) -> FactInference {
    let inferred = values.iter().map(infer_literal).collect::<Vec<_>>();
    let diagnostics = inferred
        .iter()
        .flat_map(|value| value.diagnostics.clone())
        .collect();
    let facts = inferred
        .into_iter()
        .map(|value| value.fact)
        .collect::<Vec<_>>();
    let kind = homogeneous_kind(&facts);
    // A literal vector is an array value/context token, never an implicit
    // cell. Heterogeneous or partially unknown element categories retain the
    // proven vector shape while widening only the kind.
    let fact = ValueFact::proven(
        kind,
        ShapeFact::from(vec![Some(1), Some(values.len())]),
        StorageFact::Dense,
    );
    FactInference { fact, diagnostics }
}

fn infer_matrix(rows: &[Vec<LiteralValue>]) -> FactInference {
    let columns = rows.first().map(Vec::len).unwrap_or(0);
    if rows.iter().any(|row| row.len() != columns) {
        return FactInference {
            fact: ValueFact::unknown(DynamicReason::ConflictingControlFlow),
            diagnostics: vec![InferenceDiagnostic::error(
                "RM-TYPE-LITERAL-MATRIX",
                "matrix literal rows must have equal widths",
            )],
        };
    }
    let facts = rows
        .iter()
        .flatten()
        .map(|literal| infer_literal(literal).fact)
        .collect::<Vec<_>>();
    FactInference::exact(ValueFact::proven(
        homogeneous_kind(&facts),
        ShapeFact::from(vec![Some(rows.len()), Some(columns)]),
        StorageFact::Dense,
    ))
}

fn homogeneous_kind(facts: &[ValueFact]) -> ValueKindFact {
    let Some(first) = facts.first() else {
        return ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
    };
    if facts.iter().all(|fact| fact.kind == first.kind) {
        first.kind.clone()
    } else {
        ValueKindFact::Unknown
    }
}

fn numeric_scalar(class: NumericClass, domain: NumericDomain) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact { class, domain }))
}

pub fn infer_struct_literal(fields: BTreeMap<String, ValueFact>) -> FactInference {
    FactInference::exact(ValueFact::scalar(ValueKindFact::Struct(
        StructFact::scalar(fields, true),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LiteralContext, OperatorKind};

    #[test]
    fn scalar_literal_arithmetic_preserves_represented_precision_and_unknown_boundaries() {
        let difference = infer_binary_literal(
            OperatorKind::Subtract,
            &LiteralValue::Number(50.0),
            &LiteralValue::Number(1.0),
        );
        assert_eq!(
            LiteralContext::numeric_from_literal(&difference),
            Some(49.0)
        );
        let single = LiteralValue::Real {
            text: "16777216".into(),
            class: NumericClass::Single,
        };
        let rounded = infer_binary_literal(OperatorKind::Add, &single, &LiteralValue::Number(1.0));
        assert!(matches!(
            rounded,
            LiteralValue::Real {
                class: NumericClass::Single,
                ..
            }
        ));
        assert_eq!(
            LiteralContext::numeric_from_literal(&rounded),
            Some(16777216.0)
        );
        let next = infer_binary_literal(
            OperatorKind::Subtract,
            &rounded,
            &LiteralValue::Number(16777216.0),
        );
        assert_eq!(LiteralContext::numeric_from_literal(&next), Some(0.0));
        for left in [
            LiteralValue::Unknown,
            LiteralValue::Integer {
                text: "18446744073709551615".into(),
                class: NumericClass::UInt64,
            },
        ] {
            assert_eq!(
                infer_binary_literal(OperatorKind::Subtract, &left, &LiteralValue::Number(1.0)),
                LiteralValue::Unknown
            );
        }
        assert_eq!(
            infer_binary_literal(
                OperatorKind::Mrdivide,
                &LiteralValue::Number(1.0),
                &LiteralValue::Number(0.0)
            ),
            LiteralValue::Unknown
        );
        let negative = infer_unary_literal(OperatorKind::UnaryMinus, &single);
        assert!(matches!(
            negative,
            LiteralValue::Real {
                class: NumericClass::Single,
                ..
            }
        ));
        assert_eq!(
            LiteralContext::numeric_from_literal(&negative),
            Some(-16777216.0)
        );
        assert_eq!(
            infer_unary_literal(
                OperatorKind::UnaryMinus,
                &LiteralValue::Integer {
                    text: "18446744073709551615".into(),
                    class: NumericClass::UInt64
                }
            ),
            LiteralValue::Unknown
        );
    }
}
