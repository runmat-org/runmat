use crate::{
    DynamicReason, FactInference, InferenceDiagnostic, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub fn infer_literal(literal: &LiteralValue) -> FactInference {
    let fact = match literal {
        LiteralValue::Number(_) => numeric_scalar(NumericClass::Double, NumericDomain::Real),
        LiteralValue::Real { class, .. } => numeric_scalar(*class, NumericDomain::Real),
        LiteralValue::Integer { class, .. } => numeric_scalar(*class, NumericDomain::Real),
        LiteralValue::Complex { class, .. } => numeric_scalar(*class, NumericDomain::Complex),
        LiteralValue::Bool(_) => ValueFact::scalar(ValueKindFact::Logical),
        LiteralValue::Character(value) => ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(value.chars().count())]),
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
    FactInference::exact(ValueFact::scalar(ValueKindFact::Struct(StructFact {
        fields,
        fields_complete: true,
    })))
}
