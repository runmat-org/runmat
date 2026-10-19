use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone)]
enum Repetitions {
    Scalar(usize),
    Vector(Vec<usize>),
    Dynamic,
    Invalid,
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(source) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-REPELEM-ARITY",
            "repelem requires an input and at least one replication control",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-REPELEM-ARITY",
            "repelem requires at least one replication control",
            1,
        ));
    }
    if !matches!(
        &source.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::String
            | ValueKindFact::Cell(_)
            | ValueKindFact::Struct(_)
    ) || source.storage == StorageFact::Sparse
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-REPELEM-INPUT",
            "unsupported repelem input representation",
            0,
        ));
    }
    let repetitions: Vec<_> = request
        .arguments
        .iter()
        .enumerate()
        .skip(1)
        .map(|(index, control)| {
            let invalid_fact = known_nonvector(&control.shape)
                || !matches!(
                    &control.kind,
                    ValueKindFact::Unknown
                        | ValueKindFact::Logical
                        | ValueKindFact::Numeric(NumericFact {
                            domain: NumericDomain::Real,
                            ..
                        })
                );
            let parsed = if invalid_fact {
                Repetitions::Invalid
            } else {
                request
                    .literals
                    .literal_args
                    .get(index)
                    .map_or(Repetitions::Dynamic, literal_repetitions)
            };
            if matches!(parsed, Repetitions::Invalid) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-REPELEM-CONTROL",
                    "replication controls must be nonnegative integer scalars or vectors",
                    index,
                ));
            }
            parsed
        })
        .collect();

    let mut output = source.clone();
    output.shape = infer_shape(&source.shape, &repetitions, &mut diagnostics);
    output.alias = AliasFact::Unknown;
    output.view = ViewFact::Unknown;
    output.layout = LayoutFact::Unknown;
    output.contiguity = ContiguityFact::Unknown;
    if source.storage == StorageFact::Scalar {
        output.storage = StorageFact::Dense;
    }
    if let ValueKindFact::Cell(cell) = &mut output.kind {
        cell.elements.clear();
        cell.elements_complete = false;
    }
    if let ValueKindFact::Struct(structure) = &mut output.kind {
        structure.elements.clear();
        structure.elements_complete = false;
    }
    if !diagnostics.is_empty() {
        output.shape = ShapeFact::Unknown;
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn literal_repetitions(literal: &LiteralValue) -> Repetitions {
    match literal {
        LiteralValue::Unknown => Repetitions::Dynamic,
        LiteralValue::Vector(values) => values
            .iter()
            .map(nonnegative_integer)
            .collect::<Option<Vec<_>>>()
            .map_or(Repetitions::Invalid, Repetitions::Vector),
        other => nonnegative_integer(other).map_or(Repetitions::Invalid, Repetitions::Scalar),
    }
}

fn nonnegative_integer(literal: &LiteralValue) -> Option<usize> {
    match literal {
        LiteralValue::Bool(value) => Some(usize::from(*value)),
        LiteralValue::Integer { .. } => literal
            .integer_value_i128()
            .and_then(|value| usize::try_from(value).ok()),
        LiteralValue::Number(value) => floating_count(*value),
        LiteralValue::Real { text, .. } => text.parse::<f64>().ok().and_then(floating_count),
        _ => None,
    }
}

fn floating_count(value: f64) -> Option<usize> {
    (value.is_finite()
        && value.fract() == 0.0
        && value >= 0.0
        && value < (usize::MAX as u128 + 1) as f64)
        .then_some(value as usize)
}

fn infer_shape(
    source: &ShapeFact,
    repetitions: &[Repetitions],
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ShapeFact {
    if matches!(source, ShapeFact::Ranked { .. }) {
        return ShapeFact::Unknown;
    }
    let Some(mut dims) = source.known_dims() else {
        return ShapeFact::Unknown;
    };
    if dims.is_empty() {
        dims = vec![Some(1), Some(1)];
    } else if dims.len() == 1 {
        dims.push(Some(1));
    }
    if repetitions.len() == 1 {
        let definite_non_singleton: Vec<_> = dims
            .iter()
            .enumerate()
            .filter_map(|(index, dim)| dim.is_some_and(|size| size > 1).then_some(index))
            .collect();
        if definite_non_singleton.len() > 1 {
            diagnostics.push(InferenceDiagnostic::error(
                "RM-CATALOG-REPELEM-VECTOR",
                "the one-control form requires vector input",
            ));
            return ShapeFact::Unknown;
        }
        if dims.iter().any(Option::is_none) {
            return ShapeFact::Unknown;
        }
        let axis = definite_non_singleton.first().copied().unwrap_or_else(|| {
            if dims[0] == Some(0) {
                0
            } else {
                1
            }
        });
        dims[axis] = expanded(dims[axis], &repetitions[0], diagnostics);
    } else {
        let rank = dims.len().max(repetitions.len()).max(2);
        dims.resize(rank, Some(1));
        for (axis, repetition) in repetitions.iter().enumerate() {
            dims[axis] = expanded(dims[axis], repetition, diagnostics);
        }
    }
    ShapeFact::from(dims)
}

fn known_nonvector(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Shaped { dims } => {
            dims.iter()
                .filter(|dim| matches!(dim, DimensionFact::Known(size) if *size > 1))
                .count()
                > 1
        }
        _ => false,
    }
}

fn expanded(
    dimension: Option<usize>,
    repetition: &Repetitions,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    match repetition {
        Repetitions::Scalar(count) => dimension.and_then(|size| size.checked_mul(*count)),
        Repetitions::Vector(values) => {
            if dimension.is_some_and(|size| values.len() != size) {
                diagnostics.push(InferenceDiagnostic::error(
                    "RM-CATALOG-REPELEM-COUNT-LENGTH",
                    "count-vector length must match its source dimension",
                ));
            }
            values
                .iter()
                .try_fold(0usize, |total, value| total.checked_add(*value))
        }
        Repetitions::Dynamic | Repetitions::Invalid => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::entries::array::shape::repelem::REPELEM_CATALOG_ENTRY;

    #[test]
    fn scalar_and_vector_counts_keep_exact_geometry_without_guessing_dynamic_counts() {
        let numeric = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        let source = ValueFact::proven(
            numeric.clone(),
            ShapeFact::from(vec![Some(1), Some(3)]),
            StorageFact::Dense,
        );
        let control = ValueFact::proven(
            numeric,
            ShapeFact::from(vec![Some(1), Some(3)]),
            StorageFact::Dense,
        );
        let mut request = CallRequest {
            arguments: vec![source, control],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Vector(vec![
                    LiteralValue::Number(1.0),
                    LiteralValue::Number(2.0),
                    LiteralValue::Number(3.0),
                ]),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let exact = infer(&request, &REPELEM_CATALOG_ENTRY);
        assert!(exact.diagnostics.is_empty());
        assert_eq!(
            exact.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(6)])
        );
        request.literals.literal_args[1] = LiteralValue::Unknown;
        let dynamic = infer(&request, &REPELEM_CATALOG_ENTRY);
        assert_eq!(
            dynamic.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }
}
