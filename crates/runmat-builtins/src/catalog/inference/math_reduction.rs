use super::{argument_error, finish_fixed};
use crate::{BuiltinCatalogEntry, LogicalReductionKind};
use runmat_types::{DynamicReason, LiteralValue, ShapeFact, StorageFact, ValueFact, ValueKindFact};

pub(super) fn infer_logical(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
    _kind: LogicalReductionKind,
) -> runmat_types::CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGICAL-REDUCTION-ARITY",
            format!("{} requires an input value", entry.identity.name),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGICAL-REDUCTION-ARITY",
            format!("{} accepts at most three inputs", entry.identity.name),
            3,
        ));
    }

    if !matches!(
        input.kind,
        ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Character
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LOGICAL-REDUCTION-INPUT",
            format!(
                "{} requires numeric, logical, complex, or character input",
                entry.identity.name
            ),
            0,
        ));
    }

    let shape = reduced_shape(request, &input.shape, &mut diagnostics);
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    let output = ValueFact::proven(ValueKindFact::Logical, shape, storage);
    finish_fixed(entry, request, output, diagnostics)
}

fn reduced_shape(
    request: &runmat_types::CallRequest,
    input: &ShapeFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ShapeFact {
    let Some(mut dimensions) = input.known_dims() else {
        return input
            .rank()
            .map_or(ShapeFact::Unknown, |rank| ShapeFact::Ranked { rank });
    };
    let mut selected = None;
    let mut all_dimensions = false;
    let mut nan_mode_seen = false;

    for (index, argument) in request.literals.literal_args.iter().enumerate().skip(1) {
        match argument {
            LiteralValue::Character(text)
            | LiteralValue::String(text)
            | LiteralValue::Keyword(text)
                if text.eq_ignore_ascii_case("all") =>
            {
                if selected.is_some() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-LOGICAL-REDUCTION-ARGUMENTS",
                        "the `all` selector cannot be combined with dimensions",
                        index,
                    ));
                }
                all_dimensions = true;
            }
            LiteralValue::Character(text)
            | LiteralValue::String(text)
            | LiteralValue::Keyword(text)
                if matches!(text.to_ascii_lowercase().as_str(), "omitnan" | "includenan") =>
            {
                if nan_mode_seen {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-LOGICAL-REDUCTION-ARGUMENTS",
                        "only one NaN handling option may be specified",
                        index,
                    ));
                }
                nan_mode_seen = true;
            }
            LiteralValue::Vector(_) => {
                selected = request.literals.numeric_vector_at(index);
            }
            _ => {
                selected = request
                    .literals
                    .numeric_at(index)
                    .and_then(valid_dimension)
                    .map(|dimension| vec![Some(dimension)]);
            }
        }
    }

    if all_dimensions {
        return ShapeFact::Scalar;
    }
    let selected = match selected {
        Some(values) => values,
        None if request.arguments.len() > 1 && !nan_mode_seen => {
            return ShapeFact::Ranked {
                rank: dimensions.len(),
            };
        }
        None => vec![Some(first_nonsingleton(&dimensions) + 1)],
    };
    for dimension in selected {
        let Some(dimension) = dimension else {
            return ShapeFact::Ranked {
                rank: dimensions.len(),
            };
        };
        let Some(index) = dimension.checked_sub(1) else {
            return ShapeFact::Ranked {
                rank: dimensions.len(),
            };
        };
        if let Some(extent) = dimensions.get_mut(index) {
            *extent = Some(1);
        }
    }
    ShapeFact::from(dimensions)
}

fn valid_dimension(value: f64) -> Option<usize> {
    (value.is_finite() && value.fract() == 0.0 && value >= 1.0).then_some(value as usize)
}

fn first_nonsingleton(dimensions: &[Option<usize>]) -> usize {
    dimensions
        .iter()
        .position(|dimension| !matches!(dimension, Some(1)))
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "math_reduction/tests.rs"]
mod tests;
