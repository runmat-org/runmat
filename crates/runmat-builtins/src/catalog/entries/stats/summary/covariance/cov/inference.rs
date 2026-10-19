use super::CovRowsOption;
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn diagnostic(code: &'static str, message: &'static str, index: usize) -> InferenceDiagnostic {
    argument_error(code, message, index)
}

fn is_text(value: &ValueFact) -> bool {
    matches!(
        &value.kind,
        ValueKindFact::String | ValueKindFact::Character
    )
}

fn numeric_scalar(value: &ValueFact) -> bool {
    value.shape.element_count() == Some(1)
        && matches!(&value.storage, StorageFact::Scalar)
        && matches!(
            &value.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
}

fn valid_normalization(request: &CallRequest, index: usize) -> Option<bool> {
    if let Some(literal) = request.literals.literal_args.get(index) {
        match literal {
            LiteralValue::Integer { text, .. } => {
                return Some(matches!(text.parse::<i128>(), Ok(0 | 1)));
            }
            LiteralValue::Number(_) | LiteralValue::Real { .. } => {
                return request.literals.numeric_at(index).map(|value| {
                    let rounded = value.round();
                    value.is_finite()
                        && (rounded - value).abs() <= 1.0e-12
                        && (rounded == 0.0 || rounded == 1.0)
                });
            }
            LiteralValue::Bool(_) => return Some(true),
            _ => {}
        }
    }
    None
}

fn same_known_size(left: &ValueFact, right: &ValueFact) -> Option<bool> {
    let left_geometry = super::super::geometry::observations(left, 0, false, &mut Vec::new());
    let right_geometry = super::super::geometry::observations(right, 1, false, &mut Vec::new());
    let left_vector = matches!(
        (&left_geometry.rows, &left_geometry.columns),
        (DimensionFact::Known(1), _) | (_, DimensionFact::Known(1))
    );
    let right_vector = matches!(
        (&right_geometry.rows, &right_geometry.columns),
        (DimensionFact::Known(1), _) | (_, DimensionFact::Known(1))
    );
    if left_vector && right_vector {
        Some(left.shape.element_count()? == right.shape.element_count()?)
    } else if matches!(
        (&left.shape, &right.shape),
        (ShapeFact::Scalar, ShapeFact::Scalar)
    ) {
        Some(true)
    } else if let (ShapeFact::Shaped { dims: left }, ShapeFact::Shaped { dims: right }) =
        (&left.shape, &right.shape)
    {
        if left
            .iter()
            .chain(right.iter())
            .any(|dimension| !matches!(dimension, DimensionFact::Known(_)))
        {
            None
        } else {
            Some(left == right)
        }
    } else {
        None
    }
}

fn dataset_or_weight(
    first: &ValueFact,
    candidate: &ValueFact,
    normalization_explicit: bool,
    rows_option: Option<CovRowsOption>,
) -> Option<bool> {
    // Runtime resolves a single array argument as a weight vector when its
    // length equals the observation count, except for the vector/vector
    // cov(x,y) form. Unknown geometry stays dynamic rather than claiming 2x2.
    let first_geometry = super::super::geometry::observations(first, 0, false, &mut Vec::new());
    let candidate_geometry =
        super::super::geometry::observations(candidate, 1, false, &mut Vec::new());
    let DimensionFact::Known(rows) = first_geometry.rows else {
        return None;
    };
    let DimensionFact::Known(columns) = first_geometry.columns else {
        return None;
    };
    let DimensionFact::Known(candidate_rows) = candidate_geometry.rows else {
        return None;
    };
    let DimensionFact::Known(candidate_columns) = candidate_geometry.columns else {
        return None;
    };
    let candidate_vector = candidate_rows == 1 || candidate_columns == 1;
    let weight_length = candidate_rows == rows || candidate_columns == rows;
    if candidate_vector && weight_length {
        rows_option.map(|rows_option| {
            (rows != 1 && columns != 1)
                || normalization_explicit
                || rows_option != CovRowsOption::All
        })
    } else {
        Some(false)
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(first) = request.arguments.first() else {
        diagnostics.push(diagnostic(
            "RM-CATALOG-COV-ARITY",
            "cov requires observations",
            0,
        ));
        let mut result = infer_call(
            &CallContract::fixed(vec![ValueFact::unknown(DynamicReason::RuntimeValue)]),
            request,
        );
        result.diagnostics.extend(diagnostics);
        return result;
    };
    let geometry = super::super::geometry::observations(first, 0, false, &mut diagnostics);
    let mut arrays = Vec::new();
    let mut normalization_explicit = false;
    let mut rows_option = Some(CovRowsOption::All);
    let mut unknown = false;
    for (index, value) in request.arguments.iter().enumerate().skip(1) {
        if is_text(value) {
            if let Some(text) = request.literals.literal_string_at(index) {
                let parsed = CovRowsOption::parse(&text);
                if parsed.is_none() {
                    diagnostics.push(diagnostic(
                        "RM-CATALOG-COV-ROWS",
                        "unsupported rows option",
                        index,
                    ));
                }
                rows_option = parsed;
            } else {
                rows_option = None;
            }
        } else if numeric_scalar(value) {
            if normalization_explicit {
                diagnostics.push(diagnostic(
                    "RM-CATALOG-COV-NORMALIZATION",
                    "normalization flag cannot be repeated",
                    index,
                ));
            }
            normalization_explicit = true;
            if valid_normalization(request, index) == Some(false) {
                diagnostics.push(diagnostic(
                    "RM-CATALOG-COV-NORMALIZATION",
                    "normalization flag must be zero or one",
                    index,
                ));
            }
        } else if matches!(
            &value.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        ) {
            arrays.push((index, value));
        } else if matches!(&value.kind, ValueKindFact::Unknown) {
            unknown = true;
        } else {
            diagnostics.push(diagnostic(
                "RM-CATALOG-COV-ARGUMENT",
                "unsupported covariance argument",
                index,
            ));
        }
    }
    if arrays.len() > 2 {
        diagnostics.push(diagnostic(
            "RM-CATALOG-COV-ARITY",
            "at most two array arguments may follow X",
            arrays[2].0,
        ));
    }
    let mut width = geometry.variables;
    let mut paired_observations = None;
    let mut precision_ambiguous = false;
    if let Some((index, second)) = arrays.first() {
        let second_geometry =
            super::super::geometry::observations(second, *index, false, &mut diagnostics);
        let weight = if arrays.len() > 1 {
            Some(false)
        } else {
            dataset_or_weight(first, second, normalization_explicit, rows_option)
        };
        match weight {
            Some(true) => {
                if let (DimensionFact::Known(rows), Some(length)) =
                    (&geometry.rows, second.shape.element_count())
                {
                    if length == 0 {
                        diagnostics.push(diagnostic(
                            "RM-CATALOG-COV-WEIGHTS",
                            "weight vector cannot be empty",
                            *index,
                        ));
                    }
                    if *rows != length {
                        diagnostics.push(diagnostic(
                            "RM-CATALOG-COV-WEIGHTS",
                            "weight length must equal observation count",
                            *index,
                        ));
                    }
                }
            }
            Some(false) => {
                paired_observations = Some(second);
                width = DimensionFact::Known(2);
                if same_known_size(first, second) == Some(false) {
                    diagnostics.push(diagnostic(
                        "RM-CATALOG-COV-SIZE",
                        "paired data must have matching size",
                        *index,
                    ));
                }
            }
            None => {
                width = DimensionFact::Unknown;
                precision_ambiguous = true;
            }
        }
        if arrays.len() > 1 {
            let (weight_index, weights) = arrays[1];
            let weight_geometry = super::super::geometry::observations(
                weights,
                weight_index,
                false,
                &mut diagnostics,
            );
            if let (DimensionFact::Known(rows), Some(length)) =
                (&geometry.rows, weights.shape.element_count())
            {
                if length == 0 {
                    diagnostics.push(diagnostic(
                        "RM-CATALOG-COV-WEIGHTS",
                        "weight vector cannot be empty",
                        weight_index,
                    ));
                }
                if *rows != length {
                    diagnostics.push(diagnostic(
                        "RM-CATALOG-COV-WEIGHTS",
                        "weight length must equal observation count",
                        weight_index,
                    ));
                }
            }
            if matches!(weight_geometry.variables, DimensionFact::Known(n) if n > 1) {
                diagnostics.push(diagnostic(
                    "RM-CATALOG-COV-WEIGHTS",
                    "weights must be a vector",
                    weight_index,
                ));
            }
        }
        let _ = second_geometry;
    }
    if unknown {
        width = DimensionFact::Unknown;
    }
    let shape = super::super::geometry::square_shape(width);
    let class = if unknown || precision_ambiguous {
        None
    } else {
        paired_observations.map_or_else(
            || super::super::geometry::output_class(&[first]),
            |second| super::super::geometry::output_class(&[first, second]),
        )
    };
    let mut output =
        super::super::geometry::numeric_output(first, shape, NumericDomain::Real, class);
    if arrays
        .iter()
        .any(|(_, value)| !matches!(&value.residency, ResidencyFact::Host))
    {
        output.residency = ResidencyFact::Unknown;
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
