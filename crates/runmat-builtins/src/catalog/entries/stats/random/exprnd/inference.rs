use crate::{
    catalog::inference::{argument_error, finish_fixed, numeric_kind},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(crate) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    let Some(mean) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXPRND-ARITY",
            "exprnd requires a mean",
            0,
        ));
        return finish_fixed(entry, request, output, diagnostics);
    };
    for (index, argument) in request.arguments.iter().enumerate() {
        if !matches!(
            argument.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) || matches!(
            argument.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Complex,
                ..
            })
        ) || argument.storage == StorageFact::Sparse
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-EXPRND-CLASS",
                "exprnd inputs must be dense real numeric values",
                index,
            ));
        }
    }
    if let Some(numeric) = mean.numeric() {
        output.kind = numeric_kind(
            if numeric.class == NumericClass::Single {
                NumericClass::Single
            } else {
                NumericClass::Double
            },
            NumericDomain::Real,
        );
    }
    if let Some(literal) = request.literals.literal_args.first() {
        if invalid_mean(literal) {
            diagnostics.push(argument_error(
                "RM-CATALOG-EXPRND-MEAN",
                "exprnd mean must be finite and positive",
                0,
            ));
        }
    }
    output.shape = if request.arguments.len() == 1 {
        normalized_mean_shape(&mean.shape)
    } else {
        explicit_shape(request, &mut diagnostics)
    };
    if request.arguments.len() > 1
        && mean.shape.element_count().is_some_and(|n| n != 1)
        && fully_known(&mean.shape)
        && fully_known(&output.shape)
        && !mean.shape.is_proven_equivalent(&output.shape)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXPRND-SIZE",
            "explicit size must match nonscalar mean",
            0,
        ));
    }
    // Scalar doubles can use an ambient provider; resident means may gather and
    // remain host-resident if the owner cannot physically represent the result.
    output.residency = if mean.residency == ResidencyFact::Host
        && (mean
            .numeric()
            .is_some_and(|n| n.class == NumericClass::Single)
            || mean.shape.element_count().is_some_and(|n| n != 1))
    {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    output.storage = match (
        output.shape.element_count(),
        output.numeric().map(|n| n.class),
    ) {
        (_, Some(NumericClass::Single)) => StorageFact::Dense,
        (Some(n), _) if n != 1 => StorageFact::Dense,
        _ => StorageFact::Unknown,
    };
    output.layout = LayoutFact::ColumnMajor;
    output.contiguity = ContiguityFact::Contiguous;
    output.view = ViewFact::Materialized;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    finish_fixed(entry, request, output, diagnostics)
}

fn fully_known(shape: &ShapeFact) -> bool {
    shape
        .known_dims()
        .is_some_and(|dims| dims.iter().all(Option::is_some))
}

fn normalized_mean_shape(shape: &ShapeFact) -> ShapeFact {
    let mut dims = match shape {
        ShapeFact::Scalar => return ShapeFact::Scalar,
        ShapeFact::Unknown => return ShapeFact::Unknown,
        ShapeFact::Ranked { rank } if *rank > 2 => return ShapeFact::Unknown,
        ShapeFact::Ranked { rank } => vec![DimensionFact::Unknown; *rank],
        ShapeFact::Shaped { dims } => dims.clone(),
    };
    if dims.is_empty() {
        dims = vec![DimensionFact::Known(1); 2];
    }
    if dims.len() == 1 {
        dims.push(DimensionFact::Known(1));
    }
    while dims.len() > 2 && dims.last() == Some(&DimensionFact::Known(1)) {
        dims.pop();
    }
    // A dynamic trailing extent can become one, changing normalized rank.
    if dims.len() > 2 && !matches!(dims.last(), Some(DimensionFact::Known(_))) {
        return ShapeFact::Unknown;
    }
    ShapeFact::Shaped { dims }
}

fn invalid_mean(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Number(n) => !n.is_finite() || *n <= 0.0,
        LiteralValue::Integer { text, .. } | LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .is_ok_and(|n| !n.is_finite() || n <= 0.0),
        LiteralValue::Vector(values) => values.iter().any(invalid_mean),
        LiteralValue::Matrix(rows) => rows.iter().flatten().any(invalid_mean),
        _ => false,
    }
}

fn explicit_shape(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> ShapeFact {
    let mut dimensions = Vec::new();
    for index in 1..request.arguments.len() {
        let Some(literal) = request.literals.literal_args.get(index) else {
            return ShapeFact::Unknown;
        };
        let values = match literal {
            LiteralValue::Empty => Vec::new(),
            LiteralValue::Vector(values) => values.iter().collect(),
            LiteralValue::Matrix(rows) => {
                let columns = rows.first().map_or(0, Vec::len);
                if rows.iter().any(|row| row.len() != columns) || (rows.len() > 1 && columns > 1) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-EXPRND-DIMENSION",
                        "size must be a scalar or vector",
                        index,
                    ));
                    return ShapeFact::Unknown;
                }
                rows.iter().flatten().collect()
            }
            LiteralValue::Unknown => return ShapeFact::Unknown,
            value => vec![value],
        };
        for value in values {
            match dimension(value) {
                Ok(Some(n)) => dimensions.push(Some(n)),
                Ok(None) => return ShapeFact::Unknown,
                Err(()) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-EXPRND-DIMENSION",
                        "dimensions must be finite integers in range",
                        index,
                    ));
                    return ShapeFact::Unknown;
                }
            }
        }
    }
    if dimensions.is_empty() {
        dimensions = vec![Some(0), Some(0)];
    }
    if dimensions.len() == 1 {
        dimensions.push(dimensions[0]);
    }
    while dimensions.len() > 2 && dimensions.last() == Some(&Some(1)) {
        dimensions.pop();
    }
    ShapeFact::from(dimensions)
}

fn dimension(literal: &LiteralValue) -> Result<Option<usize>, ()> {
    let n = match literal {
        LiteralValue::Integer { text, .. } => {
            return text.parse::<i128>().map_err(|_| ()).and_then(|n| {
                if n <= 0 {
                    Ok(Some(0))
                } else {
                    usize::try_from(n).map(Some).map_err(|_| ())
                }
            })
        }
        LiteralValue::Number(n) => *n,
        LiteralValue::Real { text, .. } => text.parse().map_err(|_| ())?,
        LiteralValue::Unknown => return Ok(None),
        _ => return Err(()),
    };
    if !n.is_finite() || n.fract() != 0.0 || n >= usize::MAX as f64 {
        return Err(());
    }
    Ok(Some(n.max(0.0) as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments: literals
                .iter()
                .map(|_| ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real)))
                .collect(),
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::default(),
        }
    }
    fn infer_request(request: &CallRequest) -> CallInference {
        infer(request, &super::super::EXPRND_CATALOG_ENTRY)
    }
    #[test]
    fn scalar_single_and_array_shapes() {
        let mut r = request(vec![LiteralValue::Unknown]);
        r.arguments[0].kind = numeric_kind(NumericClass::Single, NumericDomain::Real);
        r.arguments[0].shape = ShapeFact::from(vec![Some(2), Some(3)]);
        let out = infer_request(&r).outputs.remove(0);
        assert_eq!(out.shape, r.arguments[0].shape);
        assert_eq!(out.kind, r.arguments[0].kind);
        assert_eq!(out.storage, StorageFact::Dense);
        assert_eq!(out.residency, ResidencyFact::Host);
    }
    #[test]
    fn size_vector_contents_not_shape() {
        let mut r = request(vec![LiteralValue::Number(2.0), LiteralValue::Unknown]);
        r.arguments[1].shape = ShapeFact::from(vec![Some(1), Some(2)]);
        assert_eq!(infer_request(&r).outputs[0].shape, ShapeFact::Unknown);
        r.literals.literal_args[1] = LiteralValue::Vector(vec![
            LiteralValue::Number(3.0),
            LiteralValue::Number(4.0),
            LiteralValue::Number(1.0),
        ]);
        assert_eq!(
            infer_request(&r).outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(4)])
        );
    }
    #[test]
    fn invalid_inputs_and_output_selection() {
        for literals in [
            vec![],
            vec![LiteralValue::Number(0.0)],
            vec![LiteralValue::Number(1.0), LiteralValue::Number(0.5)],
        ] {
            assert!(!infer_request(&request(literals)).diagnostics.is_empty());
        }
        let mut r = request(vec![LiteralValue::Number(1.0)]);
        r.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        assert!(!infer_request(&r).diagnostics.is_empty());
    }

    #[test]
    fn row_and_column_literal_sizes_match_runtime() {
        for rows in [
            vec![vec![LiteralValue::Number(2.0), LiteralValue::Number(3.0)]],
            vec![
                vec![LiteralValue::Number(2.0)],
                vec![LiteralValue::Number(3.0)],
            ],
        ] {
            let r = request(vec![LiteralValue::Number(1.0), LiteralValue::Matrix(rows)]);
            let result = infer_request(&r);
            assert!(result.diagnostics.is_empty());
            assert_eq!(
                result.outputs[0].shape,
                ShapeFact::from(vec![Some(2), Some(3)])
            );
        }
        let matrix = LiteralValue::Matrix(vec![vec![LiteralValue::Number(2.0); 2]; 2]);
        assert!(
            !infer_request(&request(vec![LiteralValue::Number(1.0), matrix]))
                .diagnostics
                .is_empty()
        );
    }

    #[test]
    fn mean_shape_normalizes_minimum_rank_and_trailing_ones() {
        for (shape, expected) in [
            (vec![], vec![Some(1), Some(1)]),
            (vec![Some(3)], vec![Some(3), Some(1)]),
            (vec![Some(2), Some(3), Some(1)], vec![Some(2), Some(3)]),
            (
                vec![Some(2), Some(3), Some(4)],
                vec![Some(2), Some(3), Some(4)],
            ),
        ] {
            let mut r = request(vec![LiteralValue::Unknown]);
            r.arguments[0].shape = ShapeFact::from(shape);
            assert_eq!(
                infer_request(&r).outputs[0].shape,
                ShapeFact::from(expected)
            );
        }
        let mut r = request(vec![LiteralValue::Unknown]);
        r.arguments[0].shape = ShapeFact::from(vec![Some(2), Some(3), None]);
        assert_eq!(infer_request(&r).outputs[0].shape, ShapeFact::Unknown);
    }

    #[test]
    fn unknown_extents_do_not_prove_a_shape_mismatch() {
        let mut r = request(vec![
            LiteralValue::Unknown,
            LiteralValue::Vector(vec![LiteralValue::Number(0.0), LiteralValue::Number(3.0)]),
        ]);
        r.arguments[0].shape = ShapeFact::from(vec![Some(0), None]);
        assert!(infer_request(&r).diagnostics.is_empty());
    }

    #[test]
    fn dynamic_mean_and_size_do_not_invent_class_or_residency() {
        let mut r = request(vec![LiteralValue::Unknown, LiteralValue::Unknown]);
        r.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
        let result = infer_request(&r);
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
        assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    }
}
