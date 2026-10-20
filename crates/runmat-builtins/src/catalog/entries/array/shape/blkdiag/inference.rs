use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        return finish_fixed(
            entry,
            request,
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(0), Some(0)]),
                StorageFact::Dense,
            ),
            diagnostics,
        );
    }

    let mut rows = Some(0usize);
    let mut cols = Some(0usize);
    for (index, value) in request.arguments.iter().enumerate() {
        if !matches!(
            &value.kind,
            ValueKindFact::Numeric(_)
                | ValueKindFact::Logical
                | ValueKindFact::Character
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-BLKDIAG-INPUT",
                "blkdiag blocks must be numeric, logical, or character matrices",
                index,
            ));
        }
        let Some((height, width)) = matrix_dims(&value.shape) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-BLKDIAG-SHAPE",
                "blkdiag blocks must be matrices with only singleton trailing dimensions",
                index,
            ));
            rows = None;
            cols = None;
            continue;
        };
        rows = rows.and_then(|total| height.and_then(|height| total.checked_add(height)));
        cols = cols.and_then(|total| width.and_then(|width| total.checked_add(width)));
    }

    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    result.shape = if diagnostics.is_empty() {
        ShapeFact::from(vec![rows, cols])
    } else {
        ShapeFact::Unknown
    };
    if request
        .arguments
        .iter()
        .all(|value| value.kind == request.arguments[0].kind)
    {
        result.kind = request.arguments[0].kind.clone();
    }
    if request
        .arguments
        .iter()
        .all(|value| value.residency == ResidencyFact::Host)
    {
        result.residency = ResidencyFact::Host;
    }
    if request
        .arguments
        .iter()
        .all(|value| matches!(value.storage, StorageFact::Dense | StorageFact::Scalar))
    {
        result.storage = StorageFact::Dense;
    }
    finish_fixed(entry, request, result, diagnostics)
}

fn matrix_dims(shape: &ShapeFact) -> Option<(Option<usize>, Option<usize>)> {
    match shape {
        ShapeFact::Scalar => Some((Some(1), Some(1))),
        ShapeFact::Shaped { dims } => {
            if dims
                .iter()
                .skip(2)
                .any(|dim| matches!(dim, DimensionFact::Known(n) if *n != 1))
            {
                return None;
            }
            if dims
                .iter()
                .skip(2)
                .any(|dim| !matches!(dim, DimensionFact::Known(1)))
            {
                return Some((None, None));
            }
            let known = |index: usize| match dims.get(index) {
                Some(DimensionFact::Known(n)) => Some(*n),
                Some(_) => None,
                None => Some(1),
            };
            if dims.len() == 1 {
                Some((Some(1), known(0)))
            } else {
                Some((known(0), known(1)))
            }
        }
        ShapeFact::Unknown | ShapeFact::Ranked { .. } => Some((None, None)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::entries::array::shape::blkdiag::BLKDIAG_CATALOG_ENTRY;

    #[test]
    fn block_shapes_sum_and_nonmatrix_input_is_diagnosed() {
        let numeric = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        let block = |rows, cols| {
            ValueFact::proven(
                numeric.clone(),
                ShapeFact::from(vec![Some(rows), Some(cols)]),
                StorageFact::Dense,
            )
        };
        let mut request = CallRequest {
            arguments: vec![block(2, 3), ValueFact::scalar(numeric.clone()), block(4, 1)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let known = infer(&request, &BLKDIAG_CATALOG_ENTRY);
        assert!(known.diagnostics.is_empty());
        assert_eq!(
            known.outputs[0].shape,
            ShapeFact::from(vec![Some(7), Some(5)])
        );
        request.arguments[1].shape = ShapeFact::from(vec![Some(2), Some(2), Some(2)]);
        let invalid = infer(&request, &BLKDIAG_CATALOG_ENTRY);
        assert!(!invalid.diagnostics.is_empty());
        assert_eq!(invalid.outputs[0].shape, ShapeFact::Unknown);
    }
}
