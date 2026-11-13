//! Shared static admission for the two triangulation query identities.
use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, DimensionFact, InferenceDiagnostic, NumericDomain, NumericFact, ShapeFact,
    StorageFact, ValueKindFact,
};

fn vector_length(shape: &ShapeFact) -> Result<Option<usize>, ()> {
    match shape {
        ShapeFact::Scalar => Ok(Some(1)),
        ShapeFact::Shaped { dims } if dims.len() == 1 => match &dims[0] {
            DimensionFact::Known(length) => Ok(Some(*length)),
            _ => Ok(None),
        },
        ShapeFact::Shaped { dims } if dims.len() >= 2 && possible_trailing_singletons(dims) => {
            match (&dims[0], &dims[1]) {
                (DimensionFact::Known(1), DimensionFact::Known(length))
                | (DimensionFact::Known(length), DimensionFact::Known(1)) => Ok(Some(*length)),
                (DimensionFact::Known(rows), DimensionFact::Known(cols))
                    if *rows != 1 && *cols != 1 =>
                {
                    Err(())
                }
                _ => Ok(None),
            }
        }
        ShapeFact::Ranked { rank: 1.. } | ShapeFact::Unknown => Ok(None),
        _ => Err(()),
    }
}

pub(super) fn possible_trailing_singletons(dims: &[DimensionFact]) -> bool {
    dims.iter()
        .skip(2)
        .all(|dim| !matches!(dim, DimensionFact::Known(n) if *n != 1))
}

pub(super) fn admit(
    request: &CallRequest,
    code: &'static str,
    name: &'static str,
) -> (Option<usize>, Vec<InferenceDiagnostic>) {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 2 | 3) {
        diagnostics.push(argument_error(
            format!("{code}-ARITY"),
            format!("{name} requires a DelaunayTri object and query matrix or x,y vectors"),
            request.arguments.len().saturating_sub(1),
        ));
    }
    match request.arguments.first().map(|fact| &fact.kind) {
        Some(ValueKindFact::Object(object)) => {
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| class != &super::delaunay_tri::DELAUNAY_TRI_CLASS.owned())
            {
                diagnostics.push(argument_error(
                    format!("{code}-RECEIVER"),
                    "expected a DelaunayTri object",
                    0,
                ));
            }
        }
        Some(ValueKindFact::Unknown) | None => {}
        _ => diagnostics.push(argument_error(
            format!("{code}-RECEIVER"),
            "expected a DelaunayTri object",
            0,
        )),
    }
    for (index, fact) in request.arguments.iter().enumerate().skip(1) {
        match &fact.kind {
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                format!("{code}-CLASS"),
                "query coordinates must be real numeric",
                index,
            )),
        }
        if fact.storage == StorageFact::Sparse {
            diagnostics.push(argument_error(
                format!("{code}-STORAGE"),
                "query coordinates must be dense",
                index,
            ));
        }
    }
    let rows = match request.arguments.get(1..).unwrap_or(&[]) {
        [matrix] => match &matrix.shape {
            ShapeFact::Shaped { dims } if dims.len() == 1 => {
                if matches!(dims[0], DimensionFact::Known(columns) if columns != 2) {
                    diagnostics.push(argument_error(
                        format!("{code}-SHAPE"),
                        "query matrix must be N-by-2",
                        1,
                    ));
                }
                Some(1)
            }
            ShapeFact::Shaped { dims } if dims.len() >= 2 && possible_trailing_singletons(dims) => {
                if matches!(dims[1], DimensionFact::Known(columns) if columns != 2) {
                    diagnostics.push(argument_error(
                        format!("{code}-SHAPE"),
                        "query matrix must be N-by-2",
                        1,
                    ));
                    None
                } else {
                    match &dims[0] {
                        DimensionFact::Known(rows) => Some(*rows),
                        _ => None,
                    }
                }
            }
            ShapeFact::Ranked { rank: 1.. } | ShapeFact::Unknown => None,
            _ => {
                diagnostics.push(argument_error(
                    format!("{code}-SHAPE"),
                    "query matrix must be N-by-2",
                    1,
                ));
                None
            }
        },
        [x, y] => {
            let left = vector_length(&x.shape);
            let right = vector_length(&y.shape);
            if left.is_err()
                || right.is_err()
                || matches!((left, right), (Ok(Some(a)), Ok(Some(b))) if a != b)
            {
                diagnostics.push(argument_error(
                    format!("{code}-SHAPE"),
                    "x and y must be equally sized vectors",
                    1,
                ));
                None
            } else {
                left.ok().flatten().or_else(|| right.ok().flatten())
            }
        }
        _ => None,
    };
    (rows, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_shape_admits_trailing_singletons_but_not_genuine_nd() {
        assert_eq!(
            vector_length(&ShapeFact::from(vec![Some(1), Some(2), Some(1)])),
            Ok(Some(2))
        );
        assert_eq!(
            vector_length(&ShapeFact::from(vec![Some(1), Some(2), Some(2)])),
            Err(())
        );
        assert_eq!(vector_length(&ShapeFact::from(vec![Some(2)])), Ok(Some(2)));
        assert_eq!(vector_length(&ShapeFact::Ranked { rank: 3 }), Ok(None));
    }
}
