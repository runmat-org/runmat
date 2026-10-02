use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn vector_length(shape: &ShapeFact) -> Result<Option<usize>, ()> {
    match shape {
        ShapeFact::Scalar => Ok(Some(1)),
        ShapeFact::Shaped { dims } if dims.len() == 2 => {
            let rows = &dims[0];
            let cols = &dims[1];
            match (rows, cols) {
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
        ShapeFact::Ranked { rank: 2 } | ShapeFact::Unknown => Ok(None),
        _ => Err(()),
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 2 | 3) {
        diagnostics.push(argument_error(
            "RM-CATALOG-NEARESTNEIGHBOR-ARITY",
            "nearestNeighbor requires a DelaunayTri object and query matrix or x,y vectors",
            request.arguments.len().saturating_sub(1),
        ));
    }
    match request.arguments.first().map(|fact| &fact.kind) {
        Some(ValueKindFact::Object(object)) => {
            if object.runtime_class.as_ref().is_some_and(|class| {
                class != &super::super::delaunay_tri::DELAUNAY_TRI_CLASS.owned()
            }) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-NEARESTNEIGHBOR-RECEIVER",
                    "expected a DelaunayTri object",
                    0,
                ));
            }
        }
        Some(ValueKindFact::Unknown) | None => {}
        _ => diagnostics.push(argument_error(
            "RM-CATALOG-NEARESTNEIGHBOR-RECEIVER",
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
                "RM-CATALOG-NEARESTNEIGHBOR-CLASS",
                "query coordinates must be real numeric",
                index,
            )),
        }
        if fact.storage == StorageFact::Sparse {
            diagnostics.push(argument_error(
                "RM-CATALOG-NEARESTNEIGHBOR-STORAGE",
                "query coordinates must be dense",
                index,
            ));
        }
    }
    let rows = match request.arguments.get(1..).unwrap_or(&[]) {
        [matrix] => match &matrix.shape {
            ShapeFact::Shaped { dims } if dims.len() == 2 => {
                if matches!(dims[1], DimensionFact::Known(columns) if columns != 2) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NEARESTNEIGHBOR-SHAPE",
                        "query matrix must be N-by-2",
                        1,
                    ));
                }
                match &dims[0] {
                    DimensionFact::Known(rows) => Some(*rows),
                    _ => None,
                }
            }
            ShapeFact::Ranked { rank: 2 } | ShapeFact::Unknown => None,
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-NEARESTNEIGHBOR-SHAPE",
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
                    "RM-CATALOG-NEARESTNEIGHBOR-SHAPE",
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
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![rows, Some(1)]),
        StorageFact::Dense,
    );
    output.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
