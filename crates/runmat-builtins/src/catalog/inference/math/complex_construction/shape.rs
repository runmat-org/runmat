use runmat_types::{InferenceDiagnostic, ShapeFact, ValueFact};

pub(super) fn same_size_or_scalar(
    left: &ValueFact,
    right: &ValueFact,
) -> Result<ShapeFact, InferenceDiagnostic> {
    if left.is_scalar() {
        return Ok(right.shape.clone());
    }
    if right.is_scalar() || left.shape.is_proven_equivalent(&right.shape) {
        return Ok(left.shape.clone());
    }
    if left.shape.known_dims().is_some() && right.shape.known_dims().is_some() {
        return Err(InferenceDiagnostic::error(
            "RM-CATALOG-COMPLEX-SIZE",
            "complex requires equal non-scalar component sizes",
        ));
    }
    Ok(ShapeFact::Unknown)
}
