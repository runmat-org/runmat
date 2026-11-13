use runmat_types::{ShapeFact, ValueFact, ValueKindFact};

/// The runtime image I/O boundary accepts a scalar string or one-row char
/// vector. Unknown extents remain possible, not silently rejected.
pub(super) fn could_be_scalar_text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::String => !value.shape.is_definitely_nonscalar(),
        ValueKindFact::Character => match &value.shape {
            ShapeFact::Shaped { dims } => !dims.first().is_some_and(
                |dim| matches!(dim, runmat_types::DimensionFact::Known(rows) if *rows != 1),
            ),
            _ => true,
        },
        ValueKindFact::Unknown => true,
        _ => false,
    }
}
