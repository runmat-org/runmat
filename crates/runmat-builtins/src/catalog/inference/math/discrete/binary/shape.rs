use crate::catalog::inference::argument_error;
use runmat_types::{
    NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn resolve(
    left: &ValueFact,
    right: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<ShapeFact> {
    if left.shape.is_proven_equivalent(&right.shape) {
        Some(left.shape.clone())
    } else if left.is_scalar() {
        Some(right.shape.clone())
    } else if right.is_scalar() {
        Some(left.shape.clone())
    } else if matches!(left.shape, ShapeFact::Unknown | ShapeFact::Ranked { .. })
        || matches!(right.shape, ShapeFact::Unknown | ShapeFact::Ranked { .. })
    {
        None
    } else {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTEGER-BINARY-SHAPE",
            "inputs must have the same size or one input must be scalar",
            1,
        ));
        None
    }
}

pub(super) fn numeric_output(class: NumericClass, shape: ShapeFact) -> ValueFact {
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    )
}
