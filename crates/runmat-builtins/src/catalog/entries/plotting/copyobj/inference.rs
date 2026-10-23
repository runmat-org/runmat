use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-COPYOBJ-ARITY",
            "expected source and destination",
            0,
        ));
    }
    for (index, input) in request.arguments.iter().enumerate() {
        if !matches!(
            input.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) || input
            .numeric()
            .is_some_and(|n| n.domain == NumericDomain::Complex)
            || matches!(input.residency, ResidencyFact::Device { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-COPYOBJ-HANDLE",
                "expected host real numeric graphics handles",
                index,
            ));
        }
    }
    if let [source, parent] = request.arguments.as_slice() {
        if source.shape.element_count().is_some_and(|count| count != 0)
            && parent.shape.element_count() == Some(0)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-COPYOBJ-EMPTY-PARENT",
                "parent array must not be empty for a nonempty source",
                1,
            ));
        }
    }
    let shape = match request.arguments.as_slice() {
        [source, parent] if source.shape.element_count() == Some(0) || parent.is_scalar() => {
            source.shape.clone()
        }
        [source, parent] if source.is_scalar() => parent.shape.clone(),
        [source, parent] => {
            if source.shape.element_count().is_some()
                && parent.shape.element_count().is_some()
                && source.shape.element_count() != parent.shape.element_count()
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-COPYOBJ-SHAPE",
                    "nonscalar arrays must have matching element counts",
                    1,
                ));
            }
            source.shape.clone()
        }
        _ => ShapeFact::Unknown,
    };
    let scalar = shape.element_count() == Some(1);
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        if scalar { ShapeFact::Scalar } else { shape },
        if scalar {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        },
    );
    finish_fixed(entry, request, output, diagnostics)
}
