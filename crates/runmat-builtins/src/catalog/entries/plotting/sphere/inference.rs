use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut size = if request.arguments.is_empty() {
        Some(super::SPHERE_DEFAULT_N + 1)
    } else {
        None
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SPHERE-ARITY",
            "sphere accepts no input or one face count",
            1,
        ));
    }
    if let Some(n) = request.arguments.first() {
        if !matches!(
            n.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) || n
            .numeric()
            .is_some_and(|x| x.domain == NumericDomain::Complex)
            || matches!(n.storage, StorageFact::Sparse | StorageFact::Opaque)
            || n.shape.element_count().is_some_and(|count| count != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SPHERE-FACE-COUNT",
                "n must be one real numeric whole nonnegative face count",
                0,
            ));
        }
        if let Some(raw) = request.literals.numeric_at(0) {
            // Match the numerical owner's historical tolerance and exact
            // checked grid ceiling; do not infer from a lossy integer cast.
            let rounded = raw.round();
            size = (raw.is_finite()
                && (rounded - raw).abs() <= super::SPHERE_WHOLE_COUNT_TOLERANCE
                && rounded >= 0.0
                && rounded < usize::MAX as f64)
                .then(|| (rounded as usize).checked_add(1))
                .flatten()
                .filter(|size| {
                    size.checked_mul(*size)
                        .is_some_and(|n| n <= super::SPHERE_MAX_COORDINATE_ELEMENTS)
                });
            if size.is_none() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-SPHERE-FACE-COUNT",
                    "n must fit the supported nonnegative whole face-count grid",
                    0,
                ));
            }
        }
    }
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![size, size]),
        StorageFact::Dense,
    );
    output.residency = ResidencyFact::Host;
    finish_fixed_outputs(
        entry,
        request,
        vec![output.clone(), output.clone(), output],
        diagnostics,
    )
}
