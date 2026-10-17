use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut rows = Some(2);
    let mut columns = Some(21);
    let coordinate_form = request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 0);
    if (!coordinate_form && !request.arguments.is_empty()) || request.arguments.len() > 2 {
        // A leading numeric axes handle and trailing surface properties are
        // classified by the runtime graphics owner, not by tensor shape.
        rows = None;
        columns = None;
    } else if let Some(radius) = request.arguments.first() {
        if !matches!(
            radius.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Unknown
        ) || matches!(
            radius.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        ) || radius
            .numeric()
            .is_some_and(|numeric| numeric.domain == NumericDomain::Complex)
            || matches!(radius.storage, StorageFact::Sparse | StorageFact::Opaque)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-CYLINDER-RADIUS",
                "cylinder requires a host real dense radius profile",
                0,
            ));
        }
        if let Some(dims) = radius.shape.known_dims() {
            if dims.len() > 2
                || (dims.len() == 2
                    && dims[0].is_some_and(|n| n > 1)
                    && dims[1].is_some_and(|n| n > 1))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CYLINDER-VECTOR",
                    "radius profile must be a scalar or vector",
                    0,
                ));
            }
        }
        rows = radius
            .shape
            .element_count()
            .map(|n| if n == 1 { 2 } else { n });
    }
    if coordinate_form && request.arguments.len() <= 2 {
        if let Some(resolution) = request.arguments.get(1) {
            columns = request.literals.numeric_at(1).and_then(|raw| {
                (raw.is_finite() && raw >= 1.0 && raw.fract() == 0.0 && raw < usize::MAX as f64)
                    .then(|| (raw as usize).checked_add(1))
                    .flatten()
            });
            if !matches!(
                resolution.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Unknown
            ) || resolution
                .numeric()
                .is_some_and(|numeric| numeric.domain == NumericDomain::Complex)
                || resolution.shape.element_count().is_some_and(|n| n != 1)
                || matches!(
                    resolution.residency,
                    ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
                )
                || request
                    .literals
                    .numeric_at(1)
                    .is_some_and(|n| !n.is_finite() || n < 1.0 || n.fract() != 0.0)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CYLINDER-RESOLUTION",
                    "n must be a host finite positive whole scalar",
                    1,
                ));
            }
        }
    }
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![rows, columns]),
        StorageFact::Dense,
    );
    finish_fixed_outputs(
        entry,
        request,
        vec![output.clone(), output.clone(), output],
        diagnostics,
    )
}
