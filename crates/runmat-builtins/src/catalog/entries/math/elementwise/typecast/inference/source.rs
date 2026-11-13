use super::representation::lanes;
use crate::catalog::inference::argument_error;
use runmat_types::{
    InferenceDiagnostic, NumericDomain, NumericFact, ResidencyFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn validate(source: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if matches!(source.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TYPECAST-SPARSE",
            "typecast requires full input",
            0,
        ));
    }
    if !matches!(
        source.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TYPECAST-INPUT",
            "typecast requires numeric or logical input",
            0,
        ));
    }
    if source.shape.known_dims().is_some_and(|dims| {
        dims.into_iter()
            .flatten()
            .filter(|dimension| *dimension > 1)
            .count()
            > 1
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TYPECAST-SHAPE",
            "typecast input must be a scalar or vector",
            0,
        ));
    }
    if matches!(source.residency, ResidencyFact::Device { .. })
        && !matches!(
            source.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Unknown
        )
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-TYPECAST-GPU-INPUT",
            "typecast accepts only real numeric provider-resident input",
            0,
        ));
    }
}

pub(super) fn byte_width(source: &ValueFact) -> Option<usize> {
    match source.kind {
        ValueKindFact::Numeric(fact) => Some(fact.class.byte_width() * lanes(fact.domain)),
        ValueKindFact::Logical => Some(1),
        _ => None,
    }
}
