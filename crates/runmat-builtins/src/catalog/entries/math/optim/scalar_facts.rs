//! Static facts shared by the two callback-driven scalar Brent solvers.

use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, DynamicReason, InferenceDiagnostic, NumericClass, NumericDomain, NumericFact,
    ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn callback(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(value) = request.arguments.first() else {
        return;
    };
    if !matches!(
        value.kind,
        ValueKindFact::Callable(_)
            | ValueKindFact::Object(_)
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SCALAR-OPT-CALLBACK",
            "the first argument must identify a callable scalar function",
            0,
        ));
    }
}

pub(super) fn real_numeric(
    value: Option<&ValueFact>,
    index: usize,
    allowed_counts: &[usize],
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(value) = value else { return };
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) || value
        .shape
        .element_count()
        .is_some_and(|count| !allowed_counts.contains(&count))
        || value.storage == StorageFact::Sparse
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SCALAR-OPT-INPUT",
            "solver input must be a real numeric scalar or supported interval",
            index,
        ));
    }
}

pub(super) fn options(
    value: Option<&ValueFact>,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(value) = value else { return };
    if !matches!(
        value.kind,
        ValueKindFact::Struct(_) | ValueKindFact::Unknown
    ) || value.shape.element_count().is_some_and(|count| count != 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SCALAR-OPT-OPTIONS",
            "options must be a scalar struct",
            index,
        ));
    }
}

pub(super) fn host_scalar(class: NumericClass) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }))
}

pub(super) fn conditional_host_scalar() -> ValueFact {
    let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
    fact.shape = ShapeFact::Scalar;
    fact.storage = StorageFact::Scalar;
    fact.residency = runmat_types::ResidencyFact::Host;
    fact
}

pub(super) fn diagnostic_struct() -> ValueFact {
    let mut fields = BTreeMap::new();
    fields.insert("iterations".into(), host_scalar(NumericClass::Double));
    fields.insert("funcCount".into(), host_scalar(NumericClass::Double));
    fields.insert("algorithm".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert("message".into(), ValueFact::scalar(ValueKindFact::String));
    ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(fields, true)),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    )
}
