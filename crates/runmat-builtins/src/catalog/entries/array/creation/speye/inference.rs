use crate::catalog::inference::{finish_fixed, literal_text, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let literals = &request.literals.literal_args;
    let typename = literals.last().and_then(literal_text);
    let dims_end = request
        .arguments
        .len()
        .saturating_sub(usize::from(typename.is_some()));
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = match typename.as_deref() {
        Some(name) if name.eq_ignore_ascii_case("single") => {
            numeric_kind(NumericClass::Single, NumericDomain::Real)
        }
        Some(name) if name.eq_ignore_ascii_case("logical") => ValueKindFact::Logical,
        Some(name) if !name.eq_ignore_ascii_case("double") => ValueKindFact::Unknown,
        _ => numeric_kind(NumericClass::Double, NumericDomain::Real),
    };
    output.shape = match dims_end {
        0 => ShapeFact::from(vec![Some(1), Some(1)]),
        1 => {
            if let Some(pair) = request.literals.numeric_vector_at(0) {
                ShapeFact::from(pair)
            } else if let Some(n) = super::super::shape_facts::scalar_dimension(request, 0) {
                ShapeFact::from(vec![Some(n), Some(n)])
            } else {
                ShapeFact::Ranked { rank: 2 }
            }
        }
        2 => ShapeFact::from(vec![
            super::super::shape_facts::scalar_dimension(request, 0),
            super::super::shape_facts::scalar_dimension(request, 1),
        ]),
        _ => ShapeFact::Unknown,
    };
    output.storage = StorageFact::Sparse;
    output.residency = ResidencyFact::Host;
    finish_fixed(entry, request, output, Vec::new())
}
