use crate::{catalog::inference::finish_fixed, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    // The table object, runtime variable selector, and chart property values
    // are admitted by the single runtime owner. Static facts do not assert a
    // particular grouping or resident transfer from a dynamic table value.
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        Vec::new(),
    )
}
