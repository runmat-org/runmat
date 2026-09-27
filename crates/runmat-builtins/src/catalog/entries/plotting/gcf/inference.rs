use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        vec![]
    } else {
        vec![argument_error(
            "RM-CATALOG-GCF-ARITY",
            "gcf accepts no inputs",
            0,
        )]
    };
    // A graphics handle is a runtime identity, not a statically known figure number.
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        diagnostics,
    )
}
