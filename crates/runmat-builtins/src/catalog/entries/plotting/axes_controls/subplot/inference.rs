use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-SUBPLOT-ARITY",
            "subplot expects rows, columns, and one-based position",
            0,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if !matches!(
            argument.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        ) || argument
            .shape
            .element_count()
            .is_some_and(|count| count != 1)
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-SUBPLOT-GRID",
                "subplot controls must be host real numeric scalars",
                index,
            ));
        }
    }
    let output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
