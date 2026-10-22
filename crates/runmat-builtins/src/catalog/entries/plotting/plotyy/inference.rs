use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
fn output(shape: ShapeFact) -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    fact.residency = ResidencyFact::Host;
    fact
}
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(4..=6).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PLOTYY-ARITY",
            "plotyy requires four coordinates and at most two selectors",
            request.arguments.len(),
        ));
    }
    finish_fixed_outputs(
        entry,
        request,
        vec![
            output(ShapeFact::from(vec![Some(1), Some(2)])),
            output(ShapeFact::Unknown),
            output(ShapeFact::Unknown),
        ],
        diagnostics,
    )
}
