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
    if !(1..=4).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PLOTMATRIX-ARITY",
            "plotmatrix accepts matrix operands and optional axes/style controls",
            request.arguments.len(),
        ));
    }
    // Axes aliases and Y-versus-LineSpec are value-selected. Do not fabricate
    // matrix extents without proving that parse; BigAx alone is always scalar.
    finish_fixed_outputs(
        entry,
        request,
        vec![
            output(ShapeFact::Ranked { rank: 2 }),
            output(ShapeFact::Ranked { rank: 2 }),
            output(ShapeFact::Scalar),
            output(ShapeFact::from(vec![Some(1), None])),
            output(ShapeFact::from(vec![Some(1), None])),
        ],
        diagnostics,
    )
}
