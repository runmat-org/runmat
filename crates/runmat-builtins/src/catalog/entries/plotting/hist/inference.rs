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
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-HIST-ARITY",
            "hist requires sample data",
            0,
        ));
    }
    // The actual bin owner accepts positional counts/centres and modern named
    // controls. Only infer the uncomplicated positional forms, never a name.
    let bins = if request.arguments.len() == 1 {
        Some(10)
    } else if request.arguments.len() == 2 {
        request
            .literals
            .numeric_at(1)
            .filter(|x| x.is_finite() && *x >= 1.0 && x.fract() == 0.0 && *x < usize::MAX as f64)
            .map(|x| x as usize)
            .or_else(|| {
                request
                    .arguments
                    .get(1)
                    .filter(|x| x.numeric().is_some())
                    .and_then(|x| x.shape.element_count())
                    .filter(|x| *x > 1)
            })
    } else {
        None
    };
    let shape = request
        .arguments
        .first()
        .and_then(|arg| arg.shape.known_dims())
        .map(|dims| {
            if dims.len() <= 1 || (dims.len() == 2 && (dims[0] == Some(1) || dims[1] == Some(1))) {
                ShapeFact::from(vec![Some(1), bins])
            } else if dims.len() >= 2 {
                ShapeFact::from(vec![
                    bins,
                    dims[1..]
                        .iter()
                        .try_fold(1usize, |n, x| n.checked_mul((*x)?)),
                ])
            } else {
                ShapeFact::Unknown
            }
        })
        .unwrap_or(ShapeFact::Unknown);
    let fact = output(shape);
    finish_fixed_outputs(entry, request, vec![fact.clone(), fact], diagnostics)
}
