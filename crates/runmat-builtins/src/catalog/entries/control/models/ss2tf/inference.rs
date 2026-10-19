use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 4 | 5) {
        diagnostics.push(argument_error(
            "RM-CATALOG-SS2TF-ARITY",
            "ss2tf expects four matrices and an optional input index",
            0,
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SS2TF-OUTPUTS",
            "ss2tf returns at most two outputs",
            0,
        ));
    }
    let matrices = request.arguments.iter().take(4).collect::<Vec<_>>();
    for (index, value) in matrices.iter().enumerate() {
        let valid = match &value.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Numeric(numeric) => {
                numeric.domain == NumericDomain::Real
                    && matches!(numeric.class, NumericClass::Single | NumericClass::Double)
                    && value.storage != StorageFact::Sparse
            }
            _ => false,
        };
        if !valid || value.shape.known_dims().is_some_and(|dims| dims.len() > 2) {
            diagnostics.push(argument_error(
                "RM-CATALOG-SS2TF-MATRIX",
                "ss2tf requires real Single or Double matrices",
                index,
            ));
        }
    }
    let class = if matrices.len() == 4 {
        let classes = matrices
            .iter()
            .map(|value| match &value.kind {
                ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
                    Some(numeric.class)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if classes
            .iter()
            .all(|class| *class == Some(NumericClass::Single))
        {
            Some(NumericClass::Single)
        } else if classes.iter().all(Option::is_some) {
            Some(NumericClass::Double)
        } else {
            None
        }
    } else {
        None
    };
    let output_rows = matrices
        .get(2)
        .and_then(|value| matrix_dims(value))
        .and_then(|dims| dims.0);
    let degree = matrices
        .first()
        .and_then(|value| matrix_dims(value))
        .and_then(|dims| dims.0)
        .and_then(|states| states.checked_add(1));
    let coefficients = |rows| match class {
        Some(class) => ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![rows, degree]),
            StorageFact::Dense,
        ),
        None => {
            let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
            result.shape = ShapeFact::from(vec![rows, degree]);
            result.storage = StorageFact::Dense;
            result
        }
    };
    let mut contract = CallContract::fixed(vec![coefficients(output_rows), coefficients(Some(1))]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn matrix_dims(value: &ValueFact) -> Option<(Option<usize>, Option<usize>)> {
    let dims = value.shape.known_dims()?;
    match dims.as_slice() {
        [] => Some((Some(1), Some(1))),
        [columns] => Some((Some(1), *columns)),
        [rows, columns] => Some((*rows, *columns)),
        _ => None,
    }
}
