use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = vec![];
    let count = request.arguments.len();
    if count > 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-VIEW-ARITY",
            "view accepts at most an axes target and two angle scalars",
            3,
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        let nonfinite = request
            .literals
            .numeric_at(index)
            .is_some_and(|n| !(n as f32).is_finite())
            || request
                .literals
                .literal_vector_at(index)
                .is_some_and(|values| {
                    values
                        .iter()
                        .filter_map(LiteralContext::numeric_from_literal)
                        .any(|n| !(n as f32).is_finite())
                });
        if nonfinite {
            diagnostics.push(argument_error(
                "RM-CATALOG-VIEW-FINITE",
                "view angles must remain finite in single-precision camera state",
                index,
            ));
        }
        if !matches!(
            argument.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) || argument
            .numeric()
            .is_some_and(|n| n.domain == NumericDomain::Complex)
            || argument.storage == StorageFact::Sparse
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-VIEW-NUMERIC",
                "view accepts real host numeric values",
                index,
            ));
        }
        // With two inputs the first may be an axes identity or an angle;
        // only a scalar can serve either role. A single input admits one or
        // two elements regardless of rank, matching the existing domain parser.
        let maximum = if count == 1 || (count == 2 && index == 1) {
            2
        } else {
            1
        };
        if argument
            .shape
            .element_count()
            .is_some_and(|n| n == 0 || n > maximum)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-VIEW-SHAPE",
                "view requires a scalar target/preset/angle or two-element angle array",
                index,
            ));
        }
    }
    let numeric = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    let scalar = ValueFact::scalar(numeric.clone());
    let first = match request.outputs.requested.known_count() {
        Some(1) => ValueFact::proven(
            numeric,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        ),
        Some(_) => scalar.clone(),
        None => {
            let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
            output.kind = numeric;
            output.residency = ResidencyFact::Host;
            output
        }
    };
    let mut contract = CallContract::fixed(vec![first, scalar]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
