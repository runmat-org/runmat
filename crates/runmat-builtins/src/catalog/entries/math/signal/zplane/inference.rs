use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if !(1..=4).contains(&count) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ZPLANE-ARITY",
            "zplane requires SOS, coefficient, or zero/pole arguments",
            count,
        ));
    }
    let offset = usize::from(count == 4);
    if count == 4
        && request
            .arguments
            .first()
            .is_some_and(|first| first.shape.is_definitely_nonscalar())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-ZPLANE-AXES",
            "four-argument zplane requires a scalar axes handle first",
            0,
        ));
    }
    if count == 4
        && request.arguments.first().is_some_and(|first| {
            matches!(&first.kind, ValueKindFact::Numeric(fact)
        if fact.domain == NumericDomain::Complex && fact.class.integer_class().is_some())
        })
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-ZPLANE-AXES",
            "axes handle cannot be a complex integer value",
            0,
        ));
    }
    for (index, value) in request.arguments.iter().enumerate().skip(offset) {
        if matches!(&value.kind, ValueKindFact::Numeric(fact)
            if fact.domain == NumericDomain::Complex && fact.class.integer_class().is_some())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ZPLANE-COMPLEX-INTEGER",
                "zplane cannot evaluate complex integer data through a floating root kernel",
                index,
            ));
        }
        if !matches!(
            value.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
        ) || matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ZPLANE-INPUT",
                "filter data must be a dense numeric value",
                index,
            ));
        }
        if request.literals.integer_is_exact_binary64_at(index) == Some(false) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ZPLANE-INTEGER-EXACTNESS",
                "integer filter data must convert exactly to double",
                index,
            ));
        }
    }
    if count == 1 {
        if request
            .arguments
            .first()
            .is_some_and(|value| matches!(value.kind, ValueKindFact::Logical))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ZPLANE-SOS-CLASS",
                "SOS must be a numeric matrix rather than a logical array",
                0,
            ));
        }
        if request
            .arguments
            .first()
            .is_some_and(|value| match &value.shape {
                ShapeFact::Scalar => true,
                ShapeFact::Shaped { dims } => {
                    dims.is_empty()
                        || dims.len() > 2
                        || match dims.as_slice() {
                            [DimensionFact::Known(columns)] => *columns != 6,
                            [_, DimensionFact::Known(columns)] => *columns != 6,
                            _ => false,
                        }
                }
                _ => false,
            })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ZPLANE-SOS-SHAPE",
                "SOS must be a two-dimensional matrix with six columns",
                0,
            ));
        }
    }
    // Three arguments are ambiguous: (z,p,k) or (ax,b,a). A scalar first
    // argument may be an encoded axes handle, so only prove gain invalidity
    // when the first argument cannot be an axes handle.
    let proven_zpk = count == 3
        && request
            .arguments
            .first()
            .is_some_and(|first| first.shape.is_definitely_nonscalar());
    if proven_zpk || count == 4 {
        let gain_index = count - 1;
        if request.arguments[gain_index]
            .shape
            .is_definitely_nonscalar()
            || request
                .literals
                .numeric_at(gain_index)
                .is_some_and(|n| !n.is_finite())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ZPLANE-GAIN",
                "gain must be a finite numeric scalar",
                gain_index,
            ));
        }
    }
    let output = if diagnostics.is_empty() {
        let mut value = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        );
        value.residency = ResidencyFact::Host;
        value
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}
