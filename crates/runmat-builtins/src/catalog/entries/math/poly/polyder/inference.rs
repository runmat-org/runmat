use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYDER-ARITY",
            "polyder accepts one polynomial or two polynomial operands",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for (index, input) in request.arguments.iter().enumerate() {
        validate_coefficient(input, index, &mut diagnostics);
    }
    let count = request.outputs.requested.known_count();
    if count.is_some_and(|count| count >= 2) && request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYDER-QUOTIENT",
            "the two-output quotient form requires two polynomial operands",
            0,
        ));
    }
    let first = request.arguments.first();
    let second = request.arguments.get(1);
    let output = coefficient_output(first, second);
    let outputs = match count {
        Some(0) => Vec::new(),
        Some(1) => vec![output],
        Some(count) => {
            let denominator = coefficient_output(second, second);
            (0..count)
                .map(|index| match index {
                    0 => output.clone(),
                    1 => denominator.clone(),
                    _ => ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    })),
                })
                .collect()
        }
        None => vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = count.is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn validate_coefficient(
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYDER-CLASS",
            "polyder coefficients must be numeric or logical",
            index,
        ));
    }
    if let Some(dims) = value.shape.known_dims() {
        if dims.iter().filter(|dim| dim.is_some_and(|n| n > 1)).count() > 1 {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLYDER-VECTOR",
                "polynomial coefficients must form a vector",
                index,
            ));
        }
    }
}

fn coefficient_output(first: Option<&ValueFact>, second: Option<&ValueFact>) -> ValueFact {
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = output_shape(first);
    output.storage = StorageFact::Unknown;
    output.residency = ResidencyFact::Unknown;
    let inputs = [first, second];
    if inputs.iter().flatten().all(|input| {
        matches!(
            input.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
        ) && matches!(input.residency, ResidencyFact::Host)
    }) && first.is_some()
    {
        output.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        output.residency = ResidencyFact::Host;
    }
    output
}

fn output_shape(input: Option<&ValueFact>) -> ShapeFact {
    let Some(input) = input else {
        return ShapeFact::Unknown;
    };
    if input.shape.element_count().is_some_and(|count| count <= 1) {
        return ShapeFact::Scalar;
    }
    let Some(dims) = input.shape.known_dims() else {
        return ShapeFact::Unknown;
    };
    let orientation = dims.iter().position(|dim| dim.is_some_and(|n| n != 1));
    match orientation {
        Some(0) | Some(2..) => ShapeFact::from(vec![None, Some(1)]),
        Some(1) => ShapeFact::from(vec![Some(1), None]),
        None => ShapeFact::Unknown,
    }
}
