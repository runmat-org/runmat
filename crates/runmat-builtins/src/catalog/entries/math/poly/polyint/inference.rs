use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CertaintyFact, DynamicReason,
    InferenceDiagnostic, NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYINT-ARITY",
            "polyint requires coefficients and at most one integration constant",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let coefficients = request.arguments.first();
    validate_coefficients(coefficients, &mut diagnostics);
    validate_constant(request.arguments.get(1), &mut diagnostics);
    let output = output_fact(coefficients, request.arguments.get(1));
    let count = request.outputs.requested.known_count();
    if count.is_some_and(|count| count > 1) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYINT-OUTPUTS",
            "polyint returns one coefficient vector",
            0,
        ));
    }
    let mut contract = CallContract::fixed(if count == Some(0) {
        Vec::new()
    } else {
        vec![output]
    });
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = count.is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn validate_coefficients(value: Option<&ValueFact>, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(value) = value else { return };
    validate_floating(value, 0, diagnostics);
    if let Some(dims) = value.shape.known_dims() {
        if dims.iter().filter(|dim| dim.is_some_and(|n| n > 1)).count() > 1 {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLYINT-VECTOR",
                "polynomial coefficients must form a vector",
                0,
            ));
        }
    }
}

fn validate_constant(value: Option<&ValueFact>, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(value) = value else { return };
    validate_floating(value, 1, diagnostics);
    if value.shape.element_count().is_some_and(|count| count != 1) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYINT-CONSTANT",
            "integration constant must be scalar",
            1,
        ));
    }
}

fn validate_floating(value: &ValueFact, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single | NumericClass::Double,
            ..
        }) | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLYINT-CLASS",
            "polyint accepts single- or double-precision numeric input, not integer or logical values",
            index,
        ));
    }
}

fn output_fact(coefficients: Option<&ValueFact>, constant: Option<&ValueFact>) -> ValueFact {
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    if let Some(count) = coefficients.and_then(|value| value.shape.element_count()) {
        if let Some(length) = count.checked_add(1) {
            output.shape = ShapeFact::from(vec![Some(1), Some(length)]);
            if length > 1 {
                output.storage = StorageFact::Dense;
            }
        }
    }
    let Some(coefficients) = coefficients else {
        return output;
    };
    let ValueKindFact::Numeric(numeric) = coefficients.kind else {
        return output;
    };
    if !matches!(numeric.class, NumericClass::Single | NumericClass::Double) {
        return output;
    }
    let domain = match constant.map(|value| &value.kind) {
        None
        | Some(ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })) => numeric.domain,
        Some(ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        })) => NumericDomain::Complex,
        _ => return output,
    };
    let class = if domain == NumericDomain::Complex {
        NumericClass::Double
    } else {
        numeric.class
    };
    output.kind = ValueKindFact::Numeric(NumericFact { class, domain });
    // A provider hook may retain residency; unsupported work may gather and
    // restore. The source fact does not prove which route will be selected.
    output.residency = ResidencyFact::Unknown;
    output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    output
}
