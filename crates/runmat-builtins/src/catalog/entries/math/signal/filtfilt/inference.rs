use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FILTFILT-ARITY",
            "filtfilt expects numerator coefficients, denominator coefficients, and a vector signal",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for (index, value) in request.arguments.iter().enumerate() {
        if matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
            || !matches!(
                value.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            )
            || matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Complex,
                    class,
                }) if class.integer_class().is_some()
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FILTFILT-OPERAND",
                "filtfilt requires dense real or complex numeric vectors",
                index,
            ));
        }
        if value.shape.element_count() == Some(0)
            || matches!(&value.shape, ShapeFact::Shaped { dims } if dims.iter().filter(|dim| matches!(dim, DimensionFact::Known(extent) if *extent > 1)).count() > 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FILTFILT-VECTOR",
                "filtfilt coefficients and signal must be nonempty vectors",
                index,
            ));
        }
    }

    let output = output_fact(request, &diagnostics);
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn output_fact(request: &CallRequest, diagnostics: &[InferenceDiagnostic]) -> ValueFact {
    if !diagnostics.is_empty() {
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let Some(signal) = request.arguments.get(2) else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let mut any_single = false;
    let mut any_complex = false;
    for operand in &request.arguments {
        match operand.kind {
            ValueKindFact::Numeric(NumericFact { class, domain }) => {
                any_single |= class == NumericClass::Single;
                any_complex |= domain == NumericDomain::Complex;
            }
            ValueKindFact::Logical => {}
            ValueKindFact::Unknown => return ValueFact::unknown(DynamicReason::RuntimeValue),
            _ => return ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
        }
    }
    let class = if any_single {
        NumericClass::Single
    } else {
        NumericClass::Double
    };
    let domain = if any_complex {
        NumericDomain::Complex
    } else {
        NumericDomain::Real
    };
    let storage = if class == NumericClass::Double
        && domain == NumericDomain::Real
        && signal.shape.element_count() == Some(1)
    {
        StorageFact::Scalar
    } else if signal.shape.element_count().is_some() {
        StorageFact::Dense
    } else {
        StorageFact::Unknown
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        signal.shape.clone(),
        storage,
    );
    if request
        .arguments
        .iter()
        .any(|operand| operand.residency != ResidencyFact::Host)
    {
        output.residency = ResidencyFact::Unknown;
    }
    output
}
