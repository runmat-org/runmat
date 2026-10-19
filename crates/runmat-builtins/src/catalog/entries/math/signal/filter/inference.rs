use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(3..=5).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FILTER-ARITY",
            "filter expects coefficients, a signal, and at most initial state and dimension",
            request.arguments.len().saturating_sub(1),
        ));
    }
    for index in 0..request.arguments.len().min(4) {
        validate_numeric_operand(request, index, &mut diagnostics);
    }
    if let Some(dimension) = request.arguments.get(4) {
        if dimension.shape.is_definitely_nonscalar()
            || !matches!(
                dimension.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            )
            || request
                .literals
                .numeric_at(4)
                .is_some_and(|value| !value.is_finite() || value < 1.0 || value.fract() != 0.0)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FILTER-DIMENSION",
                "filter dimension must be a positive integer scalar",
                4,
            ));
        }
    }

    let output = output_fact(request, &diagnostics);
    let mut contract =
        CallContract::fixed(vec![output.clone(), final_state_fact(request, &output)]);
    // The runtime may pad surplus language outputs, but they are not numerical
    // results of the filter and do not have a proven class or shape here.
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::unknown(
        DynamicReason::UnsupportedRepresentation,
    )));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn validate_numeric_operand(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(value) = request.arguments.get(index) else {
        return;
    };
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
            "RM-CATALOG-FILTER-OPERAND",
            "filter requires dense real or complex numeric operands",
            index,
        ));
    }
    if index < 2
        && (value.shape.element_count() == Some(0)
            || matches!(&value.shape, ShapeFact::Shaped { dims } if dims.iter().filter(|dim| matches!(dim, DimensionFact::Known(extent) if *extent > 1)).count() > 1))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FILTER-COEFFICIENTS",
            "filter coefficients must be nonempty vectors",
            index,
        ));
    }
}

fn output_fact(request: &CallRequest, diagnostics: &[InferenceDiagnostic]) -> ValueFact {
    let Some(signal) = request.arguments.get(2) else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let class = match signal.kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        }) => NumericClass::Single,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => NumericClass::Double,
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut complex = false;
    for operand in request.arguments.iter().take(4) {
        match operand.kind {
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Complex,
                ..
            }) => complex = true,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical => {}
            ValueKindFact::Unknown => return ValueFact::unknown(DynamicReason::RuntimeValue),
            _ => return ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
        }
    }
    if !diagnostics.is_empty() {
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let domain = if complex {
        NumericDomain::Complex
    } else {
        NumericDomain::Real
    };
    let storage = if class == NumericClass::Double && signal.shape.element_count() == Some(1) {
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

fn final_state_fact(request: &CallRequest, output: &ValueFact) -> ValueFact {
    let Some(numeric) = output.numeric() else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let mut state = ValueFact::proven(
        ValueKindFact::Numeric(numeric),
        ShapeFact::Unknown,
        StorageFact::Unknown,
    );
    state.residency = ResidencyFact::Unknown;
    if request
        .arguments
        .iter()
        .all(|operand| operand.residency == ResidencyFact::Host)
    {
        state.residency = ResidencyFact::Host;
    }
    state
}
