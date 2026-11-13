use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROOTS-ARITY",
            "roots requires one coefficient vector",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let coefficient = request.arguments.first();
    if let Some(value) = coefficient {
        validate_input(value, &mut diagnostics);
    }
    let count = request.outputs.requested.known_count();
    if count.is_some_and(|count| count > 1) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROOTS-OUTPUTS",
            "roots returns one column vector",
            0,
        ));
    }
    let mut contract = CallContract::fixed(if count == Some(0) {
        Vec::new()
    } else {
        vec![output_fact(coefficient)]
    });
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = count.is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn validate_input(value: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROOTS-CLASS",
            "roots requires numeric or logical coefficients",
            0,
        ));
    }
    if let Some(dims) = value.shape.known_dims() {
        let nonsingleton = dims
            .iter()
            .filter(|dim| dim.is_some_and(|dim| dim > 1))
            .count();
        if nonsingleton > 1 {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROOTS-VECTOR",
                "coefficient array must have at most one nonsingleton dimension",
                0,
            ));
        }
    }
}

fn output_fact(input: Option<&ValueFact>) -> ValueFact {
    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    result.shape = ShapeFact::from(vec![None, Some(1)]);
    result.storage = StorageFact::Dense;
    result.residency = ResidencyFact::Host;
    if input
        .and_then(|fact| fact.shape.element_count())
        .is_some_and(|n| n <= 1)
    {
        result.kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        result.shape = ShapeFact::from(vec![Some(0), Some(1)]);
    }
    result
}
