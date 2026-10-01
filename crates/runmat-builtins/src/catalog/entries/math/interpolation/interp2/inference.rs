use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CertaintyFact, DynamicReason,
    LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GridForm {
    Implicit,
    Explicit,
    Dynamic,
}

fn form(request: &CallRequest) -> GridForm {
    if request.arguments.len() < 5 {
        return GridForm::Implicit;
    }
    if matches!(
        request.literals.literal_args.get(3),
        Some(LiteralValue::Character(_) | LiteralValue::String(_) | LiteralValue::Keyword(_))
    ) {
        return GridForm::Implicit;
    }
    match request.arguments[3].kind {
        ValueKindFact::Character | ValueKindFact::String => GridForm::Implicit,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => GridForm::Explicit,
        _ => GridForm::Dynamic,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(3..=7).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-INTERP2-ARITY",
            "interp2 expects a grid and two query coordinates, with optional axes and options",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let (grid, xq, yq) = match form(request) {
        GridForm::Implicit => (Some(0), Some(1), Some(2)),
        GridForm::Explicit => (Some(2), Some(3), Some(4)),
        GridForm::Dynamic => (None, None, None),
    };
    for index in [grid, xq, yq].into_iter().flatten() {
        if let Some(value) = request.arguments.get(index) {
            if !matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INTERP2-NUMERIC",
                    "interp2 grids and queries must be real numeric or logical",
                    index,
                ));
            }
        }
    }
    let output = match (xq, yq) {
        (Some(left), Some(right)) => {
            output_fact(request.arguments.get(left), request.arguments.get(right))
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}

fn output_fact(xq: Option<&ValueFact>, yq: Option<&ValueFact>) -> ValueFact {
    let (Some(xq), Some(yq)) = (xq, yq) else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let shape = match (xq.shape.element_count(), yq.shape.element_count()) {
        (Some(1), Some(1)) => ShapeFact::Scalar,
        (Some(1), _) => yq.shape.clone(),
        (_, Some(1)) => xq.shape.clone(),
        (Some(_), Some(_)) if xq.shape.is_proven_equivalent(&yq.shape) => xq.shape.clone(),
        _ => ShapeFact::Unknown,
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Unknown,
    );
    if output.shape == ShapeFact::Unknown {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output
}
