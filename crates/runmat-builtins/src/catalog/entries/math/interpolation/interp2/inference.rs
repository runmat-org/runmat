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
            if Some(index) == grid
                && effective_method(request, index) != EffectiveMethod::LinearOrNearest
                && matches!(value.kind, ValueKindFact::Numeric(_))
            {
                continue;
            }
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
    let output = match (grid, xq, yq) {
        (Some(grid), Some(left), Some(right)) => output_fact(request, grid, left, right),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EffectiveMethod {
    LinearOrNearest,
    Spline,
    Unknown,
}

/// The runtime processes every option in order and the last method wins.
/// Unknown string-like options can change it; known numeric extrapolation
/// controls and the `extrap` keyword cannot.
fn effective_method(request: &CallRequest, grid: usize) -> EffectiveMethod {
    let first_option = if grid == 0 { 3 } else { 5 };
    let mut method = EffectiveMethod::LinearOrNearest;
    for index in first_option..request.arguments.len() {
        let literal = request.literals.literal_args.get(index);
        let spelling = match literal {
            Some(
                LiteralValue::Character(value)
                | LiteralValue::String(value)
                | LiteralValue::Keyword(value),
            ) => Some(value.as_str()),
            Some(
                LiteralValue::Number(_)
                | LiteralValue::Real { .. }
                | LiteralValue::Integer { .. }
                | LiteralValue::Bool(_),
            ) => continue,
            _ if matches!(
                request.arguments[index].kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) =>
            {
                continue
            }
            _ => return EffectiveMethod::Unknown,
        };
        match spelling.map(str::to_ascii_lowercase).as_deref() {
            Some("linear" | "nearest") => method = EffectiveMethod::LinearOrNearest,
            Some("spline") => method = EffectiveMethod::Spline,
            Some("extrap") => {}
            _ => return EffectiveMethod::Unknown,
        }
    }
    method
}

fn output_fact(request: &CallRequest, grid: usize, xq: usize, yq: usize) -> ValueFact {
    let (Some(samples), Some(xq), Some(yq)) = (
        request.arguments.get(grid),
        request.arguments.get(xq),
        request.arguments.get(yq),
    ) else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let shape = match (xq.shape.element_count(), yq.shape.element_count()) {
        (Some(1), Some(1)) => ShapeFact::Scalar,
        (Some(1), _) => yq.shape.clone(),
        (_, Some(1)) => xq.shape.clone(),
        (Some(_), Some(_)) if xq.shape.is_proven_equivalent(&yq.shape) => xq.shape.clone(),
        _ => ShapeFact::Unknown,
    };
    let kind = if effective_method(request, grid) == EffectiveMethod::Spline {
        match &samples.kind {
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single | NumericClass::Double,
                ..
            }) => samples.kind.clone(),
            _ => ValueKindFact::Unknown,
        }
    } else if effective_method(request, grid) == EffectiveMethod::LinearOrNearest {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    } else {
        ValueKindFact::Unknown
    };
    let mut output = if kind == ValueKindFact::Unknown {
        let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
        value.shape = shape;
        value
    } else {
        ValueFact::proven(kind, shape, StorageFact::Unknown)
    };
    if output.shape == ShapeFact::Unknown {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output
}
