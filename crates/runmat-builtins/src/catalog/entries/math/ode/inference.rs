use super::options::OdeOptionField;
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, EffectKind,
    InferenceDiagnostic, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

/// The current host solvers return double trajectories. The first output is
/// `y` for a single requested result and `t` for multiple requested results.
/// That differs from MATLAB's solution-structure single-output form; the
/// declaration remains faithful to the implemented interface until that form
/// is implemented and qualified.
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(3..=4).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ODE-ARITY",
            "ODE solvers require odefun, tspan, y0 and at most one options struct",
            request.arguments.len().saturating_sub(1),
        ));
    }
    validate_callback(request.arguments.first(), &mut diagnostics);
    let times = validate_time_span(request.arguments.get(1), &mut diagnostics);
    let states = validate_initial_state(request.arguments.get(2), &mut diagnostics);
    validate_options(request.arguments.get(3), &mut diagnostics);

    let returned_rows = times.filter(|count| *count > 2);
    let t = matrix_fact(returned_rows, Some(1));
    let y = matrix_fact(returned_rows, states);
    let requested = request.outputs.requested.known_count();
    let outputs = match requested {
        Some(0) => Vec::new(),
        Some(1) => vec![y],
        Some(count) => (0..count)
            .map(|index| match index {
                0 => t.clone(),
                1 => y.clone(),
                _ => ValueFact::scalar(double_real()),
            })
            .collect(),
        None => vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if requested.is_none() {
        contract.maximum_outputs = None;
    }
    let mut callback = infer_callback(request);
    contract.effects.0.append(&mut callback.effects.0);
    contract.capabilities.0.append(&mut callback.capabilities.0);
    diagnostics.append(&mut callback.diagnostics);
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = requested.is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn double_real() -> ValueKindFact {
    ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    })
}

fn matrix_fact(rows: Option<usize>, cols: Option<usize>) -> ValueFact {
    ValueFact::proven(
        double_real(),
        ShapeFact::from(vec![rows, cols]),
        StorageFact::Dense,
    )
}

fn validate_callback(value: Option<&ValueFact>, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(value) = value else {
        return;
    };
    if !matches!(
        value.kind,
        ValueKindFact::Callable(_)
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ODE-CALLBACK",
            "odefun must be a callable or a function name",
            0,
        ));
    }
}

fn validate_time_span(
    value: Option<&ValueFact>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    let value = value?;
    if !real_numeric_or_unknown(value) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ODE-TSPAN-CLASS",
            "tspan must contain real numeric time points",
            1,
        ));
    }
    let count = value.shape.element_count();
    if count.is_some_and(|count| count < 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ODE-TSPAN-COUNT",
            "tspan must contain at least two time points",
            1,
        ));
    }
    count
}

fn validate_initial_state(
    value: Option<&ValueFact>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    let value = value?;
    if !real_numeric_or_unknown(value) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ODE-Y0-CLASS",
            "y0 must contain real numeric initial states",
            2,
        ));
    }
    let count = value.shape.element_count();
    if count == Some(0) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ODE-Y0-EMPTY",
            "y0 cannot be empty",
            2,
        ));
    }
    count
}

fn real_numeric_or_unknown(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    )
}

fn validate_options(value: Option<&ValueFact>, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(value) = value else {
        return;
    };
    let ValueKindFact::Struct(structure) = &value.kind else {
        if !matches!(value.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ODE-OPTIONS",
                "options must be a struct",
                3,
            ));
        }
        return;
    };
    for (name, field) in &structure.fields {
        let Some(option) = OdeOptionField::from_public_name(name) else {
            continue;
        };
        if matches!(
            field.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
        ) && field.shape.element_count() == Some(0)
            && matches!(field.storage, StorageFact::Dense | StorageFact::Unknown)
        {
            continue;
        }
        let valid = if option.is_floating_control() {
            matches!(
                field.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Unknown
            ) && field.is_scalar()
                && matches!(
                    field.storage,
                    StorageFact::Scalar | StorageFact::Dense | StorageFact::Unknown
                )
        } else {
            matches!(
                field.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Unknown
            ) && field.is_scalar()
        };
        if !valid && !matches!(field.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ODE-OPTION-FIELD",
                format!("{} must be a numeric scalar", option.public_name()),
                3,
            ));
        }
    }
}

fn infer_callback(request: &CallRequest) -> CallInference {
    let Some(ValueFact {
        kind: ValueKindFact::Callable(callable),
        ..
    }) = request.arguments.first()
    else {
        return CallInference {
            outputs: Vec::new(),
            dynamic_outputs: false,
            discarded: Vec::new(),
            effects: runmat_types::EffectSet([EffectKind::Unknown].into()),
            capabilities: Default::default(),
            diagnostics: Vec::new(),
        };
    };
    let state = request
        .arguments
        .get(2)
        .map(|initial| ValueFact::proven(double_real(), initial.shape.clone(), initial.storage))
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let mut callback = crate::infer_callable_call(
        callable,
        &CallRequest {
            arguments: vec![ValueFact::scalar(double_real()), state],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    if let Some(derivative) = callback.outputs.first() {
        if !real_numeric_or_unknown(derivative) {
            callback.diagnostics.push(argument_error(
                "RM-CATALOG-ODE-DERIVATIVE-CLASS",
                "odefun must return a real numeric derivative",
                0,
            ));
        }
        if let (Some(expected), Some(actual)) = (
            request
                .arguments
                .get(2)
                .and_then(|state| state.shape.element_count()),
            derivative.shape.element_count(),
        ) {
            if expected != actual {
                callback.diagnostics.push(argument_error(
                    "RM-CATALOG-ODE-DERIVATIVE-SHAPE",
                    "odefun derivative length must match the initial state",
                    0,
                ));
            }
        }
    }
    callback
}
