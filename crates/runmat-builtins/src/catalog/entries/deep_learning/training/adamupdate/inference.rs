use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

fn numeric_operand_possible(fact: &ValueFact) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown | ValueKindFact::Object(_) => true,
        ValueKindFact::Numeric(number) if fact.storage == StorageFact::Scalar => {
            number.domain == NumericDomain::Real
        }
        ValueKindFact::Numeric(number) => {
            number.domain == NumericDomain::Real
                && matches!(number.class, NumericClass::Double | NumericClass::Single)
        }
        _ => false,
    }
}

fn known_shape_conflict(left: &ValueFact, right: &ValueFact) -> bool {
    match (left.shape.known_dims(), right.shape.known_dims()) {
        (Some(left), Some(right)) if left.len() == right.len() => left
            .iter()
            .zip(right.iter())
            .any(|(a, b)| a.zip(*b).is_some_and(|(a, b)| a != b)),
        (Some(left), Some(right)) => {
            left.iter().all(Option::is_some) && right.iter().all(Option::is_some)
        }
        _ => false,
    }
}

fn state_output(state: Option<&ValueFact>, params: &ValueFact) -> ValueFact {
    match state {
        Some(value) if value.shape.element_count() == Some(0) => params.clone(),
        Some(value) => value.clone(),
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}

fn scalar_integer_output(mut value: ValueFact) -> ValueFact {
    if let ValueKindFact::Numeric(number) = &value.kind {
        if value.storage == StorageFact::Scalar
            && !matches!(number.class, NumericClass::Double | NumericClass::Single)
        {
            value.kind = ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            });
        }
    }
    value
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
    if !(5..=9).contains(&args.len()) {
        diagnostics.push(argument_error(
            "RM-DL-ADAM-ARITY",
            "adamupdate requires five to nine positional arguments",
            args.len().saturating_sub(1),
        ));
    }
    for index in 0..args.len().min(4) {
        let arg = &args[index];
        if !numeric_operand_possible(arg) {
            diagnostics.push(argument_error(
                "RM-DL-ADAM-CLASS",
                "optimizer arrays must be real floating-point arrays or supported dlarray wrappers",
                index,
            ));
        }
        if index == 0 && arg.shape.element_count() == Some(0) {
            diagnostics.push(argument_error(
                "RM-DL-ADAM-EMPTY",
                "parameters must be nonempty",
                index,
            ));
        }
        if index > 0
            && index < 4
            && arg.shape.element_count() != Some(0)
            && args
                .first()
                .is_some_and(|params| known_shape_conflict(params, arg))
        {
            diagnostics.push(argument_error(
                "RM-DL-ADAM-SHAPE",
                "gradient and nonempty optimizer state must match parameter shape",
                index,
            ));
        }
    }
    if let Some(iteration) = args.get(4) {
        if !matches!(
            iteration.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Unknown
        ) || iteration
            .shape
            .element_count()
            .is_some_and(|count| count != 1)
            || request
                .literals
                .numeric_at(4)
                .is_some_and(|value| !value.is_finite() || value <= 0.0 || value.fract() != 0.0)
        {
            diagnostics.push(argument_error(
                "RM-DL-ADAM-ITERATION",
                "iteration must be a positive integer scalar",
                4,
            ));
        }
    }
    for (index, arg) in args.iter().enumerate().skip(5) {
        if !matches!(arg.kind, ValueKindFact::Numeric(_) | ValueKindFact::Unknown)
            || arg.shape.element_count().is_some_and(|count| count != 1)
        {
            diagnostics.push(argument_error(
                "RM-DL-ADAM-CONTROL",
                "optimizer controls must be numeric scalars",
                index,
            ));
        }
        if let Some(value) = request.literals.numeric_at(index) {
            let valid = value.is_finite()
                && if index == 6 || index == 7 {
                    (0.0..1.0).contains(&value)
                } else {
                    value > 0.0
                };
            if !valid {
                diagnostics.push(argument_error(
                    "RM-DL-ADAM-CONTROL",
                    "optimizer control is outside its supported range",
                    index,
                ));
            }
        }
    }
    let params = args
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let host_only = args
        .iter()
        .all(|value| matches!(value.residency, ResidencyFact::Host));
    let mut first = scalar_integer_output(state_output(args.get(2), &params));
    let mut second = scalar_integer_output(state_output(args.get(3), &params));
    let mut updated = scalar_integer_output(params);
    if !host_only {
        // Provider execution is conditional on owner/storage support; a host
        // gather fallback can change placement. Do not prove device output.
        first.residency = ResidencyFact::Unknown;
        second.residency = ResidencyFact::Unknown;
        updated.residency = ResidencyFact::Unknown;
    }
    finish_fixed_outputs(entry, request, vec![updated, first, second], diagnostics)
}
