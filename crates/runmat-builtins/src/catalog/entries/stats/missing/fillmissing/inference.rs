use super::{FillMissingMethodKind, FillMissingOptionKind};
use crate::catalog::inference::literal_text;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, AliasFact, CallContract, CallInference, CallRequest, DynamicReason, FactJoin,
    InferenceDiagnostic, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

fn known_dimension(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Vector(values) if values.len() == 1 => known_dimension(&values[0]),
        LiteralValue::Integer { text, .. } => text.parse::<usize>().is_ok_and(|dim| dim > 0),
        _ => LiteralContext::numeric_from_literal(literal).is_some_and(|dim| {
            dim.is_finite() && dim >= 1.0 && dim.fract() == 0.0 && dim < usize::MAX as f64
        }),
    }
}

// Only a complete, source-known form can establish output facts. Runtime
// parsing remains authoritative for dynamic arguments and detailed errors.
fn admitted_method(
    request: &CallRequest,
    method: Option<FillMissingMethodKind>,
) -> Option<FillMissingMethodKind> {
    if request.arguments.len() != request.literals.literal_args.len() {
        return None;
    }
    let method_fact = request.arguments.get(1)?;
    let method_is_text = match &method_fact.kind {
        ValueKindFact::String => method_fact.shape.element_count() == Some(1),
        ValueKindFact::Character => method_fact
            .shape
            .known_dims()
            .is_some_and(|dims| dims.first() == Some(&Some(1))),
        _ => false,
    };
    if !method_is_text {
        return None;
    }
    let method = method?;
    let first_option = if method == FillMissingMethodKind::Constant {
        3
    } else {
        2
    };
    if request.arguments.len() < first_option {
        return None;
    }
    let options = &request.literals.literal_args[first_option..];
    let valid = match options {
        [] => true,
        [dimension] => known_dimension(dimension),
        [name, dimension] => {
            literal_text(name).and_then(|text| FillMissingOptionKind::parse(&text))
                == Some(FillMissingOptionKind::Dimension)
                && known_dimension(dimension)
        }
        _ => false,
    };
    valid.then_some(method)
}

fn supported_storage(input: &ValueFact) -> bool {
    match input.storage {
        StorageFact::Scalar | StorageFact::Dense => true,
        StorageFact::Opaque => matches!(input.residency, ResidencyFact::Device { .. }),
        StorageFact::Unknown | StorageFact::Sparse => false,
    }
}

fn scalar_numeric_fill(fill: &ValueFact) -> bool {
    fill.shape.element_count() == Some(1)
        && matches!(fill.residency, ResidencyFact::Host)
        && match &fill.kind {
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) => matches!(fill.storage, StorageFact::Scalar | StorageFact::Dense),
            ValueKindFact::Logical => fill.storage == StorageFact::Scalar,
            _ => false,
        }
}

fn scalar_text_fill(fill: &ValueFact) -> bool {
    fill.shape.element_count() == Some(1)
        && matches!(fill.residency, ResidencyFact::Host)
        && fill.kind == ValueKindFact::String
        && matches!(fill.storage, StorageFact::Scalar | StorageFact::Dense)
}

fn integer(class: NumericClass) -> bool {
    matches!(
        class,
        NumericClass::Int8
            | NumericClass::Int16
            | NumericClass::Int32
            | NumericClass::Int64
            | NumericClass::UInt8
            | NumericClass::UInt16
            | NumericClass::UInt32
            | NumericClass::UInt64
    )
}

fn admitted_input(request: &CallRequest, input: &ValueFact, method: FillMissingMethodKind) -> bool {
    if !supported_storage(input) {
        return false;
    }
    match &input.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
            if input.storage == StorageFact::Scalar
                && !integer(numeric.class)
                && method != FillMissingMethodKind::Constant
            {
                return false;
            }
            if method == FillMissingMethodKind::Constant
                && !integer(numeric.class)
                && input.storage != StorageFact::Scalar
            {
                request.arguments.get(2).is_some_and(scalar_numeric_fill)
            } else {
                true
            }
        }
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        }) => input.storage == StorageFact::Scalar && method == FillMissingMethodKind::Constant,
        ValueKindFact::Logical => true,
        ValueKindFact::String if input.storage == StorageFact::Scalar => {
            method == FillMissingMethodKind::Constant
        }
        ValueKindFact::String => match method {
            FillMissingMethodKind::Constant => {
                request.arguments.get(2).is_some_and(scalar_text_fill)
            }
            FillMissingMethodKind::Previous
            | FillMissingMethodKind::Next
            | FillMissingMethodKind::Nearest => true,
            _ => false,
        },
        _ => false,
    }
}

fn result_fact(
    request: &CallRequest,
    input: &ValueFact,
    method: FillMissingMethodKind,
) -> ValueFact {
    let mut output = input.clone();
    output.residency = ResidencyFact::Host;
    output.alias = AliasFact::Unique;
    if input.storage == StorageFact::Opaque {
        output.storage = StorageFact::Dense;
    }
    let scalar_can_be_missing = matches!(
        input.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real | NumericDomain::Complex,
        }) | ValueKindFact::String
    ) && input.storage == StorageFact::Scalar;
    if scalar_can_be_missing && method == FillMissingMethodKind::Constant {
        if let Some(fill) = request.arguments.get(2) {
            return output.join(fill);
        }
    }
    output
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let method_text = request.literals.literal_args.get(1).and_then(literal_text);
    let method = method_text
        .as_deref()
        .and_then(FillMissingMethodKind::parse);
    let admitted = request
        .arguments
        .first()
        .zip(admitted_method(request, method))
        .and_then(|(input, method)| {
            admitted_input(request, input, method).then_some((input, method))
        });
    let (result, mask) = admitted.map_or_else(
        || {
            (
                ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            )
        },
        |(input, method)| {
            (
                result_fact(request, input, method),
                ValueFact::proven(
                    ValueKindFact::Logical,
                    input.shape.clone(),
                    StorageFact::Dense,
                ),
            )
        },
    );
    let mut contract = CallContract::fixed(vec![result, mask]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    if request.arguments.len() < 2 {
        inferred.diagnostics.push(InferenceDiagnostic::error(
            "RM.MISSING.INVALID_ARGUMENT",
            "fillmissing requires data and a fill method",
        ));
    } else if let Some(method_text) = method_text {
        match method {
            None => inferred.diagnostics.push(InferenceDiagnostic::error(
                "RM.MISSING.INVALID_ARGUMENT",
                format!("fillmissing: unsupported method '{method_text}'"),
            )),
            Some(method)
                if request.arguments.first().is_some_and(|input| {
                    input.kind == ValueKindFact::String
                        && input.storage == StorageFact::Dense
                        && !matches!(
                            method,
                            FillMissingMethodKind::Constant
                                | FillMissingMethodKind::Previous
                                | FillMissingMethodKind::Next
                                | FillMissingMethodKind::Nearest
                        )
                }) =>
            {
                inferred.diagnostics.push(InferenceDiagnostic::error(
                    "RM.MISSING.UNSUPPORTED_TYPE",
                    "fillmissing: string arrays support constant, previous, next, and nearest",
                ));
            }
            _ => {}
        }
    }
    inferred
}
