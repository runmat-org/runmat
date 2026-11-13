use super::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use runmat_types::{
    DynamicReason, NumericClass, NumericDomain, NumericFact, ObjectFact, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn object(value: &ValueFact, class: runmat_types::StaticClassIdentity) -> Option<bool> {
    match &value.kind {
        ValueKindFact::Unknown => None,
        ValueKindFact::Object(value) => value
            .runtime_class
            .as_ref()
            .map(|identity| identity.is(class)),
        _ => Some(false),
    }
}
fn double(shape: ShapeFact) -> ValueFact {
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    let mut out = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    out.residency = ResidencyFact::Host;
    out
}
fn class(shape: ShapeFact, runtime_class: runmat_types::StaticClassIdentity) -> ValueFact {
    let mut out = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runtime_class.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        shape,
        StorageFact::Opaque,
    );
    out.residency = ResidencyFact::Host;
    out
}
pub(super) fn predicate(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-ISDATETIME-ARITY",
            "isdatetime expects one input",
            0,
        ));
    }
    let mut out = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    out.residency = ResidencyFact::Host;
    finish_fixed(entry, request, out, diagnostics)
}
pub(super) fn tick(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(entry, request, double(ShapeFact::Scalar), vec![])
}
fn binary(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    plus: bool,
    comparison: bool,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CALENDAR-METHOD-ARITY",
            "calendarDuration methods expect two inputs",
            0,
        ));
    }
    if let Some(lhs) = request.arguments.first() {
        if object(lhs, runmat_types::standard::CALENDAR_DURATION) == Some(false) {
            diagnostics.push(argument_error(
                "RM-CALENDAR-METHOD-LEFT",
                "expected calendarDuration receiver",
                0,
            ));
        }
    }
    let rhs = request.arguments.get(1);
    if let Some(rhs) = rhs {
        if object(rhs, runmat_types::standard::CALENDAR_DURATION) == Some(false)
            && (!plus || object(rhs, runmat_types::standard::DATETIME) == Some(false))
        {
            diagnostics.push(argument_error(
                "RM-CALENDAR-METHOD-RIGHT",
                "expected calendarDuration, or datetime for plus",
                1,
            ));
        }
    }
    let shape = match request.arguments.as_slice() {
        [lhs, rhs]
            if lhs.shape.element_count() == Some(1) && rhs.shape.element_count() == Some(1) =>
        {
            ShapeFact::Scalar
        }
        [lhs, rhs] if lhs.shape.element_count() == Some(1) => rhs.shape.clone(),
        [lhs, rhs] if rhs.shape.element_count() == Some(1) || lhs.shape == rhs.shape => {
            lhs.shape.clone()
        }
        [lhs, rhs] => {
            if lhs.shape.element_count().is_some() && rhs.shape.element_count().is_some() {
                diagnostics.push(argument_error(
                    "RM-CALENDAR-METHOD-SHAPE",
                    "operands must have equal shapes or one element",
                    1,
                ));
            }
            ShapeFact::Unknown
        }
        _ => ShapeFact::Unknown,
    };
    let out = if comparison {
        double(shape)
    } else if plus && rhs.is_some_and(|v| object(v, runmat_types::standard::DATETIME) == Some(true))
    {
        class(shape, runmat_types::standard::DATETIME)
    } else if plus
        && rhs.is_none_or(|v| object(v, runmat_types::standard::CALENDAR_DURATION) != Some(true))
    {
        let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
        value.shape = shape;
        value.residency = ResidencyFact::Host;
        value
    } else {
        class(shape, runmat_types::standard::CALENDAR_DURATION)
    };
    finish_fixed(entry, request, out, diagnostics)
}
pub(super) fn plus(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    binary(request, entry, true, false)
}
pub(super) fn minus(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    binary(request, entry, false, false)
}
pub(super) fn comparison(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    binary(request, entry, false, true)
}
