//! Domain facts for opaque, value-semantic decomposition objects.
mod constructor;

use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

fn object() -> ValueFact {
    let mut properties = BTreeMap::new();
    for name in ["Type", "Datatype"] {
        properties.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    for name in [
        "CheckCondition",
        "IsConjugateTransposed",
        "IsReal",
        "IsSparse",
    ] {
        properties.insert(name.into(), ValueFact::scalar(ValueKindFact::Logical));
    }
    properties.insert(
        "MatrixSize".into(),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        ),
    );
    // ScaleFactor can become real or complex; its value is not a class fact.
    let mut scale = ValueFact::unknown(DynamicReason::RuntimeValue);
    scale.shape = ShapeFact::Scalar;
    scale.residency = ResidencyFact::Host;
    properties.insert("ScaleFactor".into(), scale);
    let mut result = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::DECOMPOSITION.owned()),
            properties,
            // RankTolerance and Triangle are conditional properties.
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    result.residency = ResidencyFact::Host;
    result
}

fn is_object(value: &ValueFact) -> bool {
    matches!(&value.kind, ValueKindFact::Object(fact)
        if fact.runtime_class.as_ref() == Some(&runmat_types::standard::DECOMPOSITION.owned()))
}

fn arity(request: &CallRequest, count: usize) -> Vec<InferenceDiagnostic> {
    if request.arguments.len() == count {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-DECOMPOSITION-ARITY",
            format!("expected {count} inputs"),
            0,
        )]
    }
}

fn receiver(request: &CallRequest, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if let Some(value) = request.arguments.get(index) {
        if !is_object(value) && !matches!(value.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-DECOMPOSITION-RECEIVER",
                "expected a decomposition object",
                index,
            ));
        }
    }
}

pub(super) fn constructor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    constructor::infer(request, entry)
}

pub(super) fn unary(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 1);
    receiver(request, 0, &mut diagnostics);
    // Reconstruct stable property facts; transpose and scale invalidate values.
    finish_fixed(entry, request, object(), diagnostics)
}

pub(super) fn scale(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 2);
    for (index, value) in request.arguments.iter().enumerate() {
        if !is_object(value) && value.shape.element_count().is_some_and(|n| n != 1) {
            diagnostics.push(argument_error(
                "RM-DECOMPOSITION-SCALE",
                "scale factor must be scalar",
                index,
            ));
        }
    }
    finish_fixed(entry, request, object(), diagnostics)
}

pub(super) fn right_divide(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = arity(request, 2);
    let result = if request.arguments.first().is_some_and(is_object) {
        object()
    } else {
        // Right solve precision, complexity and dimensions require the opaque
        // coefficient payload; input object shape is not coefficient shape.
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed(entry, request, result, diagnostics)
}

pub(super) fn left_divide(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 2);
    receiver(request, 0, &mut diagnostics);
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        diagnostics,
    )
}

pub(super) fn property(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity(request, 2);
    receiver(request, 0, &mut diagnostics);
    // LiteralContext cannot encode a substruct path, including subsequent
    // indexing. Do not infer the property from the shape of that path value.
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        diagnostics,
    )
}

#[cfg(test)]
mod tests;
