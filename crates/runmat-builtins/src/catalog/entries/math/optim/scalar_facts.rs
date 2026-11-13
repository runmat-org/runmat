//! Static callback and scalar facts shared by optimization and quadrature.

use crate::catalog::inference::argument_error;
use runmat_types::{
    CallInference, CallRequest, DimensionFact, DynamicReason, InferenceDiagnostic, LiteralContext,
    NumericClass, NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact,
    StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn callback(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(value) = request.arguments.first() else {
        return;
    };
    let valid_kind = matches!(
        value.kind,
        ValueKindFact::Callable(_)
            | ValueKindFact::Object(_)
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Unknown
    );
    let valid_text_geometry = match value.kind {
        ValueKindFact::Character => !matches!(
            &value.shape,
            ShapeFact::Shaped { dims }
                if matches!(dims.first(), Some(DimensionFact::Known(rows)) if *rows != 1)
        ),
        ValueKindFact::String => {
            value.shape.element_count().is_none_or(|count| count == 1)
                && !matches!(
                    &value.shape,
                    ShapeFact::Shaped { dims }
                        if dims.iter().any(|dim| matches!(dim, DimensionFact::Known(count) if *count != 1))
                )
        }
        _ => true,
    };
    if !valid_kind || !valid_text_geometry {
        diagnostics.push(argument_error(
            "RM-CATALOG-SCALAR-OPT-CALLBACK",
            "the first argument must identify a callable scalar function",
            0,
        ));
    }
}

pub(super) fn callback_invocation(
    request: &CallRequest,
    argument: ValueFact,
) -> Option<CallInference> {
    callback_invocation_with_arguments(request, vec![argument])
}

pub(super) fn callback_invocation_with_arguments(
    request: &CallRequest,
    arguments: Vec<ValueFact>,
) -> Option<CallInference> {
    let ValueKindFact::Callable(callable) = &request.arguments.first()?.kind else {
        return None;
    };
    Some(crate::infer_callable_call(
        callable,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    ))
}

pub(super) fn real_scalar_callback_result(
    callback: &CallInference,
    diagnostics: &mut Vec<InferenceDiagnostic>,
    code: &'static str,
) {
    let Some(result) = callback.outputs.first() else {
        return;
    };
    let class_ok = matches!(
        result.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    );
    let scalar = result.shape.element_count().is_none_or(|count| count == 1);
    if !class_ok || !scalar || result.storage == StorageFact::Sparse {
        diagnostics.push(argument_error(
            code,
            "quadrature callback must return a real numeric scalar",
            0,
        ));
    }
}

pub(super) fn real_numeric(
    value: Option<&ValueFact>,
    index: usize,
    allowed_counts: &[usize],
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(value) = value else { return };
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) || value
        .shape
        .element_count()
        .is_some_and(|count| !allowed_counts.contains(&count))
        || value.storage == StorageFact::Sparse
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SCALAR-OPT-INPUT",
            "solver input must be a real numeric scalar or supported interval",
            index,
        ));
    }
}

pub(super) fn options(
    value: Option<&ValueFact>,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(value) = value else { return };
    if !matches!(
        value.kind,
        ValueKindFact::Struct(_) | ValueKindFact::Unknown
    ) || value.shape.element_count().is_some_and(|count| count != 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-SCALAR-OPT-OPTIONS",
            "options must be a scalar struct",
            index,
        ));
    }
}

pub(super) fn host_scalar(class: NumericClass) -> ValueFact {
    let mut fact = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }));
    // Single scalars use the runtime's one-element native tensor carrier.
    if class == NumericClass::Single {
        fact.storage = StorageFact::Dense;
    }
    fact
}

pub(super) fn conditional_host_scalar() -> ValueFact {
    let mut fact = ValueFact::unknown(DynamicReason::RuntimeValue);
    fact.shape = ShapeFact::Scalar;
    fact.storage = StorageFact::Scalar;
    fact.residency = runmat_types::ResidencyFact::Host;
    fact
}

pub(super) fn diagnostic_struct() -> ValueFact {
    let mut fields = BTreeMap::new();
    fields.insert("iterations".into(), host_scalar(NumericClass::Double));
    fields.insert("funcCount".into(), host_scalar(NumericClass::Double));
    fields.insert("algorithm".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert("message".into(), ValueFact::scalar(ValueKindFact::String));
    ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(fields, true)),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    )
}
