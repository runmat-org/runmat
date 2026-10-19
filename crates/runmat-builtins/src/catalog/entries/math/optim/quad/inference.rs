use super::super::scalar_facts;
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, StorageFact, ValueFact, ValueKindFact,
};

fn finite_scalar(
    request: &CallRequest,
    index: usize,
    role: &'static str,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    let Some(fact) = request.arguments.get(index) else {
        return;
    };
    let class_ok = matches!(
        fact.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    );
    let shape_ok = fact.shape.element_count().is_none_or(|count| count == 1);
    let finite = request
        .literals
        .literal_args
        .get(index)
        .and_then(runmat_types::LiteralContext::numeric_from_literal)
        .is_none_or(f64::is_finite);
    if !class_ok || !shape_ok || !finite || fact.storage == StorageFact::Sparse {
        diagnostics.push(argument_error("RM-CATALOG-QUAD-SCALAR", role, index));
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-QUAD-ARITY",
            "quad requires a callback and two bounds",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-QUAD-OUTPUTS",
            "quad returns at most two outputs",
            2,
        ));
    }
    scalar_facts::callback(request, &mut diagnostics);
    finite_scalar(
        request,
        1,
        "quad lower bound must be a finite real scalar",
        &mut diagnostics,
    );
    finite_scalar(
        request,
        2,
        "quad upper bound must be a finite real scalar",
        &mut diagnostics,
    );
    if request.arguments.len() > 3 {
        let fact = &request.arguments[3];
        let empty = fact.shape.element_count() == Some(0)
            && matches!(
                fact.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            );
        if !empty {
            finite_scalar(
                request,
                3,
                "quad tolerance must be a positive real scalar or empty",
                &mut diagnostics,
            );
        }
        if let Some(tol) = request
            .literals
            .literal_args
            .get(3)
            .and_then(runmat_types::LiteralContext::numeric_from_literal)
        {
            if tol <= 0.0 {
                diagnostics.push(argument_error(
                    "RM-CATALOG-QUAD-TOL",
                    "quad tolerance must be positive",
                    3,
                ));
            }
        }
    }
    if request.arguments.len() > 4 {
        let fact = &request.arguments[4];
        let admitted = matches!(
            fact.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        ) && fact.storage != StorageFact::Sparse
            && !fact
                .shape
                .element_count()
                .is_some_and(|count| count != 0 && count != 1);
        if !admitted {
            diagnostics.push(argument_error(
                "RM-CATALOG-QUAD-TRACE",
                "quad trace must be a scalar toggle or empty",
                4,
            ));
        }
    }
    let mut callback_arguments = vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))];
    callback_arguments.extend(request.arguments.iter().skip(5).cloned());
    let callback = scalar_facts::callback_invocation_with_arguments(request, callback_arguments);
    if let Some(callback) = &callback {
        scalar_facts::real_scalar_callback_result(
            callback,
            &mut diagnostics,
            "RM-CATALOG-QUAD-CALLBACK-RESULT",
        );
        diagnostics.extend(callback.diagnostics.iter().cloned());
    }
    let scalar = if diagnostics.is_empty() {
        scalar_facts::host_scalar(NumericClass::Double)
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    let mut contract = CallContract::fixed(vec![scalar.clone(), scalar]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if let Some(callback) = callback {
        contract.effects.0.extend(callback.effects.0);
        contract.capabilities.0.extend(callback.capabilities.0);
    }
    let mut inference = infer_call(&contract, request);
    inference.diagnostics.extend(diagnostics);
    inference
}
