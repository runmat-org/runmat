use super::super::{nonlinear_facts as facts, scalar_facts};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let initial = if request.arguments.len() == 1 {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(value) => value.fields.get("x0"),
            _ => None,
        }
    } else {
        request.arguments.get(1)
    };
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FMINSEARCH-ARITY",
            "expected fun, x0 and optional options, or a problem structure",
            0,
        ));
    }
    if request.arguments.len() == 1
        && !matches!(
            request.arguments[0].kind,
            ValueKindFact::Unknown | ValueKindFact::Struct(_)
        )
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FMINSEARCH-PROBLEM",
            "one-input form requires a scalar problem structure",
            0,
        ));
    }
    let options = if request.arguments.len() == 1 {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(value) => value.fields.get("options"),
            _ => None,
        }
    } else {
        request.arguments.get(2)
    };
    if let Some(value) = options {
        let default = value.shape.element_count() == Some(0)
            && matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
            );
        if !default
            && (!matches!(
                value.kind,
                ValueKindFact::Unknown | ValueKindFact::Struct(_)
            ) || value.shape.element_count().is_some_and(|count| count != 1))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FMINSEARCH-OPTIONS",
                "options must be a scalar structure or numeric []",
                2,
            ));
        }
        if let ValueKindFact::Struct(options) = &value.kind {
            for (name, value) in &options.fields {
                let default = value.shape.element_count() == Some(0)
                    && matches!(
                        value.kind,
                        ValueKindFact::Numeric(NumericFact {
                            domain: NumericDomain::Real,
                            ..
                        })
                    );
                if !default && super::FminsearchOptionName::from_public(name).is_none() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FMINSEARCH-OPTION",
                        "unsupported nonempty fminsearch option",
                        2,
                    ));
                }
            }
        }
    }
    if let Some(value) = initial {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real
                })
        ) || value.storage == StorageFact::Sparse
            || value.shape.element_count() == Some(0)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FMINSEARCH-INITIAL",
                "initial guess must be a nonempty real double array",
                1,
            ));
        }
    }
    let x = facts::like(initial);
    let callback = if request.arguments.len() >= 2 {
        scalar_facts::callback(request, &mut diagnostics);
        scalar_facts::callback_invocation_with_arguments(request, vec![x.clone()])
    } else {
        None
    };
    if let Some(value) = callback.as_ref().and_then(|c| c.outputs.first()) {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double | NumericClass::Single,
                    domain: NumericDomain::Real
                })
        ) || value.storage == StorageFact::Sparse
            || value.shape.element_count().is_some_and(|n| n != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FMINSEARCH-OBJECTIVE",
                "objective must return a real floating scalar",
                0,
            ));
        }
    }
    facts::finish(
        request,
        entry,
        vec![
            x,
            facts::scalar(),
            facts::scalar(),
            facts::diagnostics(&["iterations", "funcCount"]),
        ],
        diagnostics,
        callback,
    )
}
