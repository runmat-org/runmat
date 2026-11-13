use super::super::{OptimParameterLookup, OptimParameterName};
use crate::catalog::inference::{argument_error, finish_fixed, gathered_value_fact, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, FactJoin, ShapeFact, StructFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let args = &request.arguments;
    let mut diagnostics = Vec::new();
    let mut fields = BTreeMap::new();
    if args.is_empty()
        || args.len() == 1
            && request
                .literals
                .literal_args
                .first()
                .and_then(literal_text)
                .is_some_and(|name| name.eq_ignore_ascii_case("fminsearch"))
    {
        for field in super::super::FminsearchOptionName::ALL {
            fields.insert(
                field.name().to_owned(),
                ValueFact::unknown(DynamicReason::RuntimeValue),
            );
        }
        return finish_fixed(
            entry,
            request,
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true))),
            Vec::new(),
        );
    }
    // The two-structure merge omits empty new values. Without exact placement
    // and emptiness facts, describe the successful structure conservatively.
    if args.len() == 2
        && (matches!(args[0].kind, ValueKindFact::Struct(_))
            || args[0].shape.element_count() == Some(0)
                && matches!(
                    args[0].kind,
                    ValueKindFact::Numeric(runmat_types::NumericFact {
                        domain: runmat_types::NumericDomain::Real,
                        ..
                    })
                ))
        && matches!(args[1].kind, ValueKindFact::Struct(_))
    {
        return finish_fixed(
            entry,
            request,
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, false))),
            Vec::new(),
        );
    }
    if args.len() == 1 && !matches!(args[0].kind, ValueKindFact::Struct(_)) {
        if request
            .literals
            .literal_args
            .first()
            .and_then(literal_text)
            .is_some()
        {
            diagnostics.push(argument_error(
                "RM.OPTIMSET.INVALID_ARGUMENT",
                "solver defaults are currently available for fminsearch only",
                0,
            ));
        }
        return finish_fixed(
            entry,
            request,
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, false))),
            diagnostics,
        );
    }
    let mut complete = true;
    let pair_start = if let Some(first) = args.first() {
        if let ValueKindFact::Struct(StructFact {
            fields: old,
            fields_complete,
            ..
        }) = &first.kind
        {
            if first.shape.element_count().is_some_and(|count| count != 1) {
                diagnostics.push(argument_error(
                    "RM.OPTIMSET.INVALID_INPUT",
                    "optimset requires a scalar options struct",
                    0,
                ));
            }
            fields = old
                .iter()
                .map(|(name, value)| {
                    (
                        match OptimParameterName::resolve(name) {
                            OptimParameterLookup::Matched(field)=>field.name().to_owned(),_=>name.clone()
                        },
                        if matches!(OptimParameterName::resolve(name),OptimParameterLookup::Matched(field)if field.is_callback())
                        {
                            value.clone()
                        } else {
                            possibly_materialized(value)
                        },
                    )
                })
                .collect();
            complete = *fields_complete;
            1
        } else if first.shape.element_count() == Some(0)
            && matches!(
                first.kind,
                ValueKindFact::Numeric(runmat_types::NumericFact {
                    domain: runmat_types::NumericDomain::Real,
                    ..
                })
            )
        {
            1
        } else if matches!(first.kind, ValueKindFact::Unknown) && !args.len().is_multiple_of(2) {
            // On a successful odd-arity call, the unknown first value must
            // be the existing struct. Its unknown fields can be overridden
            // by the remaining, positionally bound name/value pairs.
            complete = false;
            1
        } else {
            0
        }
    } else {
        0
    };

    if !(args.len() - pair_start).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM.OPTIMSET.INVALID_ARGUMENT",
            "optimset requires complete name/value pairs",
            args.len().saturating_sub(1),
        ));
    }
    for index in (pair_start..args.len().saturating_sub(1)).step_by(2) {
        if !text_scalar_possible(&args[index]) {
            diagnostics.push(argument_error(
                "RM.OPTIMSET.INVALID_INPUT",
                "optimset option name must be a text scalar",
                index,
            ));
            continue;
        }
        let Some(name) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        else {
            // A dynamic name may replace any field already present, so those
            // earlier value facts cannot survive as proven output facts.
            fields.clear();
            complete = false;
            continue;
        };
        let field_name = match OptimParameterName::resolve(&name) {
            OptimParameterLookup::Matched(field) => field.name().to_owned(),
            OptimParameterLookup::Unknown => name,
            OptimParameterLookup::Ambiguous => {
                diagnostics.push(argument_error(
                    "RM.OPTIMSET.INVALID_INPUT",
                    "ambiguous legacy option abbreviation",
                    index,
                ));
                continue;
            }
        };
        // A stored explicit resident payload can remain on its device. A fact
        // does not expose the placement origin, so don't invent host residency.
        let value = if matches!(OptimParameterName::resolve(&field_name),OptimParameterLookup::Matched(field)if field.is_callback())
        {
            args[index + 1].clone()
        } else {
            possibly_materialized(&args[index + 1])
        };
        fields.insert(field_name, value);
    }
    let output = if diagnostics.is_empty() {
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, complete)))
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn text_scalar_possible(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Character => match &value.shape {
            ShapeFact::Shaped { dims } => dims.first().is_none_or(
                |rows| !matches!(rows, runmat_types::DimensionFact::Known(rows) if *rows != 1),
            ),
            _ => true,
        },
        ValueKindFact::Unknown => true,
        _ => false,
    }
}

fn possibly_materialized(value: &ValueFact) -> ValueFact {
    // The static fact does not encode explicit versus automatic placement.
    // Either the whole payload is retained, or the canonical gather recursively
    // materializes it. Join those real transforms without erasing its class,
    // shape or nested structure and without inventing host residency.
    value.join(&gathered_value_fact(value))
}
