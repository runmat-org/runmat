use super::math_fact_transforms::{materialize_output, preserve_shape_on_dynamic_input};
use super::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, RoundingFunction};
use runmat_types::{
    AliasFact, CallInference, CallRequest, DynamicReason, LiteralValue, MutationFact, NumericClass,
    NumericDomain, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer_round(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUND-ARITY",
            "round requires one to three inputs",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUND-ARITY",
            "round accepts at most three inputs",
            3,
        ));
    }

    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUND-SPARSE",
            "round does not currently accept sparse input",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    if let Some(digits) = request.arguments.get(1) {
        let scalar_control = digits.is_scalar()
            && matches!(
                digits.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
            );
        if !scalar_control {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-DIGITS",
                "round requires N to be an integer-valued numeric scalar",
                1,
            ));
        }
        if let Some(LiteralValue::Number(value)) = request.literals.literal_args.get(1) {
            if !value.is_finite()
                || value.fract() != 0.0
                || *value < f64::from(i32::MIN)
                || *value > f64::from(i32::MAX)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ROUND-DIGITS",
                    "round requires N to be a finite integer in the supported signed range",
                    1,
                ));
            }
        }
    }

    if let Some(mode) = request.arguments.get(2) {
        if !matches!(
            mode.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-MODE",
                "round requires mode to be \"decimals\" or \"significant\"",
                2,
            ));
        }
        let mode_literal = request
            .literals
            .literal_args
            .get(2)
            .and_then(|literal| match literal {
                LiteralValue::String(value)
                | LiteralValue::Character(value)
                | LiteralValue::Keyword(value) => Some(value.as_str()),
                _ => None,
            });
        if let Some(mode) = mode_literal {
            if !matches!(
                mode.to_ascii_lowercase().as_str(),
                "decimal" | "decimals" | "significant"
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ROUND-MODE",
                    "round requires mode to be \"decimals\" or \"significant\"",
                    2,
                ));
            }
            if mode.eq_ignore_ascii_case("significant")
                && matches!(
                    request.literals.literal_args.get(1),
                    Some(LiteralValue::Number(value)) if *value <= 0.0
                )
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ROUND-DIGITS",
                    "round requires positive N for significant-digit rounding",
                    1,
                ));
            }
        }
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-COMPLEX-INTEGER",
                "round does not accept complex fixed-width integer input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric)
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single)
                && request.arguments.len() > 1 =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-INTEGER-FORM",
                "typed integer X supports only round(X)",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric)
            if matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            materialize_output(&mut output);
            output.residency = input.residency.clone();
        }
        ValueKindFact::Numeric(_) => {}
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
            output.residency = input.residency.clone();
        }
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUND-INPUT",
                "round requires numeric, logical, or character input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_rounding(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: RoundingFunction,
) -> CallInference {
    let name = match function {
        RoundingFunction::Ceil => "ceil",
        RoundingFunction::Fix => "fix",
        RoundingFunction::Floor => "floor",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUNDING-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUNDING-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }

    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ROUNDING-SPARSE",
            format!("{name} does not currently accept sparse input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUNDING-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric)
            if matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            materialize_output(&mut output);
            output.residency = input.residency.clone();
        }
        ValueKindFact::Numeric(_) => {
            // Every real fixed-width integer is already integral. The runtime
            // returns the exact storage (and resident handle) unchanged.
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
            output.residency = input.residency.clone();
        }
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_some_and(|class| {
                class.is(runmat_types::standard::TABLE)
                    || class.is(runmat_types::standard::TIMETABLE)
            }) =>
        {
            object.properties.clear();
            object.properties_complete = false;
            output.alias = AliasFact::Unique;
            output.mutation = MutationFact::ValueSemantics;
        }
        ValueKindFact::Object(_) | ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-ROUNDING-INPUT",
                format!("{name} requires numeric, logical, character, or supported tabular input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}
