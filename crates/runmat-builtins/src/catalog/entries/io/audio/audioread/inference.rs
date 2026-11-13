use super::super::AudioReadDataType;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-AUDIOREAD-ARITY",
            "audioread requires a filename and at most two controls",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !scalar_text(value))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-AUDIOREAD-TEXT",
            "filename must be host scalar text",
            0,
        ));
    }
    if request
        .literals
        .literal_string_at(0)
        .is_some_and(|text| text.trim().is_empty())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-AUDIOREAD-EMPTY",
            "filename must not be empty",
            0,
        ));
    }
    let mode_index = match request.arguments.len() {
        2 if scalar_text(&request.arguments[1])
            && !matches!(request.arguments[1].kind, ValueKindFact::Unknown) =>
        {
            Some(1)
        }
        3 => {
            check_range(&request.arguments[1], &mut diagnostics);
            Some(2)
        }
        2 => {
            check_range_or_text(&request.arguments[1], &mut diagnostics);
            None
        }
        _ => None,
    };
    if let Some(index) = mode_index {
        if !scalar_text(&request.arguments[index]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-AUDIOREAD-MODE",
                "datatype must be scalar text",
                index,
            ));
        }
    }
    let literal_mode = mode_index.and_then(|index| request.literals.literal_string_at(index));
    let parsed_mode = literal_mode.as_deref().and_then(AudioReadDataType::parse);
    if literal_mode.is_some() && parsed_mode.is_none() {
        diagnostics.push(argument_error(
            "RM-CATALOG-AUDIOREAD-MODE",
            "datatype must be native or double",
            mode_index.unwrap_or(1),
        ));
    }
    let known_double = request.arguments.len() == 1
        || parsed_mode == Some(AudioReadDataType::Double)
        || (request.arguments.len() == 2
            && mode_index.is_none()
            && is_known_range(&request.arguments[1]));
    let mut samples = if known_double {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![None, None]),
            StorageFact::Dense,
        )
    } else {
        let mut unknown = ValueFact::unknown(DynamicReason::ExternalData);
        unknown.shape = ShapeFact::from(vec![None, None]);
        unknown.storage = StorageFact::Dense;
        unknown.layout = LayoutFact::ColumnMajor;
        unknown.contiguity = ContiguityFact::Contiguous;
        unknown.view = ViewFact::Materialized;
        unknown.residency = ResidencyFact::Host;
        unknown
    };
    samples.residency = ResidencyFact::Host;
    let sample_rate = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut contract = CallContract::fixed(vec![samples, sample_rate]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn scalar_text(value: &ValueFact) -> bool {
    super::super::super::repl_fs::text_input::scalar(value)
}

fn is_known_range(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    ) && value.shape.element_count().is_none_or(|count| count == 2)
}

fn check_range_or_text(value: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if !scalar_text(value) {
        check_range(value, diagnostics);
    }
}

fn check_range(value: &ValueFact, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if !matches!(value.kind, ValueKindFact::Unknown) && !is_known_range(value) {
        diagnostics.push(argument_error(
            "RM-CATALOG-AUDIOREAD-RANGE",
            "samples must be a two-element real double array",
            1,
        ));
    }
}
