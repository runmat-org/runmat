use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-AUDIOINFO-ARITY",
            "audioinfo requires one filename",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !super::super::super::repl_fs::text_input::scalar(value))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-AUDIOINFO-TEXT",
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
            "RM-CATALOG-AUDIOINFO-EMPTY",
            "filename must not be empty",
            0,
        ));
    }
    let mut contract = CallContract::fixed(vec![metadata()]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn metadata() -> ValueFact {
    let fields = BTreeMap::from([
        ("Filename".into(), text()),
        ("CompressionMethod".into(), text()),
        ("NumChannels".into(), double()),
        ("SampleRate".into(), double()),
        ("TotalSamples".into(), double()),
        ("Duration".into(), double()),
        ("BitsPerSample".into(), double()),
        ("BitRate".into(), double()),
        ("FileSize".into(), double()),
        ("Format".into(), text()),
    ]);
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)))
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn double() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
