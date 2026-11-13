//! Shared fact construction and typed option decoding.
use super::super::{EncodingOptionName, EncodingOutputKind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, ObjectFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn error(code: &str, message: &str) -> InferenceDiagnostic {
    InferenceDiagnostic::error(code, message)
}

pub(super) fn finish(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    output: ValueFact,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let output = if diagnostics.is_empty() {
        output
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

pub(super) fn literal_text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::Character(text)
        | LiteralValue::String(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

pub(super) fn literal_dim(value: Option<&LiteralValue>) -> Result<Option<usize>, ()> {
    match value {
        None | Some(LiteralValue::Unknown) => Ok(None),
        Some(LiteralValue::Integer { text, .. }) => text
            .parse::<usize>()
            .ok()
            .filter(|dim| *dim > 0)
            .map(Some)
            .ok_or(()),
        Some(value) => LiteralContext::numeric_from_literal(value)
            .and_then(|number| runmat_types::decode_shape_dimension(number).ok())
            .filter(|dim| *dim > 0)
            .map(Some)
            .ok_or(()),
    }
}

#[derive(Clone, Copy)]
pub(super) enum OutputSelection {
    Known(EncodingOutputKind),
    Dynamic,
    Invalid,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum EncodingInferenceFlavor {
    Encode,
    Decode,
}

impl EncodingInferenceFlavor {
    const fn default_output(self) -> EncodingOutputKind {
        match self {
            Self::Encode => EncodingOutputKind::Numeric(NumericClass::Double),
            Self::Decode => EncodingOutputKind::Categorical,
        }
    }
}

pub(super) fn output_selection(
    request: &CallRequest,
    start: usize,
    flavor: EncodingInferenceFlavor,
) -> OutputSelection {
    let count = request.arguments.len().saturating_sub(start);
    if count == 0 {
        return OutputSelection::Known(flavor.default_output());
    }
    let mut idx = 0;
    let mut selected = None;
    if count == 1 || (flavor == EncodingInferenceFlavor::Encode && !count.is_multiple_of(2)) {
        let Some(value) = request.literals.literal_args.get(start) else {
            return OutputSelection::Dynamic;
        };
        let Some(text) = literal_text(value) else {
            return if matches!(value, LiteralValue::Unknown) {
                OutputSelection::Dynamic
            } else {
                OutputSelection::Invalid
            };
        };
        selected = EncodingOutputKind::from_public(text);
        if selected.is_none() {
            return OutputSelection::Invalid;
        }
        idx = 1;
    }
    if !(count - idx).is_multiple_of(2) {
        return OutputSelection::Invalid;
    }
    while idx < count {
        let Some(name) = request.literals.literal_args.get(start + idx) else {
            return OutputSelection::Dynamic;
        };
        let Some(text) = literal_text(name) else {
            return if matches!(name, LiteralValue::Unknown) {
                OutputSelection::Dynamic
            } else {
                OutputSelection::Invalid
            };
        };
        match EncodingOptionName::from_public(text) {
            Some(EncodingOptionName::Classes) if flavor == EncodingInferenceFlavor::Encode => {}
            Some(EncodingOptionName::OutputType) => {
                let Some(value) = request.literals.literal_args.get(start + idx + 1) else {
                    return OutputSelection::Dynamic;
                };
                let Some(text) = literal_text(value) else {
                    return if matches!(value, LiteralValue::Unknown) {
                        OutputSelection::Dynamic
                    } else {
                        OutputSelection::Invalid
                    };
                };
                selected = EncodingOutputKind::from_public(text);
                if selected.is_none() {
                    return OutputSelection::Invalid;
                }
            }
            _ => return OutputSelection::Invalid,
        }
        idx += 2;
    }
    let selected = selected.unwrap_or(flavor.default_output());
    if flavor == EncodingInferenceFlavor::Encode
        && !matches!(
            selected,
            EncodingOutputKind::Numeric(_) | EncodingOutputKind::Logical
        )
    {
        OutputSelection::Invalid
    } else {
        OutputSelection::Known(selected)
    }
}

pub(super) fn selected_fact(selection: OutputSelection, shape: ShapeFact) -> ValueFact {
    match selection {
        OutputSelection::Known(EncodingOutputKind::Numeric(class)) => numeric(class, shape),
        OutputSelection::Known(EncodingOutputKind::Logical) => {
            ValueFact::proven(ValueKindFact::Logical, shape, StorageFact::Dense)
        }
        OutputSelection::Known(EncodingOutputKind::String) => {
            ValueFact::proven(ValueKindFact::String, shape, StorageFact::Dense)
        }
        OutputSelection::Known(EncodingOutputKind::Categorical) => ValueFact::proven(
            ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(runmat_types::standard::CATEGORICAL.owned()),
                properties: Default::default(),
                properties_complete: false,
                handle_semantics: Some(false),
            }),
            ShapeFact::Unknown,
            StorageFact::Unknown,
        ),
        OutputSelection::Known(EncodingOutputKind::Cell)
        | OutputSelection::Dynamic
        | OutputSelection::Invalid => ValueFact::unknown(DynamicReason::RuntimeValue),
    }
}
