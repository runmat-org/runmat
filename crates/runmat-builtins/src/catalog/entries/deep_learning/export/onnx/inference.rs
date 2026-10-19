use super::{ExportOnnxOptionKey, ExportOnnxOptionValueRole};
use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, ClassIdentity, NumericDomain, ResidencyFact, ValueFact,
    ValueKindFact,
};

fn is_network(identity: &ClassIdentity) -> bool {
    identity.is(standard::DL_NETWORK)
        || identity.is(standard::SERIES_NETWORK)
        || identity.is(standard::DAG_NETWORK)
}

fn scalar_text(value: &ValueFact) -> bool {
    if value.residency != ResidencyFact::Host && value.residency != ResidencyFact::Unknown {
        return false;
    }
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Character => value
            .shape
            .known_dims()
            .is_none_or(|dims| dims.first().copied().flatten().is_none_or(|rows| rows == 1)),
        _ => false,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let args = &request.arguments;
    if args.len() < 2 || !(args.len() - 2).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-DL-EXPORTONNX-ARITY",
            "exportONNXNetwork requires a network, filename, and complete name-value pairs",
            args.len().saturating_sub(1),
        ));
    }
    if let Some(network) = args.first() {
        let possible = match &network.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => object.runtime_class.as_ref().is_none_or(is_network),
            _ => false,
        };
        if !possible {
            diagnostics.push(argument_error(
                "RM-DL-EXPORTONNX-NETWORK",
                "net must be a supported deep-learning network object",
                0,
            ));
        }
    }
    if let Some(filename) = args.get(1) {
        if !scalar_text(filename) {
            diagnostics.push(argument_error(
                "RM-DL-EXPORTONNX-FILENAME",
                "filename must be host scalar text",
                1,
            ));
        }
    }
    for index in (2..args.len()).step_by(2) {
        if !scalar_text(&args[index]) {
            diagnostics.push(argument_error(
                "RM-DL-EXPORTONNX-OPTION-NAME",
                "option names must be host scalar text",
                index,
            ));
        }
        if let Some(name) = request.literals.literal_string_at(index) {
            if let Some(key) = ExportOnnxOptionKey::from_name(&name) {
                if let Some(value) = args.get(index + 1) {
                    let possible = match key.value_role() {
                        ExportOnnxOptionValueRole::SingleName => {
                            scalar_text(value)
                                || matches!(value.kind, ValueKindFact::Cell(_))
                                    && value.shape.element_count().is_none_or(|count| count == 1)
                        }
                        ExportOnnxOptionValueRole::PositiveInteger => {
                            matches!(
                                &value.kind,
                                ValueKindFact::Unknown
                                    | ValueKindFact::Numeric(runmat_types::NumericFact {
                                        domain: NumericDomain::Real,
                                        ..
                                    })
                            ) && value.shape.element_count().is_none_or(|count| count == 1)
                        }
                        ExportOnnxOptionValueRole::LogicalFlag => {
                            matches!(
                                &value.kind,
                                ValueKindFact::Unknown
                                    | ValueKindFact::Logical
                                    | ValueKindFact::Numeric(runmat_types::NumericFact {
                                        domain: NumericDomain::Real,
                                        ..
                                    })
                            ) && value.shape.element_count().is_none_or(|count| count == 1)
                        }
                    };
                    if !possible {
                        diagnostics.push(argument_error(
                            "RM-DL-EXPORTONNX-OPTION-VALUE",
                            "option value has an invalid class or shape",
                            index + 1,
                        ));
                    }
                }
            } else {
                diagnostics.push(argument_error(
                    "RM-DL-EXPORTONNX-OPTION",
                    "unsupported exportONNXNetwork option",
                    index,
                ));
            }
        }
    }
    finish_fixed_outputs(entry, request, vec![], diagnostics)
}
