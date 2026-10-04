use super::UnzipSourceRoute;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-UNZIP-ARITY",
            "unzip requires a ZIP source and optional output folder",
            0,
        ));
    }
    for (index, value) in request.arguments.iter().take(2).enumerate() {
        if !could_be_scalar_text(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-UNZIP-TEXT",
                "ZIP source and output folder must be scalar host text",
                index,
            ));
        }
    }
    for index in 0..request.arguments.len().min(2) {
        if request
            .literals
            .literal_string_at(index)
            .is_some_and(|text| text.trim().is_empty())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-UNZIP-EMPTY",
                "path arguments must not be empty",
                index,
            ));
        }
    }

    let element = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    let mut output = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: vec![],
            elements_complete: false,
        }),
        ShapeFact::from(vec![None, Some(1)]),
        StorageFact::Dense,
    );
    output.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if let Some(source) = request.literals.literal_string_at(0) {
        match UnzipSourceRoute::parse(source) {
            UnzipSourceRoute::Http(_) => {
                contract.effects.0.remove(&EffectKind::FilesystemRead);
                // HTTP supplies source bytes; extraction still requires the
                // active filesystem provider for output writes.
            }
            UnzipSourceRoute::LocalPath(_) => {
                contract.effects.0.remove(&EffectKind::Network);
                contract
                    .capabilities
                    .0
                    .remove(&CapabilityRequirement::Network);
            }
        }
    }
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn could_be_scalar_text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Character => match &value.shape {
            ShapeFact::Shaped { dims } => {
                !dims
                    .first()
                    .is_some_and(|dim| matches!(dim, DimensionFact::Known(rows) if *rows != 1))
                    && !dims
                        .iter()
                        .skip(2)
                        .any(|dim| matches!(dim, DimensionFact::Known(extent) if *extent != 1))
            }
            _ => true,
        },
        ValueKindFact::Unknown => true,
        _ => false,
    }
}
