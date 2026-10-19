use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, ObjectFact, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GEOMETRY-LOAD-ARITY",
            "geometry.load requires one path",
            0,
        ));
    }
    if let Some(path) = request.arguments.first() {
        let invalid_shape = match (&path.kind, &path.shape) {
            (ValueKindFact::String, ShapeFact::Shaped { dims }) => dims
                .iter()
                .any(|dim| matches!(dim, DimensionFact::Known(extent) if *extent != 1)),
            (ValueKindFact::Character, ShapeFact::Shaped { dims }) => {
                dims.len() > 2
                    || dims
                        .first()
                        .is_some_and(|dim| matches!(dim, DimensionFact::Known(rows) if *rows != 1))
            }
            (ValueKindFact::Character, ShapeFact::Ranked { rank }) => *rank > 2,
            _ => false,
        };
        if invalid_shape
            || !matches!(
                path.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GEOMETRY-LOAD-PATH",
                "path must be scalar text",
                0,
            ));
        }
    }
    let mut output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::GEOMETRY_ASSET_CLASS.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    output.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
