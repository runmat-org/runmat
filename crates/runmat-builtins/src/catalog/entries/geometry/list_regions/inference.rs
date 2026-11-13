use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, ResidencyFact, ShapeFact, StorageFact,
    StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GEOMETRY-LISTREGIONS-ARITY",
            "geometry.listRegions requires one asset",
            0,
        ));
    }
    match request.arguments.first().map(|value| &value.kind) {
        Some(ValueKindFact::Object(object))
            if object.runtime_class.as_ref().is_some_and(|class| {
                class != &super::super::load::GEOMETRY_ASSET_CLASS.owned()
            }) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GEOMETRY-LISTREGIONS-ASSET",
                "expected geometry.Asset",
                0,
            ))
        }
        Some(ValueKindFact::Object(_)) | Some(ValueKindFact::Unknown) | None => {}
        _ => diagnostics.push(argument_error(
            "RM-CATALOG-GEOMETRY-LISTREGIONS-ASSET",
            "expected geometry.Asset",
            0,
        )),
    }
    let mut fields = BTreeMap::new();
    fields.insert(
        "regions".into(),
        ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
    );
    let mut output = ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(fields, true)),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    output.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
