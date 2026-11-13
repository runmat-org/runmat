use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CellFact, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, StructFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GEOMETRY-MESHES-ARITY",
            "geometry.meshes requires one asset",
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
                "RM-CATALOG-GEOMETRY-MESHES-ASSET",
                "expected geometry.Asset",
                0,
            ))
        }
        Some(ValueKindFact::Object(_)) | Some(ValueKindFact::Unknown) | None => {}
        _ => diagnostics.push(argument_error(
            "RM-CATALOG-GEOMETRY-MESHES-ASSET",
            "expected geometry.Asset",
            0,
        )),
    }
    let numeric = |columns| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![None, Some(columns)]),
            StorageFact::Dense,
        )
    };
    let mut fields = BTreeMap::new();
    fields.insert("mesh_id".into(), ValueFact::scalar(ValueKindFact::String));
    fields.insert("vertices".into(), numeric(3));
    fields.insert("triangles".into(), numeric(3));
    fields.insert("faces".into(), numeric(3));
    fields.insert(
        "region_mappings".into(),
        ValueFact::unknown(DynamicReason::RuntimeValue),
    );
    let element = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let mut output = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(1), None]),
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
