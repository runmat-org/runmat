use crate::{
    catalog::{argument_error, finish_fixed},
    fea_contract::constructor_inference::host_text,
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, ObjectFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

use super::super::GEOMETRY_ASSET_CLASS;
use super::GEOMETRY_REGION_CLASS;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GEOMETRY-REGION-ARITY",
            "geometry.region requires an asset and a selector",
            request.arguments.len(),
        ));
    }
    if let Some(asset) = request.arguments.first() {
        if !possible_asset(asset) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GEOMETRY-REGION-ASSET",
                "geometry.region requires a host geometry.Asset",
                0,
            ));
        }
    }
    if let Some(selector) = request.arguments.get(1) {
        if !host_text(selector) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GEOMETRY-REGION-SELECTOR",
                "geometry.region selector must be scalar text",
                1,
            ));
        } else if request
            .literals
            .literal_string_at(1)
            .is_some_and(|selector| selector.is_empty())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GEOMETRY-REGION-SELECTOR",
                "geometry.region selector must not be empty",
                1,
            ));
        }
    }
    finish_fixed(entry, request, region_result(), diagnostics)
}

pub(in crate::catalog::entries::geometry) fn possible_asset(fact: &ValueFact) -> bool {
    !matches!(fact.residency, ResidencyFact::Device { .. })
        && !matches!(fact.shape.element_count(), Some(count) if count != 1)
        && match &fact.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_none_or(|class| class.is(GEOMETRY_ASSET_CLASS)),
            _ => false,
        }
}

fn region_result() -> ValueFact {
    let text = || ValueFact::scalar(ValueKindFact::String);
    let mut properties = BTreeMap::new();
    properties.insert("geometry_id".into(), text());
    properties.insert("region_id".into(), text());
    properties.insert("name".into(), text());
    properties.insert(
        "geometry_revision".into(),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt32,
            domain: NumericDomain::Real,
        })),
    );
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(GEOMETRY_REGION_CLASS.owned()),
            properties,
            // `tag` is present only when the imported region carries one.
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    fact.residency = ResidencyFact::Host;
    fact
}
