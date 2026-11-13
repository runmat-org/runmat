use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, ObjectFact, ResidencyFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

/// Unit spellings accepted by the runtime option decoder (lowercase).
const SUPPORTED_UNITS: &[&str] = &[
    "m",
    "meter",
    "meters",
    "metre",
    "metres",
    "mm",
    "millimeter",
    "millimeters",
    "millimetre",
    "millimetres",
    "in",
    "inch",
    "inches",
];

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if count == 0 || count.is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-GEOMETRY-LOAD-ARITY",
            "geometry.load requires a path followed by complete Name, Value pairs",
            count,
        ));
    }
    let mut units_seen = false;
    for index in (1..count).step_by(2) {
        let option = &request.arguments[index];
        if !crate::fea_contract::constructor_inference::host_text(option) {
            diagnostics.push(argument_error(
                "RM-CATALOG-GEOMETRY-LOAD-OPTION",
                "geometry.load option names must be scalar text",
                index,
            ));
            continue;
        }
        if let Some(name) = request.literals.literal_string_at(index) {
            let name = name
                .chars()
                .filter(char::is_ascii_alphanumeric)
                .collect::<String>();
            if name != "units" || units_seen {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GEOMETRY-LOAD-OPTION",
                    if name == "units" {
                        "geometry.load accepts Units at most once"
                    } else {
                        "geometry.load accepts only the Units option"
                    },
                    index,
                ));
            }
            units_seen |= name == "units";
        }
        if let Some(value) = request.arguments.get(index + 1) {
            if !crate::fea_contract::constructor_inference::host_text(value) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GEOMETRY-LOAD-UNITS",
                    "geometry.load Units must be text",
                    index + 1,
                ));
            } else if let Some(unit) = request.literals.literal_string_at(index + 1) {
                if !SUPPORTED_UNITS.contains(&unit.as_str()) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-GEOMETRY-LOAD-UNITS",
                        "geometry.load Units must be m, mm, or in",
                        index + 1,
                    ));
                }
            }
        }
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
