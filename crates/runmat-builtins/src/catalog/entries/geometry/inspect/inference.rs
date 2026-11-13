use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, NumericClass,
    NumericDomain, NumericFact, ObjectFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GEOMETRY-INSPECT-ARITY",
            "geometry.inspect requires one path",
            0,
        ));
    }
    if request.arguments.first().is_some_and(|path| {
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
        invalid_shape
            || !matches!(
                path.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            )
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-GEOMETRY-INSPECT-PATH",
            "path must be scalar text",
            0,
        ));
    }
    let mut properties = BTreeMap::new();
    properties.insert(
        "format".into(),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
    );
    properties.insert(
        "byte_count".into(),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    );
    let mut output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::GEOMETRY_INSPECT_RESULT_CLASS.owned()),
            properties,
            properties_complete: true,
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
