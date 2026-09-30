use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry, FULLY_CONNECTED_LAYER_CLASS,
};
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn network_possible(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => object.runtime_class.as_ref().is_none_or(|class| {
            [
                standard::DL_NETWORK,
                standard::SERIES_NETWORK,
                standard::DAG_NETWORK,
            ]
            .iter()
            .any(|expected| class.is(*expected))
        }),
        _ => false,
    }
}

fn known_output_width(network: Option<&ValueFact>) -> Option<usize> {
    let ValueKindFact::Object(network) = &network?.kind else {
        return None;
    };
    let ValueKindFact::Cell(layers) = &network.properties.get("Layers")?.kind else {
        return None;
    };
    if !layers.elements_complete {
        return None;
    }
    let mut width = None;
    for layer in &layers.elements {
        let ValueKindFact::Object(layer) = &layer.kind else {
            return None;
        };
        let class = layer.runtime_class.as_ref()?;
        if class.is(FULLY_CONNECTED_LAYER_CLASS) {
            width = layer
                .properties
                .get("Weights")?
                .shape
                .known_dims()
                .and_then(|dims| dims.first().copied().flatten());
        }
    }
    width
}

fn numeric_result(input: &ValueFact, output_width: Option<usize>) -> Option<ValueFact> {
    let class = match &input.kind {
        ValueKindFact::Numeric(number) if number.domain == NumericDomain::Real => {
            match number.class {
                NumericClass::Single => NumericClass::Single,
                _ => NumericClass::Double,
            }
        }
        ValueKindFact::Logical => NumericClass::Double,
        ValueKindFact::Unknown => return Some(ValueFact::unknown(DynamicReason::RuntimeValue)),
        _ => return None,
    };
    let rows = input
        .shape
        .known_dims()
        .and_then(|dims| match dims.as_slice() {
            [_] => Some(1),
            [rows, ..] => *rows,
            [] => Some(0),
        });
    let shape = ShapeFact::from(vec![rows, output_width]);
    Some(ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    ))
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-DL-FORWARD-ARITY",
            "forward requires exactly a network and one input array; options are not supported",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|net| !network_possible(net))
    {
        diagnostics.push(argument_error(
            "RM-DL-FORWARD-NETWORK",
            "network must be a dlnetwork or supported trained network object",
            0,
        ));
    }
    let input = request.arguments.get(1);
    let (data, dlarray) = match input.map(|fact| &fact.kind) {
        Some(ValueKindFact::Object(object))
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| class.is(standard::DL_ARRAY)) =>
        {
            let data = object
                .properties
                .get(crate::DLARRAY_DATA_PROPERTY)
                .cloned()
                .or_else(|| {
                    (!object.properties_complete)
                        .then(|| ValueFact::unknown(DynamicReason::RuntimeValue))
                });
            (data, true)
        }
        _ => (input.cloned(), false),
    };
    if data
        .as_ref()
        .is_some_and(|fact| fact.shape.rank().is_some_and(|rank| rank > 2))
    {
        diagnostics.push(argument_error(
            "RM-DL-FORWARD-SHAPE",
            "forward requires two-dimensional observations-by-features input",
            1,
        ));
    }
    let unsupported_storage = data
        .as_ref()
        .is_some_and(|fact| fact.storage == StorageFact::Sparse);
    if unsupported_storage {
        diagnostics.push(argument_error(
            "RM-DL-FORWARD-STORAGE",
            "forward requires dense numeric or logical input storage",
            1,
        ));
    }
    let unsupported_resident_dlarray = dlarray
        && data.as_ref().is_some_and(|fact| {
            matches!(&fact.residency, ResidencyFact::Device { .. })
                && matches!(
                    &fact.kind,
                    ValueKindFact::Logical
                        | ValueKindFact::Numeric(NumericFact {
                            class: NumericClass::Int8
                                | NumericClass::Int16
                                | NumericClass::Int32
                                | NumericClass::Int64
                                | NumericClass::UInt8
                                | NumericClass::UInt16
                                | NumericClass::UInt32
                                | NumericClass::UInt64,
                            ..
                        })
                )
        });
    if unsupported_resident_dlarray {
        diagnostics.push(argument_error(
            "RM-DL-FORWARD-RESIDENT-CLASS",
            "provider-resident dlarray Data must have real single or double storage",
            1,
        ));
    }
    let output_width = known_output_width(request.arguments.first());
    let output = match data.as_ref().and_then(|data| {
        (!unsupported_storage && !unsupported_resident_dlarray)
            .then(|| numeric_result(data, output_width))
            .flatten()
    }) {
        Some(mut numeric) if dlarray => {
            if let Some(data) = data.as_ref() {
                numeric.residency = data.residency.clone();
            }
            let shape = numeric.shape.clone();
            let properties = BTreeMap::from([
                (crate::DLARRAY_DATA_PROPERTY.into(), numeric),
                ("Format".into(), ValueFact::scalar(ValueKindFact::String)),
                ("Labels".into(), ValueFact::scalar(ValueKindFact::String)),
            ]);
            ValueFact::proven(
                ValueKindFact::Object(ObjectFact {
                    class: None,
                    runtime_class: Some(standard::DL_ARRAY.owned()),
                    properties,
                    properties_complete: false,
                    handle_semantics: Some(false),
                }),
                shape,
                StorageFact::Opaque,
            )
        }
        Some(numeric) => numeric,
        None => {
            if input.is_some() && !unsupported_storage && !unsupported_resident_dlarray {
                diagnostics.push(argument_error(
                    "RM-DL-FORWARD-DATA",
                    "input must be a real numeric/logical array or a dlarray containing one",
                    1,
                ));
            }
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}
