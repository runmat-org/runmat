use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, CertaintyFact, ContiguityFact, DynamicReason, LayoutFact,
    NumericClass, NumericDomain, NumericFact, ObjectFact, ResidencyFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact, ViewFact,
};
use std::collections::BTreeMap;

mod network;
mod shape;

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

fn numeric_result(input: &ValueFact, output_shape: ShapeFact) -> Option<ValueFact> {
    let class = match &input.kind {
        ValueKindFact::Numeric(number) if number.domain == NumericDomain::Real => {
            if number.class == NumericClass::Single {
                NumericClass::Single
            } else {
                NumericClass::Double
            }
        }
        ValueKindFact::Logical => NumericClass::Double,
        ValueKindFact::Unknown => return Some(ValueFact::unknown(DynamicReason::RuntimeValue)),
        _ => return None,
    };
    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    result.kind = ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    });
    result.shape = output_shape;
    result.storage = StorageFact::Dense;
    result.residency = if input.residency == ResidencyFact::Host {
        ResidencyFact::Host
    } else {
        ResidencyFact::Unknown
    };
    result.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    result.view = ViewFact::Unknown;
    result.layout = LayoutFact::Unknown;
    result.contiguity = ContiguityFact::Unknown;
    Some(result)
}

fn dlarray_result(mut data: ValueFact) -> ValueFact {
    // A fact cannot promise that an optional provider route will be selected.
    data.residency = ResidencyFact::Unknown;
    let shape = data.shape.clone();
    let mut wrapper = ValueFact::unknown(DynamicReason::RuntimeValue);
    wrapper.kind = ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(standard::DL_ARRAY.owned()),
        properties: BTreeMap::from([
            (crate::DLARRAY_DATA_PROPERTY.into(), data),
            ("Format".into(), ValueFact::scalar(ValueKindFact::String)),
            ("Labels".into(), ValueFact::scalar(ValueKindFact::String)),
        ]),
        properties_complete: false,
        handle_semantics: Some(false),
    });
    wrapper.shape = shape;
    wrapper.storage = StorageFact::Opaque;
    wrapper.residency = ResidencyFact::Host;
    wrapper
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
                && !matches!(
                    &fact.kind,
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Single | NumericClass::Double,
                        domain: NumericDomain::Real
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
    let layers = network::known_layers(request.arguments.first());
    if !dlarray
        && data
            .as_ref()
            .is_some_and(|data| shape::invalid_raw_geometry(data, layers.as_deref()))
    {
        diagnostics.push(argument_error(
            "RM-DL-FORWARD-SHAPE",
            "input rank is incompatible with the network's input layer",
            1,
        ));
    }
    let shape = data.as_ref().map_or(ShapeFact::Unknown, |data| {
        shape::output_shape(data, dlarray, layers.as_deref())
    });
    let output = match data.as_ref().and_then(|data| {
        (!unsupported_storage && !unsupported_resident_dlarray)
            .then(|| numeric_result(data, shape))
            .flatten()
    }) {
        Some(numeric) if dlarray => dlarray_result(numeric),
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
    let state_rows = layers.as_ref().and_then(|layers| {
        layers
            .iter()
            .all(|(kind, _)| {
                !matches!(
                    kind,
                    crate::DeepLearningLayerKind::Lstm | crate::DeepLearningLayerKind::Bilstm
                )
            })
            .then_some(0)
    });
    let state = super::super::dlnetwork::state_rows_table_fact(state_rows);
    finish_fixed_outputs(entry, request, vec![output, state], diagnostics)
}
