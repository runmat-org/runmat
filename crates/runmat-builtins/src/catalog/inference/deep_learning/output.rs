use super::{
    caller::{NetworkInferenceCaller, NetworkInputIssue},
    input, network, shape,
};
use network::ComplexLayerSupport;
use runmat_types::{
    standard, ContiguityFact, DynamicReason, InferenceDiagnostic, LayoutFact, NumericClass,
    NumericDomain, NumericFact, ObjectFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact, ViewFact,
};
use std::collections::BTreeMap;

pub(in crate::catalog) struct NetworkOutputFacts {
    pub output: ValueFact,
    issues: Vec<NetworkInputIssue>,
    caller: NetworkInferenceCaller,
}

impl NetworkOutputFacts {
    pub fn diagnostics(&self) -> Vec<InferenceDiagnostic> {
        self.issues
            .iter()
            .map(|issue| issue.diagnostic(self.caller))
            .collect()
    }
}

fn numeric_result(
    input: &ValueFact,
    output_shape: ShapeFact,
    complex_layers: ComplexLayerSupport,
) -> Option<ValueFact> {
    let (class, domain, residency) = match &input.kind {
        ValueKindFact::Numeric(number) if number.domain == NumericDomain::Real => (
            if number.class == NumericClass::Single {
                NumericClass::Single
            } else {
                NumericClass::Double
            },
            NumericDomain::Real,
            ResidencyFact::Unknown,
        ),
        ValueKindFact::Numeric(number)
            if number.domain == NumericDomain::Complex
                && matches!(number.class, NumericClass::Single | NumericClass::Double) =>
        {
            match complex_layers {
                ComplexLayerSupport::HostOnly => {
                    (number.class, NumericDomain::Complex, ResidencyFact::Host)
                }
                ComplexLayerSupport::Unknown => {
                    return Some(ValueFact::unknown(DynamicReason::RuntimeValue))
                }
                ComplexLayerSupport::Unsupported => return None,
            }
        }
        ValueKindFact::Logical => (
            NumericClass::Double,
            NumericDomain::Real,
            ResidencyFact::Unknown,
        ),
        ValueKindFact::Unknown => return Some(ValueFact::unknown(DynamicReason::RuntimeValue)),
        _ => return None,
    };
    let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
    result.kind = ValueKindFact::Numeric(NumericFact { class, domain });
    result.shape = output_shape;
    result.storage = StorageFact::Dense;
    // Real host-origin data can still use resident learnables. Only the
    // admitted complex input/pooling/output topology has a host-only route.
    result.residency = residency;
    result.view = ViewFact::Unknown;
    result.layout = LayoutFact::Unknown;
    result.contiguity = ContiguityFact::Unknown;
    Some(result)
}

fn dlarray_result(data: ValueFact) -> ValueFact {
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

fn predict_shape(shape: &ShapeFact) -> ShapeFact {
    // Predict may reorder observations or unformatted axes. Only rank is
    // retained from a proven network geometry.
    shape
        .rank()
        .map_or(ShapeFact::Unknown, |rank| ShapeFact::Ranked { rank })
}

pub(in crate::catalog) fn infer_network_output(
    network: Option<&ValueFact>,
    input: Option<&ValueFact>,
    caller: NetworkInferenceCaller,
) -> NetworkOutputFacts {
    let mut issues = Vec::new();
    let input_facts = input::project(input);
    let data = input_facts.data.as_ref();
    if input_facts.unsupported_storage {
        issues.push(NetworkInputIssue::UnsupportedStorage);
    }
    if input_facts.unsupported_resident_data {
        issues.push(NetworkInputIssue::UnsupportedResidentData);
    }
    let layers = network::known_layers(network);
    let complex_layers = network::complex_layer_support(layers.as_deref());
    let unsupported_complex_layers = data.is_some_and(|fact| {
        matches!(
            &fact.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single | NumericClass::Double,
                domain: NumericDomain::Complex,
            })
        ) && complex_layers == ComplexLayerSupport::Unsupported
    });
    if unsupported_complex_layers && !input_facts.unsupported_resident_data {
        issues.push(NetworkInputIssue::UnsupportedComplexLayers);
    }
    if !input_facts.dlarray
        && data.is_some_and(|data| shape::invalid_raw_geometry(data, layers.as_deref()))
    {
        issues.push(NetworkInputIssue::InvalidRawGeometry);
    }
    let shape = data.map_or(ShapeFact::Unknown, |data| {
        shape::output_shape(data, input_facts.dlarray, layers.as_deref())
    });
    let mut output = match data.and_then(|data| {
        (!input_facts.unsupported_storage && !input_facts.unsupported_resident_data)
            .then(|| numeric_result(data, shape, complex_layers))
            .flatten()
    }) {
        Some(numeric) if input_facts.dlarray => dlarray_result(numeric),
        Some(numeric) => numeric,
        None => {
            if input.is_some()
                && !input_facts.unsupported_storage
                && !input_facts.unsupported_resident_data
                && !unsupported_complex_layers
            {
                issues.push(NetworkInputIssue::InvalidData);
            }
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
    };
    if caller == NetworkInferenceCaller::Predict {
        match &mut output.kind {
            ValueKindFact::Object(object)
                if object
                    .runtime_class
                    .as_ref()
                    .is_some_and(|class| class.is(standard::DL_ARRAY)) =>
            {
                if let Some(data) = object.properties.get_mut(crate::DLARRAY_DATA_PROPERTY) {
                    data.shape = predict_shape(&data.shape);
                    output.shape = data.shape.clone();
                }
            }
            _ => {
                output.shape = predict_shape(&output.shape);
            }
        }
    }
    NetworkOutputFacts {
        output,
        issues,
        caller,
    }
}
