use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    for (index, value) in request.arguments.iter().enumerate() {
        if definitely_invalid(value) {
            diagnostics.push(argument_error(
                "RM-CATALOG-CLOSE-TARGET",
                "close expects a host figure number, TCP receiver, selector, or admitted container",
                index,
            ));
        }
    }
    let output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut result = finish_fixed(entry, request, output, diagnostics);
    if request.arguments.is_empty()
        || request.arguments.iter().enumerate().all(|(index, value)| {
            known_graphics_target(value)
                || (matches!(
                    &value.kind,
                    ValueKindFact::String | ValueKindFact::Character
                ) && request
                    .literals
                    .literal_string_at(index)
                    .is_some_and(|text| super::CloseNetworkSelector::parse(text).is_none()))
        })
    {
        result
            .capabilities
            .0
            .insert(CapabilityRequirement::UserInterface);
    } else if request.arguments.iter().enumerate().all(|(index, value)| {
        known_tcp_target(value)
            || request
                .literals
                .literal_string_at(index)
                .and_then(super::CloseNetworkSelector::parse)
                .is_some()
    }) {
        result.capabilities.0.insert(CapabilityRequirement::Network);
    }
    result
}

fn known_tcp_target(value: &ValueFact) -> bool {
    matches!(&value.kind, ValueKindFact::Object(object)
        if object.runtime_class.as_ref().is_some_and(|class|
            class == &crate::TCPCLIENT_CLASS.owned()
                || class == &crate::TCPSERVER_CLASS.owned()))
}

fn known_graphics_target(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Numeric(fact) => {
            fact.domain == NumericDomain::Real
                && !matches!(&value.residency, ResidencyFact::Device { .. })
        }
        _ => false,
    }
}

fn definitely_invalid(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::Composite(_) => false,
        ValueKindFact::Numeric(fact) => {
            fact.domain == NumericDomain::Complex
                || matches!(&value.residency, ResidencyFact::Device { .. })
        }
        ValueKindFact::String => value.shape.is_definitely_nonscalar(),
        ValueKindFact::Character => value.shape.known_dims().is_some_and(|dims| {
            dims.first()
                .is_some_and(|rows| rows.is_some_and(|rows| rows != 1))
                || dims
                    .iter()
                    .skip(2)
                    .any(|dim| dim.is_some_and(|dim| dim != 1))
        }),
        ValueKindFact::Object(object) => object.runtime_class.as_ref().is_some_and(|class| {
            class != &crate::TCPCLIENT_CLASS.owned() && class != &crate::TCPSERVER_CLASS.owned()
        }),
        ValueKindFact::Cell(_) => false,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{DynamicReason, LiteralContext, OutputSelection, RequestedOutputCount};

    fn request(arguments: Vec<ValueFact>, count: RequestedOutputCount) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(vec![]),
            outputs: OutputSelection::new(count),
        }
    }

    #[test]
    fn output_is_host_double_and_known_invalid_or_excess_rejects() {
        let entry = &super::super::CLOSE_CATALOG_ENTRY;
        let output = infer(&request(vec![], RequestedOutputCount::One), entry);
        assert!(output.diagnostics.is_empty());
        assert!(output
            .capabilities
            .0
            .contains(&CapabilityRequirement::UserInterface));
        assert!(!output
            .capabilities
            .0
            .contains(&CapabilityRequirement::Network));
        assert!(matches!(
            output.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        ));
        assert!(!infer(
            &request(
                vec![ValueFact::scalar(ValueKindFact::Logical)],
                RequestedOutputCount::One
            ),
            entry
        )
        .diagnostics
        .is_empty());
        assert!(
            !infer(&request(vec![], RequestedOutputCount::Exactly(2)), entry)
                .diagnostics
                .is_empty()
        );
        assert!(infer(
            &request(
                vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                RequestedOutputCount::One
            ),
            entry
        )
        .diagnostics
        .is_empty());
    }

    #[test]
    fn known_tcp_selector_requires_network_without_graphics() {
        let mut query = request(
            vec![ValueFact::scalar(ValueKindFact::String)],
            RequestedOutputCount::One,
        );
        query.literals =
            LiteralContext::new(vec![runmat_types::LiteralValue::String("clients".into())]);
        let result = infer(&query, &super::super::CLOSE_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert!(result
            .capabilities
            .0
            .contains(&CapabilityRequirement::Network));
        assert!(!result
            .capabilities
            .0
            .contains(&CapabilityRequirement::UserInterface));
    }
}
