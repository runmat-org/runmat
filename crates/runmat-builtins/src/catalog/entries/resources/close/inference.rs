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
    let mut routes = if request.arguments.is_empty() {
        (true, false)
    } else {
        (false, false)
    };
    for (index, value) in request.arguments.iter().enumerate() {
        let possible = possible_routes(value, request.literals.literal_string_at(index).as_deref());
        routes.0 |= possible.0;
        routes.1 |= possible.1;
    }
    if routes.0 {
        result
            .capabilities
            .0
            .insert(CapabilityRequirement::UserInterface);
    }
    if routes.1 {
        result.capabilities.0.insert(CapabilityRequirement::Network);
    }
    if routes == (false, true) {
        // A TCP-only route cannot invoke figure deletion callbacks. Retain
        // the conservative callback effects when graphics remains possible.
        for effect in [
            EffectKind::UserInterface,
            EffectKind::HostCallback,
            EffectKind::Unknown,
        ] {
            result.effects.0.remove(&effect);
        }
    }
    result
}

/// An unknown target may select either route. A fully known cell carries
/// position-preserving element facts; an incomplete one also needs its common
/// element fact so a future member cannot silently require another service.
fn possible_routes(value: &ValueFact, literal: Option<&str>) -> (bool, bool) {
    if known_tcp_target(value) {
        return (false, true);
    }
    if known_graphics_target(value) {
        return (true, false);
    }
    match &value.kind {
        ValueKindFact::String | ValueKindFact::Character => match literal {
            Some(text) if super::CloseNetworkSelector::parse(text).is_some() => (false, true),
            Some(_) => (true, false),
            None => (true, true),
        },
        ValueKindFact::Cell(cell) => {
            let mut routes = (false, false);
            for member in &cell.elements {
                let possible = possible_routes(member, None);
                routes.0 |= possible.0;
                routes.1 |= possible.1;
            }
            if !cell.elements_complete {
                let possible = possible_routes(&cell.element, None);
                routes.0 |= possible.0;
                routes.1 |= possible.1;
            }
            routes
        }
        ValueKindFact::Unknown | ValueKindFact::Composite(_) => (true, true),
        // Graphics handle parsing accepts numeric/text targets, not objects.
        // Without a known runtime class, however, this may still be a sealed
        // TCP client or server at execution time.
        ValueKindFact::Object(object) if object.runtime_class.is_none() => (false, true),
        _ => (false, false),
    }
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
        let unknown = infer(
            &request(
                vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                RequestedOutputCount::One,
            ),
            entry,
        );
        assert!(unknown
            .capabilities
            .0
            .contains(&CapabilityRequirement::UserInterface));
        assert!(unknown
            .capabilities
            .0
            .contains(&CapabilityRequirement::Network));
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

    #[test]
    fn known_cell_members_and_incomplete_element_require_possible_services() {
        let entry = &super::super::CLOSE_CATALOG_ENTRY;
        let endpoint = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(crate::TCPCLIENT_CLASS.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(true),
        }));
        let cell = ValueFact::scalar(ValueKindFact::Cell(CellFact {
            element: Box::new(endpoint.clone()),
            elements: vec![endpoint],
            elements_complete: true,
        }));
        let result = infer(&request(vec![cell], RequestedOutputCount::One), entry);
        assert!(result
            .capabilities
            .0
            .contains(&CapabilityRequirement::Network));
        assert!(!result
            .capabilities
            .0
            .contains(&CapabilityRequirement::UserInterface));
        let incomplete = ValueFact::scalar(ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: vec![],
            elements_complete: false,
        }));
        let result = infer(&request(vec![incomplete], RequestedOutputCount::One), entry);
        assert!(result
            .capabilities
            .0
            .contains(&CapabilityRequirement::Network));
        assert!(result
            .capabilities
            .0
            .contains(&CapabilityRequirement::UserInterface));
    }

    #[test]
    fn unknown_object_class_may_require_network_but_not_graphics() {
        let object = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: None,
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: None,
        }));
        let result = infer(
            &request(vec![object], RequestedOutputCount::One),
            &super::super::CLOSE_CATALOG_ENTRY,
        );
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
