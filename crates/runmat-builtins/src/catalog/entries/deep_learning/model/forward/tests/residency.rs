use super::*;
use crate::catalog::inference::deep_learning::{infer_network_output, NetworkInferenceCaller};

mod fixtures;
use fixtures::{complex_data, complex_network, public_infer, wrapped};

#[test]
fn known_complex_host_route_keeps_residency_through_wrapping_and_prediction() {
    let net = complex_network();
    for caller in [
        NetworkInferenceCaller::Forward,
        NetworkInferenceCaller::Predict,
    ] {
        for class in [NumericClass::Single, NumericClass::Double] {
            let input = complex_data(class);
            let raw = public_infer(caller, net.clone(), input.clone());
            assert!(raw.diagnostics.is_empty());
            assert_eq!(raw.outputs[0].residency, ResidencyFact::Host);
            assert_eq!(raw.outputs[0].numeric(), input.numeric());

            let input = wrapped(input);
            let result = public_infer(caller, net.clone(), input);
            assert!(result.diagnostics.is_empty());
            assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
            let ValueKindFact::Object(wrapper) = &result.outputs[0].kind else {
                panic!("expected dlarray output")
            };
            let data = &wrapper.properties[crate::DLARRAY_DATA_PROPERTY];
            assert_eq!(data.residency, ResidencyFact::Host);
            assert_eq!(
                data.numeric(),
                Some(NumericFact {
                    class,
                    domain: NumericDomain::Complex
                })
            );
            assert_eq!(data.shape, ShapeFact::Unknown);
        }
    }
}

#[test]
fn unknown_complex_topology_does_not_prove_host_residency() {
    let net = network();
    let data = complex_data(NumericClass::Single);
    for input in [data.clone(), wrapped(data)] {
        for caller in [
            NetworkInferenceCaller::Forward,
            NetworkInferenceCaller::Predict,
        ] {
            let result = infer_network_output(Some(&net), Some(&input), caller);
            assert!(result.diagnostics().is_empty());
            match &result.output.kind {
                ValueKindFact::Object(wrapper) => assert_eq!(
                    wrapper.properties[crate::DLARRAY_DATA_PROPERTY].residency,
                    ResidencyFact::Unknown,
                ),
                _ => assert_eq!(result.output.residency, ResidencyFact::Unknown),
            }
        }
    }
}

#[test]
fn known_device_complex_rejects_without_publishing_host_output_facts() {
    let net = complex_network();
    let mut data = complex_data(NumericClass::Single);
    data.residency = ResidencyFact::Device { provider: None };
    for input in [data.clone(), wrapped(data)] {
        for caller in [
            NetworkInferenceCaller::Forward,
            NetworkInferenceCaller::Predict,
        ] {
            let result = public_infer(caller, net.clone(), input.clone());
            let diagnostics = &result.diagnostics;
            assert_eq!(diagnostics.len(), 1);
            assert_eq!(diagnostics[0].argument, Some(1));
            let expected_code = match caller {
                NetworkInferenceCaller::Forward => "RM-DL-FORWARD-RESIDENT-CLASS",
                NetworkInferenceCaller::Predict => "RM-DL-PREDICT-RESIDENT-CLASS",
            };
            assert_eq!(diagnostics[0].code, expected_code);
            assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
            assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
        }
    }
}

#[test]
fn unknown_device_class_is_not_rejected_as_a_known_nonfloating_class() {
    let net = complex_network();
    let mut data = ValueFact::unknown(DynamicReason::RuntimeValue);
    data.residency = ResidencyFact::Device { provider: None };
    for input in [data.clone(), wrapped(data)] {
        let result =
            infer_network_output(Some(&net), Some(&input), NetworkInferenceCaller::Forward);
        assert!(result.diagnostics().is_empty());
    }
}
