use futures::executor::block_on;
use runmat_accelerate_api::{
    allocation::AllocationIdentity, GpuHandleProvenance, GpuTensorStorage, HostTensorView,
    NumericElementType,
};
use runmat_builtins::{DeepLearningAxisLabel, DeepLearningLayerKind};
use runmat_value::{CellArray, ComplexTensor, IntegerStorage, NumericDType, Tensor, Value};

use super::*;
use crate::builtins::deep_learning::model::execution::{NetworkCaller, NetworkOutputLayout};
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
use crate::builtins::deep_learning::{layers::*, model::dlnetwork_builtin, object};

fn network() -> crate::execution::RootedValue {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, -1.0], vec![1, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, dense], 2, 1).unwrap(),
    )]))
    .unwrap()
}

fn network_object(owner: &crate::execution::RootedValue) -> &runmat_value::ObjectInstance {
    let Value::Object(network) = owner.value() else {
        panic!("network object expected")
    };
    network
}

#[test]
fn admitted_host_projection_retains_actual_input_class_and_canonical_resources() {
    let network_owner = network();
    let network = network_object(&network_owner);
    let input = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::I32(vec![1, 2, 3, 4]), vec![2, 2]).unwrap(),
                ),
            ),
            ("Format", Value::String("CB".into())),
            ("Labels", Value::String("CB".into())),
        ],
    )
    .expect("fixture object");
    let plan = HostNetworkPlan::admit(network, &input, "predict").unwrap();
    let target = AxisPlan::new(
        runmat_builtins::DeepLearningAxisLayout::from_axes(&[
            DeepLearningAxisLabel::Batch,
            DeepLearningAxisLabel::Channel,
        ])
        .unwrap(),
        &[2, 1],
    )
    .unwrap();
    let work = NetworkWorkload::from_admitted(&plan, &target).unwrap();
    assert_eq!(work.input.kind, PayloadKind::Real(NumericDType::I32));
    assert_eq!(work.input.logical_bytes, 16);
    assert_eq!(work.input.origin, PayloadOrigin::Host);
    assert_eq!(work.execution_precision, NumericDType::F64);
    assert_eq!(work.steps.len(), 2);
    assert_eq!(work.steps[0].kind, DeepLearningLayerKind::FeatureInput);
    assert_eq!(work.steps[1].kind, DeepLearningLayerKind::FullyConnected);
    assert_eq!(work.parameters.len(), 2);
    assert_eq!(work.parameters[0].payload.logical_bytes, 16);
    assert_eq!(work.parameters[1].payload.logical_bytes, 8);
    assert_eq!(work.source_output_axes.shape(), &[1, 2]);
    assert_eq!(work.target_output_axes.shape(), &[2, 1]);
    assert_eq!(work.output_elements, 2);
    assert_eq!(work.output_bytes_if_real, 16);
}

#[test]
fn projection_rejects_a_target_shape_not_derived_from_admitted_axes() {
    let network_owner = network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    let plan = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let wrong = AxisPlan::new(plan.output_axes().layout().clone(), &[99, 1]).unwrap();
    let error = NetworkWorkload::from_admitted(&plan, &wrong)
        .err()
        .expect("a caller must not substitute output shape");
    assert!(error.message.contains("presented output geometry"));
}

#[test]
fn projection_binds_the_original_input_not_a_same_shape_substitution() {
    let network_owner = network();
    let network = network_object(&network_owner);
    let original = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I32(vec![1, 2, 3, 4]), vec![2, 2]).unwrap(),
    );
    let replacement = Value::Tensor(Tensor::from_f32(vec![1.0; 4], vec![2, 2]).unwrap());
    let admitted = HostNetworkPlan::admit(network, &original, "predict").unwrap();
    let alternative = HostNetworkPlan::admit(network, &replacement, "predict").unwrap();
    let admitted_work = NetworkWorkload::from_admitted(&admitted, admitted.output_axes()).unwrap();
    let alternative_work =
        NetworkWorkload::from_admitted(&alternative, alternative.output_axes()).unwrap();
    assert_eq!(
        admitted_work.input.kind,
        PayloadKind::Real(NumericDType::I32)
    );
    assert_eq!(
        alternative_work.input.kind,
        PayloadKind::Real(NumericDType::F32)
    );
    assert_eq!(admitted_work.input.shape, alternative_work.input.shape);
}

#[test]
fn selected_resources_are_exact_plan_rows_not_same_shaped_substitutes() {
    let network_owner = network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    let admitted = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let alternative = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let selected = [&admitted.parameter_views()[0]];
    let work =
        NetworkWorkload::from_selected(&admitted, admitted.output_axes(), &selected, &[]).unwrap();
    assert_eq!(work.parameters.len(), 1);
    assert!(work.state.is_empty());
    let foreign = [&alternative.parameter_views()[0]];
    let error = NetworkWorkload::from_selected(&admitted, admitted.output_axes(), &foreign, &[])
        .err()
        .expect("same-shaped foreign row must not authenticate");
    assert!(error.message.contains("not bound to the admitted plan"));
}

#[test]
fn provider_projection_preserves_explicit_intent_owner_and_physical_allocation() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let handle = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[2, 2],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let substitute = provider
            .upload(&HostTensorView {
                data: &[5.0, 6.0, 7.0, 8.0],
                shape: &[2, 2],
            })
            .unwrap();
        assert_ne!(
            handle.allocation_identity(),
            substitute.allocation_identity()
        );
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(handle.clone())),
                ("Format", Value::String("CB".into())),
                ("Labels", Value::String("CB".into())),
            ],
        )
        .expect("fixture object");
        let network_owner = network();
        let network = network_object(&network_owner);
        let plan = HostNetworkPlan::admit(network, &input, "forward").unwrap();
        let work = NetworkWorkload::from_admitted(&plan, plan.output_axes()).unwrap();
        assert_eq!(work.input.logical_bytes, 32);
        assert_eq!(work.input.kind, PayloadKind::Real(NumericDType::F64));
        assert_eq!(work.input.shape, vec![2, 2]);
        assert_eq!(
            work.input.origin,
            PayloadOrigin::Provider {
                allocation: handle.allocation_identity(),
                provenance: Some(GpuHandleProvenance::Explicit),
                layout: handle.view().layout(),
                element_type: handle.physical().element_type(),
                storage: handle.physical().storage(),
                requested_bytes: handle.physical().requested_bytes(),
            }
        );
        assert_ne!(
            work.input.origin,
            PayloadOrigin::Provider {
                allocation: substitute.allocation_identity(),
                provenance: runmat_accelerate_api::handle_provenance(&substitute),
                layout: substitute.view().layout(),
                element_type: substitute.physical().element_type(),
                storage: substitute.physical().storage(),
                requested_bytes: substitute.physical().requested_bytes(),
            }
        );
    });
}

#[test]
fn byte_geometry_is_checked_and_complex_lanes_are_counted() {
    assert_eq!(logical_bytes(&[2, 3], 4, 2), Some(48));
    assert_eq!(logical_bytes(&[usize::MAX, usize::MAX], 8, 1), None);
    assert_eq!(logical_bytes(&[0, 3], 8, 1), Some(0));
    let complex =
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, -1.0); 6], vec![2, 3]).unwrap());
    let facts = PayloadFacts::from_admitted(&complex, "forward").unwrap();
    assert_eq!(facts.kind, PayloadKind::Complex(NumericDType::F64));
    assert_eq!(facts.logical_bytes, 96);
}

fn invocation() -> NetworkInvocationFacts {
    NetworkInvocationFacts {
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: Some(1),
        record_tape: false,
    }
}

fn resident_origin(identity: AllocationIdentity) -> PayloadOrigin {
    PayloadOrigin::Provider {
        allocation: identity,
        provenance: Some(GpuHandleProvenance::Automatic),
        layout: GpuTensorLayout::Contiguous,
        element_type: NumericElementType::F64,
        storage: GpuTensorStorage::Real,
        requested_bytes: 32,
    }
}

#[test]
fn semantic_timing_digest_retains_alias_pattern_without_allocation_numbers() {
    let network_owner = network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    let plan = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let mut work = NetworkWorkload::from_admitted(&plan, plan.output_axes()).unwrap();
    let first = AllocationIdentity::new(1, 10, 1);
    let second = AllocationIdentity::new(1, 20, 1);
    work.input.origin = resident_origin(first);
    work.parameters[0].payload.origin = resident_origin(first);
    work.parameters[1].payload.origin = resident_origin(second);
    let aliased = work.semantic_timing_digest(invocation());

    work.input.origin = resident_origin(AllocationIdentity::new(7, 700, 3));
    work.parameters[0].payload.origin = resident_origin(AllocationIdentity::new(7, 700, 3));
    work.parameters[1].payload.origin = resident_origin(AllocationIdentity::new(7, 800, 3));
    assert_eq!(work.semantic_timing_digest(invocation()), aliased);

    work.parameters[0].payload.origin = resident_origin(AllocationIdentity::new(7, 900, 3));
    assert_ne!(work.semantic_timing_digest(invocation()), aliased);
}

#[test]
fn semantic_timing_digest_distinguishes_invocation_and_admitted_resource_facts() {
    let network_owner = network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    let plan = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let mut work = NetworkWorkload::from_admitted(&plan, plan.output_axes()).unwrap();
    let baseline = work.semantic_timing_digest(invocation());
    assert_ne!(
        baseline,
        work.semantic_timing_digest(NetworkInvocationFacts {
            requested_outputs: Some(2),
            ..invocation()
        })
    );
    assert_ne!(
        baseline,
        work.semantic_timing_digest(NetworkInvocationFacts {
            requested_outputs: None,
            ..invocation()
        })
    );
    assert_ne!(
        baseline,
        work.semantic_timing_digest(NetworkInvocationFacts {
            record_tape: true,
            ..invocation()
        })
    );
    assert_ne!(
        baseline,
        work.semantic_timing_digest(NetworkInvocationFacts {
            caller: NetworkCaller::Predict,
            ..invocation()
        })
    );
    work.parameters[0].payload.origin = resident_origin(AllocationIdentity::new(1, 10, 1));
    assert_ne!(baseline, work.semantic_timing_digest(invocation()));
    work.parameters[0].payload.kind = PayloadKind::Real(NumericDType::F32);
    assert_ne!(baseline, work.semantic_timing_digest(invocation()));
}

#[test]
fn semantic_timing_digest_uses_typed_role_not_user_layer_label() {
    let network_owner = network();
    let network = network_object(&network_owner);
    let input = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    let plan = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let mut work = NetworkWorkload::from_admitted(&plan, plan.output_axes()).unwrap();
    let baseline = work.semantic_timing_digest(invocation());
    let renamed = crate::builtins::deep_learning::model::parameters::ParameterKey {
        layer_index: work.parameters[0].key.layer_index,
        layer_name: "user-chosen-display-name".into(),
        role: work.parameters[0].key.role,
    };
    work.parameters[0].key = &renamed;
    assert_eq!(baseline, work.semantic_timing_digest(invocation()));
}
