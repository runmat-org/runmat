use futures::executor::block_on;
use runmat_accelerate_api::{AccelProvider, GpuHandleProvenance, HostTensorView};
use runmat_builtins::{DeepLearningAxisLabel, DeepLearningLayerKind};
use runmat_value::{CellArray, ComplexTensor, IntegerStorage, NumericDType, Tensor, Value};

use super::*;
use crate::builtins::deep_learning::{layers::*, model::dlnetwork_builtin, object};

fn network() -> runmat_value::ObjectInstance {
    let input = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, -1.0], vec![1, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let Value::Object(network) = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, dense], 2, 1).unwrap(),
    )]))
    .unwrap() else {
        panic!("network object expected")
    };
    network
}

#[test]
fn admitted_host_projection_retains_actual_input_class_and_canonical_resources() {
    let network = network();
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
    );
    let plan = HostNetworkPlan::admit(&network, &input, "predict").unwrap();
    let target = AxisPlan::new(
        runmat_builtins::DeepLearningAxisLayout::from_axes(&[
            DeepLearningAxisLabel::Batch,
            DeepLearningAxisLabel::Channel,
        ])
        .unwrap(),
        &[2, 1],
    )
    .unwrap();
    let work = NetworkWorkload::from_admitted(&plan, &input, &target).unwrap();
    assert_eq!(work.input.class, PayloadClass::Numeric(NumericDType::I32));
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
    assert_eq!(work.scratch_bytes, None);
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
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(handle.clone())),
                ("Format", Value::String("CB".into())),
                ("Labels", Value::String("CB".into())),
            ],
        );
        let network = network();
        let plan = HostNetworkPlan::admit(&network, &input, "forward").unwrap();
        let work = NetworkWorkload::from_admitted(&plan, &input, plan.output_axes()).unwrap();
        assert_eq!(work.input.logical_bytes, 32);
        assert_eq!(work.input.class, PayloadClass::Numeric(NumericDType::F64));
        assert_eq!(work.input.shape, vec![2, 2]);
        assert_eq!(
            work.input.origin,
            PayloadOrigin::Provider {
                allocation: handle.allocation_identity(),
                provenance: Some(GpuHandleProvenance::Explicit),
                layout: handle.view().layout(),
                requested_bytes: handle.physical().requested_bytes(),
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
    assert_eq!(facts.storage, PayloadStorage::Complex);
    assert_eq!(facts.logical_bytes, 96);
}
