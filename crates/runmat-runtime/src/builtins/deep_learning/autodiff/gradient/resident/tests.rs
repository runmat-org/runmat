use super::*;
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
use crate::builtins::deep_learning::{layers::*, model, object};
use futures::executor::block_on;
use runmat_accelerate_api::{
    handle_precision, handle_provenance, GpuHandleProvenance, HostNumericDataView,
    HostNumericTensorView, HostTensorView, ProviderPrecision,
};
use runmat_value::CellArray;

fn dense_network() -> crate::execution::RootedValue {
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
    block_on(model::dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, dense], 2, 1).unwrap(),
    )]))
    .unwrap()
}

fn scalar_network() -> crate::execution::RootedValue {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![3.0], vec![1, 1]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    block_on(model::dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, dense], 2, 1).unwrap(),
    )]))
    .unwrap()
}

#[test]
fn public_single_resident_gradient_retains_native_precision() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[2.0]),
                shape: &[1, 1],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(handle)),
                ("Format", Value::String("BC".into())),
                ("Labels", Value::String("BC".into())),
            ],
        )
        .expect("fixture object");
        let feature = block_on(feature_input_layer_builtin(
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let dense = block_on(fully_connected_layer_builtin(
            None,
            Value::Num(1.0),
            vec![
                Value::String("Weights".into()),
                Value::Tensor(Tensor::from_f32(vec![3.0], vec![1, 1]).unwrap()),
                Value::String("Bias".into()),
                Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
            ],
        ))
        .unwrap();
        let network = block_on(model::dlnetwork_builtin(vec![Value::Cell(
            CellArray::new(vec![feature, dense], 2, 1).unwrap(),
        )]))
        .unwrap();
        let tape = Rc::new(RefCell::new(super::super::super::tape::Tape::new()));
        let _activation = super::super::super::tape::TapeActivation::enter(tape);
        let input = super::super::super::tape::annotate_tree(input).unwrap();
        let network = super::super::super::tape::annotate_tree(network.value().clone()).unwrap();
        let loss = block_on(model::forward_builtin(
            network,
            input.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        provider.reset_telemetry();
        let gradient = block_on(super::super::dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![input]),
        ))
        .unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let Value::Object(gradient) = gradient else {
            panic!("dlarray expected")
        };
        let Some(Value::GpuTensor(handle)) = gradient.properties().get("Data") else {
            panic!("resident gradient expected")
        };
        assert_eq!(handle_precision(handle), Some(ProviderPrecision::F32));
        assert_eq!(
            handle_provenance(handle),
            Some(GpuHandleProvenance::Explicit)
        );
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle.clone()))
                .unwrap();
        assert_eq!(gathered.materialize_f64(), vec![3.0]);
    });
}

#[test]
fn resident_reverse_traverses_two_networks_and_accumulates_reused_parameters() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let raw = provider
            .upload(&HostTensorView {
                data: &[2.0],
                shape: &[1, 1],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(raw)),
                ("Format", Value::String("BC".into())),
                ("Labels", Value::String("BC".into())),
            ],
        )
        .expect("fixture object");
        let tape = Rc::new(RefCell::new(super::super::super::tape::Tape::new()));
        let _activation = super::super::super::tape::TapeActivation::enter(tape);
        let input = super::super::super::tape::annotate_tree(input).unwrap();
        let network_owner = scalar_network();
        let network =
            super::super::super::tape::annotate_tree(network_owner.value().clone()).unwrap();
        let first = block_on(model::forward_builtin(
            network.clone(),
            input.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let second = block_on(model::forward_builtin(
            network.clone(),
            first,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let _requested =
            crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        provider.reset_telemetry();
        let gradients = block_on(super::super::dlgradient_direct(
            second,
            crate::builtins::common::test_support::builtin_arguments(vec![input, network]),
        ))
        .unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let Value::OutputList(values) = gradients else {
            panic!("two gradients expected")
        };
        let Value::Object(dx) = &values[0] else {
            panic!("dlarray dx expected")
        };
        let Some(Value::GpuTensor(dx)) = dx.properties().get("Data") else {
            panic!("resident dx expected")
        };
        let dx =
            crate::builtins::common::test_support::gather(Value::GpuTensor(dx.clone())).unwrap();
        assert_eq!(dx.materialize_f64(), vec![9.0]);
        let Value::Object(table) = &values[1] else {
            panic!("network gradient table expected")
        };
        let variables = crate::builtins::table::table_variables(table).unwrap();
        let Some(Value::StringArray(roles)) = variables.fields.get("Parameter") else {
            panic!("canonical parameter roles expected")
        };
        assert_eq!(
            roles
                .data()
                .iter()
                .map(|value| value.as_str().expect("present parameter role"))
                .collect::<Vec<_>>(),
            ["Weights", "Bias"]
        );
        let Some(Value::Cell(parameters)) = variables.fields.get("Value") else {
            panic!("gradient values expected")
        };
        assert_eq!(parameters.data().len(), 2);
        let expected = [12.5, 4.0];
        for (value, expected) in parameters.data().iter().zip(expected) {
            let Value::Object(parameter) = value else {
                panic!("gradient dlarray expected")
            };
            let Some(Value::GpuTensor(handle)) = parameter.properties().get("Data") else {
                panic!("resident parameter gradient expected")
            };
            let actual =
                crate::builtins::common::test_support::gather(Value::GpuTensor(handle.clone()))
                    .unwrap();
            assert_eq!(actual.materialize_f64(), vec![expected]);
        }
    });
}

#[test]
fn truly_unconnected_resident_target_has_zero_gradient() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tape = Rc::new(RefCell::new(super::super::super::tape::Tape::new()));
        let _activation = super::super::super::tape::TapeActivation::enter(tape);
        let wrapped = |value: f64| {
            let handle = provider
                .upload(&HostTensorView {
                    data: &[value],
                    shape: &[1, 1],
                })
                .unwrap()
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap();
            super::super::super::tape::annotate_tree(
                object(
                    "dlarray",
                    vec![
                        ("Data", Value::GpuTensor(handle)),
                        ("Format", Value::String("BC".into())),
                        ("Labels", Value::String("BC".into())),
                    ],
                )
                .expect("fixture object"),
            )
            .unwrap()
        };
        let connected = wrapped(2.0);
        let disconnected = wrapped(7.0);
        let network_owner = scalar_network();
        let network =
            super::super::super::tape::annotate_tree(network_owner.value().clone()).unwrap();
        let loss = block_on(model::forward_builtin(
            network,
            connected,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        provider.reset_telemetry();
        let result = block_on(super::super::dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![disconnected]),
        ))
        .unwrap();
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let Value::Object(result) = result else {
            panic!("dlarray expected")
        };
        let Some(Value::GpuTensor(handle)) = result.properties().get("Data") else {
            panic!("resident zero expected")
        };
        assert_eq!(
            handle_provenance(handle),
            Some(GpuHandleProvenance::Explicit)
        );
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle.clone()))
                .unwrap();
        assert_eq!(gathered.materialize_f64(), vec![0.0]);
    });
}

#[test]
fn connected_unsupported_resident_parent_rejects_before_reverse_provider_effects() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let make_source = |value: f64| {
            let handle = provider
                .upload(&HostTensorView {
                    data: &[value],
                    shape: &[1, 1],
                })
                .unwrap()
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap();
            super::super::super::tape::annotate_tree(
                object(
                    "dlarray",
                    vec![
                        ("Data", Value::GpuTensor(handle)),
                        ("Format", Value::String("BC".into())),
                        ("Labels", Value::String("BC".into())),
                    ],
                )
                .expect("fixture object"),
            )
            .unwrap()
        };
        let tape = Rc::new(RefCell::new(super::super::super::tape::Tape::new()));
        let _activation = super::super::super::tape::TapeActivation::enter(tape.clone());
        let leaf = make_source(1.0);
        let connected = make_source(2.0);
        let leaf_id = super::super::super::dlarray_node_id(&leaf).unwrap();
        let connected_id = super::super::super::dlarray_node_id(&connected).unwrap();
        // Model an already traced resident primitive for which no resident
        // adjoint exists. Its reachable parent must not be silently zeroed.
        tape.borrow_mut().nodes[connected_id].kind = super::super::super::tape::NodeKind::Add {
            lhs: Some(leaf_id),
            rhs: None,
            lhs_shape: vec![1, 1],
            rhs_shape: vec![1, 1],
        };
        let network_owner = scalar_network();
        let network =
            super::super::super::tape::annotate_tree(network_owner.value().clone()).unwrap();
        let loss = block_on(model::forward_builtin(
            network,
            connected,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        provider.reset_telemetry();
        let before = provider.telemetry_snapshot();
        let error = block_on(super::super::dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![leaf]),
        ))
        .unwrap_err();
        assert!(error.to_string().contains("no resident adjoint"));
        assert_eq!(provider.telemetry_snapshot(), before);
    });
}

#[test]
fn public_scalar_resident_network_gradient_stays_on_provider() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let raw = provider
            .upload(&HostTensorView {
                data: &[3.0, 4.0],
                shape: &[1, 2],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(raw.clone())),
                ("Format", Value::String("BC".into())),
                ("Labels", Value::String("BC".into())),
            ],
        )
        .expect("fixture object");
        let tape = Rc::new(RefCell::new(super::super::super::tape::Tape::new()));
        let _activation = super::super::super::tape::TapeActivation::enter(tape);
        let network_owner = dense_network();
        let network =
            super::super::super::tape::annotate_tree(network_owner.value().clone()).unwrap();
        let input = super::super::super::tape::annotate_tree(input).unwrap();
        provider.reset_telemetry();
        let loss = block_on(model::forward_builtin(
            network.clone(),
            input.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let _requested =
            crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let gradients = block_on(super::super::dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![input, network]),
        ))
        .unwrap();
        let Value::OutputList(gradients) = gradients else {
            panic!("input and network gradients expected")
        };
        assert_eq!(gradients.len(), 2);
        let mut gradients = gradients
            .try_into_owned()
            .expect("unique gradient outputs")
            .into_iter();
        let network_gradient = gradients.next_back().unwrap();
        let gradient = gradients.next_back().unwrap();
        let Value::Object(gradient) = gradient else {
            panic!("dlarray gradient expected")
        };
        let Some(Value::GpuTensor(output)) = gradient.properties().get("Data") else {
            panic!("native resident gradient expected")
        };
        assert_eq!(output.shape(), &[1, 2]);
        assert_eq!(handle_precision(output), Some(ProviderPrecision::F64));
        assert_eq!(
            handle_provenance(output),
            Some(GpuHandleProvenance::Explicit)
        );
        assert!(
            crate::builtins::common::gpu_helpers::exact_provider_for_binary_inputs(&raw, output)
                .is_ok()
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(output.clone()))
                .unwrap();
        assert_eq!(gathered.materialize_f64(), vec![2.0, -1.0]);
        let Value::Object(table) = network_gradient else {
            panic!("canonical Learnables gradient table expected")
        };
        let variables = crate::builtins::table::table_variables(&table).unwrap();
        let Some(Value::StringArray(roles)) = variables.fields.get("Parameter") else {
            panic!("canonical Learnables Parameter column expected")
        };
        assert_eq!(
            roles
                .data()
                .iter()
                .map(|value| value.as_str().expect("present parameter role"))
                .collect::<Vec<_>>(),
            ["Weights", "Bias"]
        );
        let Some(Value::Cell(values)) = variables.fields.get("Value") else {
            panic!("canonical Learnables Value column expected")
        };
        assert_eq!(values.data().len(), 2);
        for (value, shape, expected) in [
            (&values.data()[0], &[1, 2][..], &[3.0, 4.0][..]),
            (&values.data()[1], &[1, 1][..], &[1.0][..]),
        ] {
            let Value::Object(value) = value else {
                panic!("Learnables gradient dlarray expected")
            };
            let Some(Value::GpuTensor(handle)) = value.properties().get("Data") else {
                panic!("native Learnables gradient expected")
            };
            assert_eq!(handle.shape(), shape);
            assert_eq!(handle_precision(handle), Some(ProviderPrecision::F64));
            let gathered =
                crate::builtins::common::test_support::gather(Value::GpuTensor(handle.clone()))
                    .unwrap();
            assert_eq!(gathered.materialize_f64(), expected);
        }
        assert_eq!(raw.shape(), &[1, 2]);
    });
}

#[test]
fn canonical_dlgradient_uses_explicit_output_count_and_cancellation() {
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};

    crate::builtins::common::test_support::with_test_provider(|provider| {
        let raw = provider
            .upload(&HostTensorView {
                data: &[3.0, 4.0],
                shape: &[1, 2],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(raw)),
                ("Format", Value::String("BC".into())),
                ("Labels", Value::String("BC".into())),
            ],
        )
        .expect("fixture object");
        let tape = Rc::new(RefCell::new(super::super::super::tape::Tape::new()));
        let _activation = super::super::super::tape::TapeActivation::enter(tape);
        let network_owner = dense_network();
        let network =
            super::super::super::tape::annotate_tree(network_owner.value().clone()).unwrap();
        let input = super::super::super::tape::annotate_tree(input).unwrap();
        let loss = block_on(model::forward_builtin(
            network,
            input.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let args = [loss, input];
        let bindings = crate::builtin::runtime_builtin_bindings_by_name("dlgradient");
        assert_eq!(bindings.len(), 1);
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        runtime.set_error_namespace("Invocation");
        let _ambient_outputs =
            crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let result = block_on((bindings[0].implementation)(
            crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(runtime.clone()), Some(1)),
        ))
        .unwrap();
        assert_eq!(result.kind(), runmat_value::ValueSequenceKind::Single);
        assert!(matches!(result.as_slice(), [Value::Object(_)]));

        let source = CancellationSource::new();
        source.cancel(CancellationReason::User);
        let cancelled = runtime.with_cancellation_token(source.token());
        provider.reset_telemetry();
        let before = provider.telemetry_snapshot();
        let error = block_on((bindings[0].implementation)(
            crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(cancelled), Some(1)),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("Invocation:Cancelled"));
        assert_eq!(provider.telemetry_snapshot(), before);
    });
}
