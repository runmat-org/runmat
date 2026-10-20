use futures::executor::block_on;
use runmat_value::{
    CellArray, ComplexElement, ComplexStorage, ComplexTensor, NumericDType, NumericStorage, Tensor,
    Value,
};

use super::{admit_training_geometry, HostNetworkPlan, HostValue};
use crate::builtins::deep_learning::model::state::{self, StateRole, StateUpdate};
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
use crate::builtins::deep_learning::{layers::*, model::*, object};

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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn feature_input_preparation_keeps_admitted_rank_one_shape_exact() {
    let network_owner = network();
    let Value::Object(network) = network_owner.value() else {
        panic!("network object expected")
    };
    let rank_one = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2]).unwrap());
    let plan = HostNetworkPlan::admit(network, &rank_one, "forward").expect("feature input");
    assert_eq!(plan.input_axes().shape(), &[2]);
    let HostValue::Real(prepared) = block_on(plan.prepare_input(&rank_one)).unwrap() else {
        panic!("real input expected")
    };
    assert_eq!(prepared.shape(), &[2]);
    assert_eq!(prepared.numeric_dtype(), NumericDType::F32);
    assert_eq!(prepared.as_f32_slice(), Some(&[1.0, 2.0][..]));

    let row_shorthand = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap());
    assert!(block_on(plan.prepare_input(&row_shorthand)).is_err());
    let rank_zero = Value::Tensor(Tensor::from_f32(vec![1.0], Vec::new()).unwrap());
    assert!(HostNetworkPlan::admit(network, &rank_zero, "forward").is_err());
}

#[test]
fn training_geometry_uses_initialization_shapes_before_any_parameter_is_filled() {
    let input_layer = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(
        crate::builtins::deep_learning::tests::fully_connected_layer_fixture(
            Value::Num(4.0),
            vec![],
        ),
    )
    .unwrap();
    let layers = vec![input_layer, dense];
    let input = Value::Tensor(Tensor::from_f32(vec![1.; 6], vec![3, 2]).unwrap());
    let Value::Object(dense) = &layers[1] else {
        panic!("dense layer")
    };
    assert!(dense.properties().get("Weights").is_none_or(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.shape().contains(&0))
    }));
    let layers = Value::Cell(CellArray::new(layers, 2, 1).unwrap());
    let admitted = admit_training_geometry(&layers, &input, "test").unwrap();
    assert_eq!(admitted.input_axes.shape(), &[3, 2]);
    assert_eq!(admitted.output_axes.shape(), &[3, 4]);
}

#[test]
fn uninitialized_recurrent_geometry_uses_canonical_state_and_gate_shapes() {
    let input_layer = block_on(sequence_input_layer_builtin(
        Value::Num(3.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let recurrent = block_on(lstm_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let layers = vec![input_layer, recurrent];
    let input = Value::Tensor(Tensor::from_f32(vec![1.; 3 * 2 * 4], vec![3, 2, 4]).unwrap());
    let layers = Value::Cell(CellArray::new(layers, 2, 1).unwrap());
    let admitted = admit_training_geometry(&layers, &input, "test").unwrap();
    assert_eq!(admitted.input_axes.shape(), &[3, 2, 4]);
    assert_eq!(admitted.output_axes.shape(), &[2, 2, 4]);
}

fn compare_training_geometry_with_filled_network(layers: Vec<Value>, input: Value) {
    let layers = Value::Cell(CellArray::new(layers, 2, 1).unwrap());
    let proposed = admit_training_geometry(&layers, &input, "test").unwrap();
    let network = block_on(dlnetwork_builtin(vec![layers])).unwrap();
    let Value::Object(network) = network.value() else {
        panic!("initialized network expected")
    };
    let filled = HostNetworkPlan::admit(network, &input, "test").unwrap();
    assert_eq!(proposed.input_axes.shape(), filled.input_axes().shape());
    assert_eq!(proposed.output_axes.shape(), filled.output_axes().shape());
    assert_eq!(proposed.output_axes.layout(), filled.output_axes().layout());
}

#[test]
fn training_geometry_matches_filled_conv_normalization_and_pooling_routes() {
    let input = block_on(sequence_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let conv = block_on(convolution_1d_layer_builtin(
        Value::Num(3.0),
        Value::Num(4.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    compare_training_geometry_with_filled_network(
        vec![input, conv],
        Value::Tensor(Tensor::from_f32(vec![1.; 2 * 3 * 5], vec![2, 3, 5]).unwrap()),
    );

    let input = block_on(feature_input_layer_builtin(
        Value::Num(3.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let normalization = block_on(layer_normalization_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    compare_training_geometry_with_filled_network(
        vec![input, normalization],
        Value::Tensor(Tensor::from_f32(vec![1.; 4 * 3], vec![4, 3]).unwrap()),
    );

    let input = block_on(sequence_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let pooling = block_on(global_average_pooling_1d_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    compare_training_geometry_with_filled_network(
        vec![input, pooling],
        Value::Tensor(Tensor::from_f32(vec![1.; 2 * 3 * 5], vec![2, 3, 5]).unwrap()),
    );
}

#[test]
fn training_geometry_matches_filled_bilstm_sequence_and_last_modes() {
    for options in [
        Vec::new(),
        vec![
            Value::String("OutputMode".into()),
            Value::String("last".into()),
        ],
    ] {
        let input = block_on(sequence_input_layer_builtin(
            Value::Num(3.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let recurrent = block_on(bilstm_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(options),
        ))
        .unwrap();
        compare_training_geometry_with_filled_network(
            vec![input, recurrent],
            Value::Tensor(Tensor::from_f32(vec![1.; 3 * 2 * 4], vec![3, 2, 4]).unwrap()),
        );
    }
}

#[test]
fn training_geometry_reads_supplied_network_canonical_state_batch() {
    let input_layer = block_on(sequence_input_layer_builtin(
        Value::Num(3.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let recurrent = block_on(lstm_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let network_owner = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input_layer, recurrent], 2, 1).unwrap(),
    )]))
    .unwrap();
    let Value::Object(network) = network_owner.value() else {
        panic!("network expected")
    };
    let input = Value::Tensor(Tensor::from_f32(vec![1.; 3 * 2 * 4], vec![3, 2, 4]).unwrap());
    let proposed =
        admit_training_geometry(&Value::Object(network.clone()), &input, "test").unwrap();
    let filled = HostNetworkPlan::admit(network, &input, "test").unwrap();
    assert_eq!(proposed.output_axes.shape(), filled.output_axes().shape());

    let state = state::publish_updates(
        network,
        vec![
            StateUpdate {
                layer_index: 1,
                role: StateRole::HiddenState,
                value: Tensor::new(vec![1.; 6], vec![2, 3]).unwrap(),
            },
            StateUpdate {
                layer_index: 1,
                role: StateRole::CellState,
                value: Tensor::new(vec![1.; 6], vec![2, 3]).unwrap(),
            },
        ],
        "test",
    )
    .unwrap();
    let Value::Object(network) = state::apply(network.clone(), state, "test").unwrap() else {
        panic!("updated network expected")
    };
    let error = admit_training_geometry(&Value::Object(network), &input, "test")
        .err()
        .expect("incompatible canonical State must fail");
    assert!(
        error.message.contains("State") || error.message.contains("state"),
        "{error:?}"
    );
}

#[test]
fn admitted_steps_retain_their_own_input_and_output_axes() {
    let network_owner = network();
    let Value::Object(network) = network_owner.value() else {
        panic!("network object expected")
    };
    let input = Value::Tensor(Tensor::new(vec![1.0; 6], vec![3, 2]).unwrap());
    let plan = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    assert_eq!(plan.steps.len(), 2);
    assert_eq!(plan.steps[0].input_axes.shape(), &[3, 2]);
    assert_eq!(plan.steps[0].output_axes.shape(), &[3, 2]);
    assert_eq!(plan.steps[1].input_axes, plan.steps[0].output_axes);
    assert_eq!(plan.steps[1].output_axes.shape(), &[3, 1]);
    assert_eq!(plan.steps[1].output_axes, plan.output_axes);

    let formatted = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::Tensor(Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap()),
            ),
            ("Format", Value::String("CB".into())),
            ("Labels", Value::String("CB".into())),
        ],
    )
    .expect("fixture object");
    let plan = HostNetworkPlan::admit(network, &formatted, "forward").unwrap();
    assert_eq!(plan.steps[0].input_axes.shape(), &[2, 3]);
    assert_eq!(plan.steps[1].input_axes, plan.steps[0].output_axes);
    assert_eq!(plan.steps[1].output_axes.shape(), &[1, 3]);
}

fn pooling_network(input_layer: Value) -> crate::execution::RootedValue {
    let pooling = block_on(global_average_pooling_1d_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input_layer, pooling], 2, 1).unwrap(),
    )]))
    .unwrap()
}

#[test]
fn public_raw_complex_sequence_gap_retains_double_and_input_storage() {
    let input_layer = block_on(sequence_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let network = pooling_network(input_layer);
    let input = ComplexTensor::new(
        vec![
            (1., 10.),
            (2., 20.),
            (3., 30.),
            (4., 40.),
            (5., 50.),
            (6., 60.),
        ],
        vec![2, 3],
    )
    .unwrap();
    let original = input.as_f64_slice().unwrap().as_ptr();
    let Value::ComplexTensor(result) = block_on(forward_builtin(
        network.value().clone(),
        Value::ComplexTensor(input.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap() else {
        panic!("complex tensor result expected")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F64);
    assert_eq!(result.shape(), &[2, 1]);
    assert_eq!(
        result.as_f64_slice().unwrap(),
        &[ComplexElement(3., 30.), ComplexElement(4., 40.)]
    );
    assert_eq!(input.as_f64_slice().unwrap().as_ptr(), original);
    assert_eq!(input.as_f64_slice().unwrap()[0], ComplexElement(1., 10.));
}

#[test]
fn public_formatted_complex_gap_supports_cbt_and_no_channel_sb() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let vector = pooling_network(
        block_on(sequence_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
    );
    let values = (1..=12)
        .map(|value| (value as f32, (value * 10) as f32))
        .collect();
    let cbt = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::ComplexTensor(ComplexTensor::from_f32(values, vec![2, 2, 3]).unwrap()),
            ),
            ("Format", Value::String("CBT".into())),
            ("Labels", Value::String("CBT".into())),
        ],
    )
    .expect("fixture object");
    let Value::Object(cbt_output) = block_on(forward_builtin(
        vector.value().clone(),
        cbt,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap() else {
        panic!("formatted complex result expected")
    };
    let Some(Value::ComplexTensor(cbt_data)) = cbt_output.properties().get("Data") else {
        panic!("complex dlarray data expected")
    };
    assert_eq!(cbt_data.numeric_dtype(), NumericDType::F32);
    assert_eq!(cbt_data.shape(), &[2, 2]);
    assert_eq!(
        cbt_data.as_f32_slice().unwrap(),
        &[
            ComplexElement(5., 50.),
            ComplexElement(6., 60.),
            ComplexElement(7., 70.),
            ComplexElement(8., 80.),
        ]
    );
    assert_eq!(
        cbt_output.properties().get("Format"),
        Some(&Value::String("CB".into()))
    );

    let spatial_size = Value::Tensor(Tensor::new(vec![3., 1.], vec![1, 2]).unwrap());
    let spatial = pooling_network(
        block_on(feature_input_layer_builtin(
            spatial_size,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
    );
    let sb = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::ComplexTensor(
                    ComplexTensor::new(
                        vec![
                            (1., 10.),
                            (2., 20.),
                            (3., 30.),
                            (4., 40.),
                            (5., 50.),
                            (6., 60.),
                        ],
                        vec![3, 2],
                    )
                    .unwrap(),
                ),
            ),
            ("Format", Value::String("SB".into())),
            ("Labels", Value::String("SB".into())),
        ],
    )
    .expect("fixture object");
    let Value::Object(sb_output) = block_on(forward_builtin(
        spatial.value().clone(),
        sb,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap() else {
        panic!("formatted no-channel complex result expected")
    };
    let Some(Value::ComplexTensor(sb_data)) = sb_output.properties().get("Data") else {
        panic!("complex dlarray data expected")
    };
    assert_eq!(sb_data.shape(), &[1, 2]);
    assert_eq!(
        sb_data.as_f64_slice().unwrap(),
        &[ComplexElement(2., 20.), ComplexElement(5., 50.)]
    );
    assert_eq!(
        sb_output.properties().get("Format"),
        Some(&Value::String("SB".into()))
    );
}

#[test]
fn public_real_gap_admits_documented_channel_less_bt_sb_and_sbt_formats() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    for (format, shape, input_size, sequence, output_format, output_shape, expected) in [
        (
            "BT",
            vec![2, 3],
            vec![1.],
            true,
            "BU",
            vec![2, 1],
            vec![3., 4.],
        ),
        (
            "SB",
            vec![3, 2],
            vec![3., 1.],
            false,
            "SB",
            vec![1, 2],
            vec![2., 5.],
        ),
        (
            "SBT",
            vec![3, 2, 2],
            vec![3., 1.],
            true,
            "SBT",
            vec![1, 2, 2],
            vec![2., 5., 8., 11.],
        ),
    ] {
        let input_size =
            Value::Tensor(Tensor::new(input_size.clone(), vec![1, input_size.len()]).unwrap());
        let input_layer = if sequence {
            block_on(sequence_input_layer_builtin(
                input_size,
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .unwrap()
        } else {
            block_on(feature_input_layer_builtin(
                input_size,
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .unwrap()
        };
        let network = pooling_network(input_layer);
        let count = shape.iter().product();
        let input = object(
            "dlarray",
            vec![
                (
                    "Data",
                    Value::Tensor(
                        Tensor::new((1..=count).map(|x| x as f64).collect(), shape).unwrap(),
                    ),
                ),
                ("Format", Value::String(format.into())),
                ("Labels", Value::String(format.into())),
            ],
        )
        .expect("fixture object");
        let Value::Object(output) = block_on(forward_builtin(
            network.value().clone(),
            input,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap() else {
            panic!("formatted real GAP output expected")
        };
        assert_eq!(
            output.properties().get("Format"),
            Some(&Value::String(output_format.into()))
        );
        let Some(Value::Tensor(result)) = output.properties().get("Data") else {
            panic!("real GAP tensor expected")
        };
        assert_eq!(result.shape(), output_shape.as_slice(), "{format}");
        assert_eq!(result.as_f64_slice().unwrap(), expected, "{format}");
    }
}

#[test]
fn missing_channel_requires_a_global_pool_route_and_singleton_input_size() {
    let bt = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::Tensor(Tensor::new(vec![1.; 6], vec![2, 3]).unwrap()),
            ),
            ("Format", Value::String("BT".into())),
            ("Labels", Value::String("BT".into())),
        ],
    )
    .expect("fixture object");
    let wrong_size = pooling_network(
        block_on(sequence_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
    );
    assert!(HostNetworkPlan::admit(
        match wrong_size.value() {
            Value::Object(object) => object,
            _ => unreachable!(),
        },
        &bt,
        "forward"
    )
    .is_err());

    let input = block_on(sequence_input_layer_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let without_pool = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, relu], 2, 1).unwrap(),
    )]))
    .unwrap();
    let Value::Object(without_pool) = without_pool.value() else {
        unreachable!()
    };
    assert!(HostNetworkPlan::admit(without_pool, &bt, "forward").is_err());

    let input = block_on(sequence_input_layer_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let pool = block_on(global_average_pooling_1d_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![1.], vec![1, 1]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let with_channel_dependent_layer = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, pool, dense], 3, 1).unwrap(),
    )]))
    .unwrap();
    let Value::Object(with_channel_dependent_layer) = with_channel_dependent_layer.value() else {
        unreachable!()
    };
    assert!(HostNetworkPlan::admit(with_channel_dependent_layer, &bt, "forward").is_err());
}

#[test]
fn complex_preparation_shares_cow_payload_and_rejects_real_only_layers_before_effects() {
    let pool_net = pooling_network(
        block_on(sequence_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap(),
    );
    let input = Value::ComplexTensor(
        ComplexTensor::from_f32(vec![(1., 2.), (3., 4.), (5., 6.), (7., 8.)], vec![2, 2]).unwrap(),
    );
    let Value::Object(network_object) = pool_net.value() else {
        panic!("network object expected")
    };
    let admitted = HostNetworkPlan::admit(network_object, &input, "forward").unwrap();
    let prepared = block_on(admitted.prepare()).unwrap();
    let HostValue::Complex(prepared_input) = block_on(prepared.prepare_input(&input)).unwrap()
    else {
        panic!("complex prepared value expected")
    };
    let Value::ComplexTensor(original) = &input else {
        unreachable!()
    };
    let (ComplexStorage::F32(left), ComplexStorage::F32(right)) =
        (original.complex_storage(), prepared_input.complex_storage())
    else {
        panic!("single-precision complex storage expected")
    };
    assert!(left.shares_allocation_with(right));

    let dense_network_owner = network();
    let Value::Object(dense_network) = dense_network_owner.value() else {
        panic!("network object expected")
    };
    let rejected = HostNetworkPlan::admit(dense_network, &input, "forward");
    assert!(rejected.is_err());
}

#[test]
fn nonsquare_bc_and_cb_use_distinct_physical_channel_axes() {
    let net = network();
    let raw_bc =
        Value::Tensor(Tensor::new(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], vec![3, 2]).unwrap());
    let Value::Tensor(raw) = block_on(forward_builtin(
        net.value().clone(),
        raw_bc,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap() else {
        panic!("raw tensor result expected")
    };
    assert_eq!(raw.shape(), &[3, 1]);
    assert_eq!(raw.materialize_f64(), vec![0.5, 2.5, 4.5]);

    let cb = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap()),
            ),
            ("Format", Value::String("CB".into())),
            ("Labels", Value::String("CB".into())),
        ],
    )
    .expect("fixture object");
    let Value::Object(wrapped) = block_on(forward_builtin(
        net.value().clone(),
        cb,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap() else {
        panic!("dlarray result expected")
    };
    let Some(Value::Tensor(result)) = wrapped.properties().get("Data") else {
        panic!("tensor data expected")
    };
    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result.materialize_f64(), vec![0.5, 2.5, 4.5]);
    assert_eq!(
        wrapped.properties().get("Format"),
        Some(&Value::String("CB".into()))
    );
}

#[test]
fn single_input_projects_double_parameters_only_at_execution_boundary() {
    let net = network();
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let Value::Tensor(output) = block_on(forward_builtin(
        net.value().clone(),
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap() else {
        panic!("tensor result expected")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.shape(), &[2, 1]);
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![-0.5, 0.5])
    );
    let Value::Object(net) = net.value() else {
        panic!("network object")
    };
    let rows = parameters::view(net, "forward").unwrap();
    assert!(rows
        .iter()
        .filter_map(|row| row.value)
        .all(|value| match value {
            Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F64,
            _ => false,
        }));
}

#[test]
fn stateless_two_output_forward_publishes_canonical_empty_state() {
    let network = network();
    let _count = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let input = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let Value::OutputList(outputs) = block_on(forward_builtin(
        network.value().clone(),
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap() else {
        panic!("[Y, state] result expected")
    };
    assert_eq!(outputs.len(), 2);
    assert!(matches!(&outputs[0], Value::Tensor(_)));
    assert!(matches!(&outputs[1], Value::Object(_)));
}
