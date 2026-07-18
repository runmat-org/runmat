use super::graph::*;
use super::layers::*;
use super::losses::*;
use super::sequences::*;
use super::training::*;
use futures::executor::block_on;
use runmat_accelerate_api::{handle_precision, handle_storage, HostTensorView};
use runmat_builtins::{
    CellArray, LogicalArray, NumericDType, StringArray, StructValue, Tensor, Value,
};

fn layer_name(value: &Value) -> String {
    let Value::Object(object) = value else {
        panic!("expected object");
    };
    match object.properties.get("Name").unwrap() {
        Value::String(name) => name.clone(),
        other => panic!("expected string name, got {other:?}"),
    }
}

#[test]
fn layer_constructors_preserve_core_properties_and_name_values() {
    let fc = block_on(fully_connected_layer_builtin(
        Value::Num(10.0),
        vec![Value::String("Name".into()), Value::String("fc1".into())],
    ))
    .unwrap();
    let Value::Object(object) = &fc else {
        panic!("expected object");
    };
    assert_eq!(object.class_name, "nnet.cnn.layer.FullyConnectedLayer");
    assert_eq!(object.properties.get("OutputSize"), Some(&Value::Num(10.0)));
    assert_eq!(layer_name(&fc), "fc1");

    let conv = block_on(convolution_1d_layer_builtin(
        Value::Num(3.0),
        Value::Num(16.0),
        vec![
            Value::String("Padding".into()),
            Value::String("causal".into()),
            Value::String("Name".into()),
            Value::String("conv".into()),
        ],
    ))
    .unwrap();
    let Value::Object(object) = &conv else {
        panic!("expected object");
    };
    assert_eq!(object.properties.get("FilterSize"), Some(&Value::Num(3.0)));
    assert_eq!(object.properties.get("NumFilters"), Some(&Value::Num(16.0)));
    assert_eq!(
        object.properties.get("Padding"),
        Some(&Value::String("causal".into()))
    );
    assert_eq!(layer_name(&conv), "conv");
}

#[test]
fn recurrent_and_input_layers_accept_common_shapes() {
    let input = block_on(sequence_input_layer_builtin(
        Value::Tensor(Tensor::new(vec![4.0, 2.0], vec![1, 2]).unwrap()),
        vec![Value::String("Name".into()), Value::String("seq".into())],
    ))
    .unwrap();
    assert_eq!(layer_name(&input), "seq");
    let lstm = block_on(lstm_layer_builtin(
        Value::Num(8.0),
        vec![
            Value::String("OutputMode".into()),
            Value::String("sequence".into()),
        ],
    ))
    .unwrap();
    let Value::Object(object) = &lstm else {
        panic!("expected object");
    };
    assert_eq!(
        object.properties.get("NumHiddenUnits"),
        Some(&Value::Num(8.0))
    );
    assert_eq!(
        object.properties.get("OutputMode"),
        Some(&Value::String("sequence".into()))
    );
}

#[test]
fn training_options_and_layer_graph_materialize_metadata() {
    let opts = block_on(training_options_builtin(
        Value::String("adam".into()),
        vec![
            Value::String("MaxEpochs".into()),
            Value::Num(5.0),
            Value::String("Verbose".into()),
            Value::Bool(false),
        ],
    ))
    .unwrap();
    let Value::Object(options) = opts else {
        panic!("expected options object");
    };
    assert_eq!(
        options.properties.get("SolverName"),
        Some(&Value::String("adam".into()))
    );
    assert_eq!(options.properties.get("MaxEpochs"), Some(&Value::Num(5.0)));
    assert_eq!(options.properties.get("Verbose"), Some(&Value::Bool(false)));

    let relu = block_on(relu_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("relu".into()),
    ]))
    .unwrap();
    let graph = block_on(layer_graph_builtin(vec![Value::Cell(
        CellArray::new(vec![relu], 1, 1).unwrap(),
    )]))
    .unwrap();
    let Value::Object(graph) = graph else {
        panic!("expected graph object");
    };
    assert_eq!(graph.class_name, "nnet.cnn.LayerGraph");
    let Value::StringArray(names) = graph.properties.get("LayerNames").unwrap() else {
        panic!("expected names");
    };
    assert_eq!(names.data, vec!["relu"]);
}

#[test]
fn analyze_network_counts_layers() {
    let relu = block_on(relu_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("relu".into()),
    ]))
    .unwrap();
    let analysis = block_on(analyze_network_builtin(Value::Cell(
        CellArray::new(vec![relu], 1, 1).unwrap(),
    )))
    .unwrap();
    let Value::Object(object) = analysis else {
        panic!("expected analysis object");
    };
    assert_eq!(object.properties.get("NumLayers"), Some(&Value::Num(1.0)));
}

#[test]
fn combvec_matches_neural_network_column_order() {
    let out = block_on(combvec_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap()),
    ]))
    .unwrap();
    let Value::Tensor(t) = out else {
        panic!("expected tensor");
    };
    assert_eq!(t.shape, vec![3, 6]);
    assert_eq!(
        t.data,
        vec![
            1.0, 2.0, 10.0, 3.0, 4.0, 10.0, 1.0, 2.0, 20.0, 3.0, 4.0, 20.0, 1.0, 2.0, 30.0, 3.0,
            4.0, 30.0,
        ]
    );
}

#[test]
fn padsequences_defaults_to_uniform_right_padding() {
    let seqs = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                Value::Tensor(Tensor::new(vec![3.0], vec![1, 1]).unwrap()),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let out = block_on(padsequences_builtin(
        seqs,
        vec![
            Value::Num(2.0),
            Value::String("PaddingValue".into()),
            Value::Num(-1.0),
        ],
    ))
    .unwrap();
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape, vec![1, 2, 1, 2]);
    assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, -1.0]);
}

#[test]
fn padsequences_left_truncation_keeps_tail_for_cell_output() {
    let seqs = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
                Value::Tensor(Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap()),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let out = block_on(padsequences_builtin(
        seqs,
        vec![
            Value::Num(2.0),
            Value::String("Length".into()),
            Value::Num(2.0),
            Value::String("PaddingDirection".into()),
            Value::String("left".into()),
            Value::String("UniformOutput".into()),
            Value::Bool(false),
        ],
    ))
    .unwrap();
    let Value::Cell(cell) = out else {
        panic!("expected cell");
    };
    assert_eq!(cell.shape, vec![1, 2]);
    let Value::Tensor(first) = &cell.data[0] else {
        panic!("expected tensor");
    };
    assert_eq!(first.data, vec![2.0, 3.0]);
}

#[test]
fn dlarray_preserves_data_and_format_labels() {
    let data = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let out = block_on(dlarray_builtin(
        data.clone(),
        vec![Value::String("CB".into())],
    ))
    .expect("dlarray");
    let Value::Object(object) = out else {
        panic!("expected object");
    };
    assert_eq!(object.class_name, "dlarray");
    assert_eq!(object.properties.get("Data"), Some(&data));
    assert_eq!(
        object.properties.get("Format"),
        Some(&Value::String("CB".into()))
    );
}

#[test]
fn dlarray_preserves_gpu_array_residency() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let out = block_on(dlarray_builtin(
            Value::GpuTensor(handle.clone()),
            vec![Value::String("CB".into())],
        ))
        .expect("dlarray gpu");
        let Value::Object(object) = out else {
            panic!("expected object");
        };
        assert_eq!(object.class_name, "dlarray");
        let Value::GpuTensor(wrapped) = object.properties.get("Data").unwrap() else {
            panic!("expected gpu tensor data");
        };
        assert_eq!(wrapped.buffer_id, handle.buffer_id);
        assert_eq!(wrapped.device_id, handle.device_id);
        assert_eq!(wrapped.shape, handle.shape);
        assert_eq!(handle_precision(wrapped), handle_precision(&handle));
        assert_eq!(handle_storage(wrapped), handle_storage(&handle));
        assert_eq!(
            object.properties.get("Format"),
            Some(&Value::String("CB".into()))
        );
    });
}

#[test]
fn crossentropy_computes_mean_loss() {
    let out = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![],
    ))
    .expect("crossentropy");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    assert!((loss - 0.11157177565710488).abs() < 1.0e-12);
}

#[test]
fn crossentropy_gpu_inputs_use_provider_for_scalar_reduction() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &shape,
            })
            .expect("upload predictions");
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload targets");

        provider.reset_telemetry();
        let out = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            vec![],
        ))
        .expect("crossentropy gpu");
        let Value::Num(loss) = out else {
            panic!("expected scalar");
        };
        assert!((loss - 0.11157177565710488).abs() < 1.0e-12);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn crossentropy_gpu_reduction_none_returns_resident_losses() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &shape,
            })
            .expect("upload predictions");
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload targets");

        provider.reset_telemetry();
        let out = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            vec![
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
                Value::String("Reduction".into()),
                Value::String("none".into()),
            ],
        ))
        .expect("crossentropy gpu unreduced");
        assert!(matches!(out, Value::GpuTensor(_)));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered = crate::builtins::common::test_support::gather(out).expect("gather losses");
        assert_eq!(gathered.shape, shape);
        assert!((gathered.data[0] + 0.8_f64.ln()).abs() < 1.0e-12);
        assert!((gathered.data[1] + 0.8_f64.ln()).abs() < 1.0e-12);
    });
}

#[test]
fn crossentropy_gpu_supports_host_weights_without_gathering_predictions() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [2usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2, 0.25, 0.75],
                shape: &shape,
            })
            .expect("upload predictions");
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0, 0.0, 1.0],
                shape: &shape,
            })
            .expect("upload targets");

        provider.reset_telemetry();
        let out = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            vec![
                Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap()),
                Value::String("DataFormat".into()),
                Value::String("CB".into()),
            ],
        ))
        .expect("weighted gpu crossentropy");
        let Value::Num(loss) = out else {
            panic!("expected scalar");
        };
        let expected = (-2.0_f64 * 0.8_f64.ln() - 0.75_f64.ln()) / 2.0;
        assert!((loss - expected).abs() < 1.0e-12);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn crossentropy_gpu_supports_host_mask_for_unreduced_losses() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &shape,
            })
            .expect("upload predictions");
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload targets");

        provider.reset_telemetry();
        let out = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            vec![
                Value::String("Mask".into()),
                Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
                Value::String("Reduction".into()),
                Value::String("none".into()),
            ],
        ))
        .expect("masked gpu crossentropy");
        assert!(matches!(out, Value::GpuTensor(_)));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered = crate::builtins::common::test_support::gather(out).expect("gather losses");
        assert_eq!(gathered.shape, shape);
        assert!((gathered.data[0] + 0.8_f64.ln()).abs() < 1.0e-12);
        assert_eq!(gathered.data[1], 0.0);
    });
}

#[test]
fn crossentropy_gpu_supports_resident_weights_and_mask_without_downloads() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &shape,
            })
            .expect("upload predictions");
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload targets");
        let weights = provider
            .upload(&HostTensorView {
                data: &[2.0, 3.0],
                shape: &shape,
            })
            .expect("upload weights");
        let mask = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload mask");

        provider.reset_telemetry();
        let out = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            vec![
                Value::GpuTensor(weights),
                Value::String("Mask".into()),
                Value::GpuTensor(mask),
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
            ],
        ))
        .expect("resident weighted masked gpu crossentropy");
        let Value::Num(loss) = out else {
            panic!("expected scalar");
        };
        assert!((loss + 0.8_f64.ln()).abs() < 1.0e-12);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn crossentropy_supports_dataformat_class_weights_and_batch_normalization() {
    let out = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        vec![
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
        ],
    ))
    .expect("weighted crossentropy");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    let expected = (-2.0_f64 * 0.8_f64.ln() - 0.75_f64.ln()) / 2.0;
    assert!(
        (loss - expected).abs() < 1.0e-12,
        "loss {loss} expected {expected}"
    );
}

#[test]
fn crossentropy_supports_weights_name_value_and_weights_format_mapping() {
    let out = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("UC".into()),
        ],
    ))
    .expect("formatted class weights");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    let expected = (-2.0_f64 * 0.8_f64.ln() - 0.75_f64.ln()) / 2.0;
    assert!((loss - expected).abs() < 1.0e-12);

    let out = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![0.5, 2.0], vec![1, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("UB".into()),
        ],
    ))
    .expect("formatted observation weights");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    let expected = (-0.5_f64 * 0.8_f64.ln() - 2.0_f64 * 0.75_f64.ln()) / 2.0;
    assert!((loss - expected).abs() < 1.0e-12);
}

#[test]
fn crossentropy_remaps_full_size_formatted_weights() {
    let out = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, 3.0, 5.0, 7.0], vec![2, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("BC".into()),
        ],
    ))
    .expect("formatted full-size weights");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    let expected = (-2.0_f64 * 0.8_f64.ln() - 7.0_f64 * 0.75_f64.ln()) / 2.0;
    assert!((loss - expected).abs() < 1.0e-12);

    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("TC".into()),
        ],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("not present in DataFormat"));
}

#[test]
fn crossentropy_supports_multilabel_reduction_none_and_mask_normalization() {
    let unreduced = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("ClassificationMode".into()),
            Value::String("multi-label".into()),
            Value::String("Reduction".into()),
            Value::String("none".into()),
        ],
    ))
    .expect("multilabel unreduced crossentropy");
    let Value::Tensor(tensor) = unreduced else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape, vec![1, 2]);
    assert!((tensor.data[0] + 0.8_f64.ln()).abs() < 1.0e-12);
    assert!((tensor.data[1] + 0.8_f64.ln()).abs() < 1.0e-12);

    let masked = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("Mask".into()),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
            Value::String("NormalizationFactor".into()),
            Value::String("mask-included".into()),
        ],
    ))
    .expect("masked crossentropy");
    let Value::Num(loss) = masked else {
        panic!("expected scalar");
    };
    assert!((loss + 0.8_f64.ln()).abs() < 1.0e-12);
}

#[test]
fn crossentropy_rejects_incompatible_shapes_masks_and_formats() {
    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        vec![],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("prediction shape"));

    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![Value::String("Mask".into()), Value::Bool(true)],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("same size"));

    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("Mask".into()),
            Value::Tensor(Tensor::new(vec![0.5, 1.0], vec![1, 2]).unwrap()),
        ],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("binary"));

    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("DataFormat".into()),
            Value::String("CC".into()),
        ],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("cannot repeat"));
}

#[test]
fn crossentropy_preserves_dlarray_output_for_dlarray_inputs() {
    let predictions = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![2, 1]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .expect("prediction dlarray");
    let targets = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .expect("target dlarray");
    let out =
        block_on(crossentropy_builtin(predictions, targets, vec![])).expect("dlarray crossentropy");
    let Value::Object(object) = out else {
        panic!("expected dlarray object");
    };
    assert_eq!(object.class_name, "dlarray");
    let Value::Num(loss) = object.properties.get("Data").unwrap() else {
        panic!("expected scalar data");
    };
    assert!((*loss + 0.8_f64.ln()).abs() < 1.0e-12);

    let predictions = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![2, 1]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .expect("prediction dlarray");
    let targets = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .expect("target dlarray");
    let out = block_on(crossentropy_builtin(
        predictions,
        targets,
        vec![
            Value::String("Reduction".into()),
            Value::String("none".into()),
        ],
    ))
    .expect("unreduced dlarray crossentropy");
    let Value::Object(object) = out else {
        panic!("expected dlarray object");
    };
    assert_eq!(
        object.properties.get("Format"),
        Some(&Value::String(String::new()))
    );
}

#[test]
fn adamupdate_computes_bias_corrected_tensor_step() {
    let _guard = crate::output_count::push_output_count(Some(3));
    let out = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1, -0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
        Value::Num(1.0),
    ]))
    .expect("adamupdate");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 3);
    let Value::Tensor(parameters) = &outputs[0] else {
        panic!("expected tensor parameters");
    };
    assert!((parameters.data[0] - 0.9990000001).abs() < 1.0e-9);
    assert!((parameters.data[1] - 2.00099999995).abs() < 1.0e-9);
    let Value::Tensor(average_grad) = &outputs[1] else {
        panic!("expected tensor averageGrad");
    };
    assert!((average_grad.data[0] - 0.01).abs() < 1.0e-12);
    assert!((average_grad.data[1] + 0.02).abs() < 1.0e-12);
    let Value::Tensor(average_sq_grad) = &outputs[2] else {
        panic!("expected tensor averageSqGrad");
    };
    assert!((average_sq_grad.data[0] - 0.00001).abs() < 1.0e-12);
    assert!((average_sq_grad.data[1] - 0.00004).abs() < 1.0e-12);
}

#[test]
fn adamupdate_default_call_returns_updated_parameters_directly() {
    let out = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Num(1.0),
    ]))
    .expect("adamupdate");
    let Value::Tensor(parameters) = out else {
        panic!("expected direct tensor output");
    };
    assert!((parameters.data[0] - 0.9990000001).abs() < 1.0e-9);
}

#[test]
fn adamupdate_preserves_dlarray_wrapper_for_state_outputs() {
    let _guard = crate::output_count::push_output_count(Some(3));
    let params = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .unwrap();
    let out = block_on(adamupdate_builtin(vec![
        params,
        Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
        Value::Num(1.0),
    ]))
    .expect("adamupdate");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    let Value::Object(updated) = &outputs[0] else {
        panic!("expected dlarray object");
    };
    assert_eq!(updated.class_name, "dlarray");
    assert_eq!(
        updated.properties.get("Format"),
        Some(&Value::String("CB".into()))
    );
    for value in &outputs[1..] {
        let Value::Object(state) = value else {
            panic!("expected dlarray state object");
        };
        assert_eq!(state.class_name, "dlarray");
        assert_eq!(
            state.properties.get("Format"),
            Some(&Value::String("CB".into()))
        );
    }
}

#[test]
fn adamupdate_rejects_invalid_shapes_outputs_and_hyperparameters() {
    let err = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
        Value::Num(1.0),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("grad must match"));

    let err = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Num(1.0),
        Value::Num(-0.001),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("learnRate must be positive"));

    let _guard = crate::output_count::push_output_count(Some(4));
    let err = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Num(1.0),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("at most 3"));
}

#[test]
fn adamupdate_accepts_zero_decay_and_rejects_integer_tensor_state() {
    let out = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Num(10_000_000_000.0),
        Value::Num(0.001),
        Value::Num(0.0),
        Value::Num(0.0),
    ]))
    .expect("zero decay and large iteration");
    let Value::Tensor(parameters) = out else {
        panic!("expected tensor parameters");
    };
    assert!(parameters.data[0].is_finite());

    let integer_params = Tensor::new_with_dtype(vec![1.0], vec![1, 1], NumericDType::U8).unwrap();
    let err = block_on(adamupdate_builtin(vec![
        Value::Tensor(integer_params),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Num(1.0),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("double or single"));
}

#[test]
fn dlfeval_dispatches_function_handles() {
    let out = block_on(dlfeval_builtin(vec![
        Value::FunctionHandle("plus".into()),
        Value::Num(2.0),
        Value::Num(3.0),
    ]))
    .expect("dlfeval plus");
    assert_eq!(out, Value::Num(5.0));
}

#[test]
fn dlfeval_preserves_requested_output_count() {
    let _guard = crate::output_count::push_output_count(Some(2));
    let out = block_on(dlfeval_builtin(vec![
        Value::FunctionHandle("deal".into()),
        Value::String("a".into()),
        Value::String("b".into()),
    ]))
    .expect("dlfeval deal");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(
        outputs,
        vec![Value::String("a".into()), Value::String("b".into())]
    );
}

#[test]
fn dlfeval_registered_dispatch_preserves_requested_outputs() {
    let out = block_on(crate::call_builtin_async_with_outputs(
        "dlfeval",
        &[
            Value::FunctionHandle("deal".into()),
            Value::String("left".into()),
            Value::String("right".into()),
        ],
        2,
    ))
    .expect("registered dlfeval deal");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(
        outputs,
        vec![Value::String("left".into()), Value::String("right".into())]
    );
}

#[test]
fn dlfeval_rejects_missing_or_non_callable_function() {
    let err = block_on(dlfeval_builtin(vec![])).unwrap_err();
    assert!(err.to_string().contains("expected a function handle"));

    let err = block_on(dlfeval_builtin(vec![Value::Num(1.0)])).unwrap_err();
    assert!(err.to_string().contains("expected a function handle"));

    let err = block_on(dlfeval_builtin(vec![Value::String("@plus".into())])).unwrap_err();
    assert!(err.to_string().contains("expected a function handle"));
}

#[test]
fn dlupdate_maps_matching_struct_and_cell_trees() {
    let mut left = StructValue::new();
    left.insert(
        "Weights",
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
    );
    left.insert(
        "Bias",
        Value::Cell(CellArray::new(vec![Value::Num(3.0)], 1, 1).unwrap()),
    );
    let mut right = StructValue::new();
    right.insert(
        "Weights",
        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
    );
    right.insert(
        "Bias",
        Value::Cell(CellArray::new(vec![Value::Num(4.0)], 1, 1).unwrap()),
    );

    let out = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("plus".into()),
        Value::Struct(left),
        Value::Struct(right),
    ]))
    .expect("dlupdate plus");
    let Value::Struct(out) = out else {
        panic!("expected struct output");
    };
    let Value::Tensor(weights) = out.fields.get("Weights").unwrap() else {
        panic!("expected tensor weights");
    };
    assert_eq!(weights.data, vec![11.0, 22.0]);
    let Value::Cell(bias) = out.fields.get("Bias").unwrap() else {
        panic!("expected cell bias");
    };
    assert_eq!(bias.data, vec![Value::Num(7.0)]);
}

#[test]
fn dlupdate_reconstructs_multiple_output_trees() {
    let _guard = crate::output_count::push_output_count(Some(2));
    let first = Value::Cell(CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap());
    let second =
        Value::Cell(CellArray::new(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).unwrap());

    let out = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("deal".into()),
        first.clone(),
        second.clone(),
    ]))
    .expect("dlupdate deal");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(outputs, vec![first, second]);
}

#[test]
fn dlupdate_maps_learnables_table_value_variable_only() {
    let left = crate::builtins::table::table_from_columns(
        vec!["Layer".into(), "Parameter".into(), "Value".into()],
        vec![
            Value::StringArray(
                StringArray::new(vec!["fc".into(), "fc".into()], vec![2, 1]).unwrap(),
            ),
            Value::StringArray(
                StringArray::new(vec!["Weights".into(), "Bias".into()], vec![2, 1]).unwrap(),
            ),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                        Value::Num(3.0),
                    ],
                    2,
                    1,
                )
                .unwrap(),
            ),
        ],
    )
    .expect("left table");
    let right = crate::builtins::table::table_from_columns(
        vec!["Layer".into(), "Parameter".into(), "Value".into()],
        vec![
            Value::StringArray(
                StringArray::new(vec!["fc".into(), "fc".into()], vec![2, 1]).unwrap(),
            ),
            Value::StringArray(
                StringArray::new(vec!["Weights".into(), "Bias".into()], vec![2, 1]).unwrap(),
            ),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
                        Value::Num(4.0),
                    ],
                    2,
                    1,
                )
                .unwrap(),
            ),
        ],
    )
    .expect("right table");

    let out = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("plus".into()),
        left,
        right,
    ]))
    .expect("dlupdate table");
    let Value::Object(object) = out else {
        panic!("expected table object");
    };
    let vars = crate::builtins::table::table_variables(&object).expect("table variables");
    assert_eq!(
        vars.fields.get("Layer"),
        Some(&Value::StringArray(
            StringArray::new(vec!["fc".into(), "fc".into()], vec![2, 1]).unwrap()
        ))
    );
    let Value::Cell(values) = vars.fields.get("Value").unwrap() else {
        panic!("expected Value cell");
    };
    let Value::Tensor(weights) = &values.data[0] else {
        panic!("expected weights tensor");
    };
    assert_eq!(weights.data, vec![11.0, 22.0]);
    assert_eq!(values.data[1], Value::Num(7.0));
}

#[test]
fn dlupdate_treats_dlarray_objects_as_leaves() {
    let data = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let dlarray = block_on(dlarray_builtin(
        data.clone(),
        vec![Value::String("CB".into())],
    ))
    .expect("dlarray");

    let out = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("class".into()),
        dlarray.clone(),
    ]))
    .expect("dlupdate dlarray");
    assert_eq!(out, Value::String("dlarray".into()));
}

#[test]
fn dlupdate_rejects_non_callable_and_mismatched_trees() {
    let err = block_on(dlupdate_builtin(vec![
        Value::String("@plus".into()),
        Value::Num(1.0),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("expected a function handle"));

    let err = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("plus".into()),
        Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
        Value::Cell(CellArray::new(vec![Value::Num(2.0), Value::Num(3.0)], 1, 2).unwrap()),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("same shape"));

    let generic_table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "B".into()],
        vec![Value::Num(1.0), Value::Num(2.0)],
    )
    .expect("generic table");
    let err = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("plus".into()),
        generic_table,
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("learnables tables"));

    let malformed_learnables = crate::builtins::table::table_from_columns(
        vec!["Layer".into(), "Parameter".into(), "Value".into()],
        vec![
            Value::StringArray(StringArray::new(vec!["fc".into()], vec![1, 1]).unwrap()),
            Value::StringArray(StringArray::new(vec!["Weights".into()], vec![1, 1]).unwrap()),
            Value::Num(1.0),
        ],
    )
    .expect("malformed learnables table");
    let err = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("plus".into()),
        malformed_learnables,
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("Value variable must be a cell"));
}

#[test]
fn dlarray_and_crossentropy_reject_unsupported_forms() {
    let err = block_on(dlarray_builtin(
        Value::Num(1.0),
        vec![Value::String("CB".into()), Value::String("Extra".into())],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("optional dimension-label"));

    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(Vec::new(), vec![1, 0]).unwrap()),
        Value::Tensor(Tensor::new(Vec::new(), vec![1, 0]).unwrap()),
        vec![],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("must not be empty"));
}

#[test]
fn unsupported_training_apis_return_clear_errors() {
    let err = block_on(train_network_builtin(vec![])).unwrap_err();
    assert!(err.to_string().contains("requires optimizer"));
    let err = block_on(export_onnx_network_builtin(vec![])).unwrap_err();
    assert!(err.to_string().contains("ONNX"));
}
