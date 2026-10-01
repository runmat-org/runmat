use super::*;
use crate::builtins::plotting::{clone_figure, current_figure_handle};
use futures::executor::block_on;
use runmat_value::{IntegerStorage, LogicalArray, StringArray};

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).expect("tensor"))
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    Value::Tensor(tensor)
}

#[test]
fn heterogeneous_cell_group_keys_do_not_collapse_into_display_text() {
    let wide = 9_007_199_254_740_993_u64;
    let labels = Value::Cell(
        CellArray::new(
            vec![
                Value::Int(IntValue::U64(wide)),
                Value::String(wide.to_string()),
            ],
            2,
            1,
        )
        .unwrap(),
    );
    let boxes = build_box_stats(
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Some(labels),
        &BoxplotOptions::default(),
    )
    .unwrap();
    assert_eq!(boxes.len(), 2);
    assert_eq!(boxes[0].median, 1.0);
    assert_eq!(boxes[1].median, 2.0);
    assert_eq!(boxes[0].label, boxes[1].label);
}

#[test]
fn multifactor_group_identity_does_not_depend_on_joined_display_text() {
    let first =
        Value::StringArray(StringArray::new(vec!["a,b".into(), "a".into()], vec![2, 1]).unwrap());
    let second =
        Value::StringArray(StringArray::new(vec!["c".into(), "b,c".into()], vec![2, 1]).unwrap());
    let factors = Value::Cell(CellArray::new(vec![first, second], 1, 2).unwrap());
    let boxes = build_box_stats(
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Some(factors),
        &BoxplotOptions::default(),
    )
    .unwrap();
    assert_eq!(boxes.len(), 2);
    assert_eq!(boxes[0].median, 1.0);
    assert_eq!(boxes[1].median, 2.0);
    assert_eq!(boxes[0].label, boxes[1].label);
}

#[test]
fn extensions_are_independently_mode_gated() {
    let data = || tensor(vec![1.0, 2.0], vec![2, 1]);
    let integer_data = || int_tensor(IntegerStorage::I16(vec![1, 2]), vec![2, 1]);
    let logical_data =
        || Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![2, 1]).expect("logical data"));
    let integer_group = || int_tensor(IntegerStorage::U16(vec![1, 2]), vec![2, 1]);
    let logical_group =
        || Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![2, 1]).expect("logical group"));
    let cases = [
        (
            vec![integer_data()],
            BOXPLOT_INTEGER_DATA_EXTENSION.error_identifier,
        ),
        (
            vec![logical_data()],
            BOXPLOT_LOGICAL_DATA_EXTENSION.error_identifier,
        ),
        (
            vec![data(), integer_group()],
            BOXPLOT_INTEGER_GROUP_EXTENSION.error_identifier,
        ),
        (
            vec![data(), logical_group()],
            BOXPLOT_LOGICAL_GROUP_EXTENSION.error_identifier,
        ),
        (
            vec![data(), Value::from("Whisker"), Value::Int(IntValue::U8(1))],
            BOXPLOT_INTEGER_OPTION_EXTENSION.error_identifier,
        ),
        (
            vec![data(), Value::from("Whisker"), Value::Bool(true)],
            BOXPLOT_LOGICAL_OPTION_EXTENSION.error_identifier,
        ),
    ];
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for (args, identifier) in cases {
        let error = block_on(boxplot_builtin(args)).expect_err("strict mode must reject");
        assert_eq!(error.identifier(), identifier);
    }
}

#[test]
fn documented_single_data_groups_and_options_need_no_extension() {
    let single = |values: Vec<f32>, shape: Vec<usize>| {
        Value::Tensor(Tensor::from_f32(values, shape).expect("single tensor"))
    };
    let args = vec![
        single(vec![1.0, 2.0], vec![2, 1]),
        single(vec![2.0, 1.0], vec![2, 1]),
        Value::from("Whisker"),
        single(vec![1.0], vec![1, 1]),
    ];
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    ensure_sample_group_extensions(&args[0], Some(&args[1]))
        .expect("documented single sample and group");
    BoxplotOptionPlan::admit(args[2..].to_vec(), Some(BoxplotOptionKind::Whisker))
        .expect("documented single option");
}

#[test]
fn all_integer_sample_classes_cross_the_explicit_statistics_boundary() {
    let cases = [
        (IntegerStorage::I8(vec![-1, 3]), 1.0),
        (IntegerStorage::I16(vec![-1, 3]), 1.0),
        (IntegerStorage::I32(vec![-1, 3]), 1.0),
        (IntegerStorage::I64(vec![-1, 3]), 1.0),
        (IntegerStorage::U8(vec![1, 3]), 2.0),
        (IntegerStorage::U16(vec![1, 3]), 2.0),
        (IntegerStorage::U32(vec![1, 3]), 2.0),
        (IntegerStorage::U64(vec![1, 3]), 2.0),
    ];
    for (storage, expected_median) in cases {
        let boxes = build_box_stats(
            int_tensor(storage, vec![2, 1]),
            None,
            &BoxplotOptions::default(),
        )
        .expect("integer box statistics");
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].median, expected_median);
    }
}

#[test]
fn all_integer_group_and_numeric_option_classes_are_covered() {
    let cases = [
        (IntegerStorage::I8(vec![2, 1]), IntValue::I8(1)),
        (IntegerStorage::I16(vec![2, 1]), IntValue::I16(1)),
        (IntegerStorage::I32(vec![2, 1]), IntValue::I32(1)),
        (IntegerStorage::I64(vec![2, 1]), IntValue::I64(1)),
        (IntegerStorage::U8(vec![2, 1]), IntValue::U8(1)),
        (IntegerStorage::U16(vec![2, 1]), IntValue::U16(1)),
        (IntegerStorage::U32(vec![2, 1]), IntValue::U32(1)),
        (IntegerStorage::U64(vec![2, 1]), IntValue::U64(1)),
    ];
    for (groups, option) in cases {
        let boxes = build_box_stats(
            tensor(vec![20.0, 10.0], vec![2, 1]),
            Some(int_tensor(groups, vec![2, 1])),
            &BoxplotOptions::default(),
        )
        .expect("integer groups");
        assert_eq!(
            boxes
                .iter()
                .map(|box_stats| box_stats.label.as_str())
                .collect::<Vec<_>>(),
            vec!["1", "2"]
        );
        let options = BoxplotOptions::parse(vec![Value::from("Whisker"), Value::Int(option)])
            .expect("integer numeric option");
        assert_eq!(options.whisker, 1.0);
    }
}

#[test]
fn integer_group_identity_and_labels_remain_exact_above_flintmax() {
    let wide = 9_007_199_254_740_993_u64;
    let boxes = build_box_stats(
        tensor(vec![10.0, 20.0, 30.0], vec![3, 1]),
        Some(int_tensor(
            IntegerStorage::U64(vec![wide + 1, wide, wide + 1]),
            vec![3, 1],
        )),
        &BoxplotOptions::default(),
    )
    .expect("wide integer grouping");
    assert_eq!(boxes.len(), 2);
    assert_eq!(boxes[0].label, wide.to_string());
    assert_eq!(boxes[0].median, 20.0);
    assert_eq!(boxes[1].label, (wide + 1).to_string());
    assert_eq!(boxes[1].median, 20.0);
}

#[test]
fn integer_labels_preserve_exact_text() {
    let wide = 9_007_199_254_740_993_u64;
    assert_eq!(
        text_vector(
            &int_tensor(IntegerStorage::U64(vec![wide]), vec![1, 1]),
            "Labels",
        )
        .expect("integer labels"),
        vec![wide.to_string()]
    );
}

#[test]
fn gpu_input_extension_rejects_before_gather_and_admits_host_rendering() {
    use runmat_accelerate_api::HostTensorView;

    crate::builtins::common::test_support::with_test_provider(|provider| {
        let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
        crate::builtins::plotting::reset_plot_state();
        let handle = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .expect("resident data");
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(boxplot_builtin(vec![Value::GpuTensor(handle.clone())]))
                .expect_err("strict mode rejects resident input");
            assert_eq!(
                error.identifier(),
                BOXPLOT_GPU_INPUT_EXTENSION.error_identifier
            );
        }
        {
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            let _outputs = crate::output_count::push_output_count(Some(0));
            assert_eq!(
                block_on(boxplot_builtin(vec![Value::GpuTensor(handle)]))
                    .expect("RunMat resident input"),
                Value::OutputList(Vec::new())
            );
        }
    });
}

#[test]
fn computes_matrix_columns_and_outliers() {
    let options = BoxplotOptions::default();
    let boxes = build_box_stats(
        tensor(
            vec![1.0, 2.0, 3.0, 100.0, 10.0, 20.0, 30.0, 40.0],
            vec![4, 2],
        ),
        None,
        &options,
    )
    .expect("stats");
    assert_eq!(boxes.len(), 2);
    assert_eq!(boxes[0].label, "1");
    assert!((boxes[0].median - 2.5).abs() < 1.0e-12);
    assert!(boxes[0].outliers.contains(&100.0));
    assert_eq!(boxes[1].label, "2");
    assert!((boxes[1].q1 - 17.5).abs() < 1.0e-12);
}

#[test]
fn groups_text_labels_with_group_order_and_labels() {
    let labels = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row("b")),
            Value::CharArray(CharArray::new_row("a")),
            Value::CharArray(CharArray::new_row("b")),
            Value::CharArray(CharArray::new_row("a")),
        ],
        4,
        1,
    )
    .expect("cell");
    let options = BoxplotOptions {
        group_order: Some(vec!["a".to_string(), "b".to_string()]),
        labels: Some(vec!["A".to_string(), "B".to_string()]),
        ..Default::default()
    };
    let boxes = build_box_stats(
        tensor(vec![1.0, 10.0, 3.0, 20.0], vec![4, 1]),
        Some(Value::Cell(labels)),
        &options,
    )
    .expect("stats");
    assert_eq!(boxes.len(), 2);
    assert_eq!(boxes[0].label, "A");
    assert_eq!(boxes[1].label, "B");
    assert!((boxes[0].median - 15.0).abs() < 1.0e-12);
    assert!((boxes[1].median - 2.0).abs() < 1.0e-12);
}

#[test]
fn matrix_grouping_accepts_row_group_vector() {
    let labels = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row("low")),
            Value::CharArray(CharArray::new_row("high")),
        ],
        2,
        1,
    )
    .expect("cell");
    let boxes = build_box_stats(
        tensor(vec![1.0, 10.0, 2.0, 20.0], vec![2, 2]),
        Some(Value::Cell(labels)),
        &BoxplotOptions::default(),
    )
    .expect("stats");
    assert_eq!(boxes.len(), 2);
    assert_eq!(boxes[0].label, "low");
    assert_eq!(boxes[1].label, "high");
    assert!((boxes[0].median - 1.5).abs() < 1.0e-12);
    assert!((boxes[1].median - 15.0).abs() < 1.0e-12);
}

#[test]
fn reads_typed_integer_data_groups_and_options_exactly() {
    let options = BoxplotOptions::parse(vec![
        Value::CharArray(CharArray::new_row("Colors")),
        int_tensor(IntegerStorage::U8(vec![0, 1, 0]), vec![1, 3]),
        Value::CharArray(CharArray::new_row("Positions")),
        int_tensor(IntegerStorage::I16(vec![5, 9]), vec![1, 2]),
    ])
    .expect("options");
    let boxes = build_box_stats(
        int_tensor(IntegerStorage::I16(vec![1, 10, 3, 20]), vec![4, 1]),
        Some(int_tensor(
            IntegerStorage::U16(vec![2, 1, 2, 1]),
            vec![4, 1],
        )),
        &options,
    )
    .expect("stats");
    assert_eq!(boxes.len(), 2);
    assert_eq!(boxes[0].label, "1");
    assert_eq!(boxes[0].position, 5.0);
    assert!((boxes[0].median - 15.0).abs() < 1.0e-12);
    assert_eq!(boxes[1].label, "2");
    assert_eq!(boxes[1].position, 9.0);
    assert!((boxes[1].median - 2.0).abs() < 1.0e-12);
}

#[test]
fn scalar_group_label_reads_typed_integer_storage_without_double_mirror() {
    let tensor = Tensor::new_integer(IntegerStorage::U16(vec![42]), vec![1, 1]).unwrap();

    let Some(GroupLabel::Integer(value)) = scalar_group_label(&Value::Tensor(tensor)).unwrap()
    else {
        panic!("expected exact integer label");
    };
    assert_eq!(value, 42);
}

#[test]
fn option_parser_accepts_documented_extreme_and_label_values() {
    let options = BoxplotOptions::parse(vec![
        Value::CharArray(CharArray::new_row("DataLim")),
        Value::Tensor(
            Tensor::new(vec![f64::NEG_INFINITY, f64::INFINITY], vec![1, 2]).expect("tensor"),
        ),
        Value::CharArray(CharArray::new_row("ExtremeMode")),
        Value::CharArray(CharArray::new_row("compress")),
        Value::CharArray(CharArray::new_row("Jitter")),
        Value::Num(0.0),
        Value::CharArray(CharArray::new_row("Symbol")),
        Value::CharArray(CharArray::new_row("square")),
        Value::CharArray(CharArray::new_row("LabelVerbosity")),
        Value::CharArray(CharArray::new_row("majorminor")),
    ])
    .expect("options");
    assert_eq!(options.data_lim, None);
    assert_eq!(options.extreme_mode, ExtremeMode::Compress);
    assert_eq!(options.jitter, 0.0);
    assert_eq!(options.outlier_marker, Some(MarkerStyle::Square));
}

#[test]
fn jitter_zero_keeps_outliers_centered_and_datalim_clips() {
    let options = BoxplotOptions {
        data_lim: Some((0.0, 10.0)),
        jitter: 0.0,
        ..Default::default()
    };
    let stats = BoxStats {
        label: "A".to_string(),
        position: 3.0,
        width: 0.5,
        color: Vec4::ONE,
        q1: 1.0,
        median: 2.0,
        q3: 3.0,
        lower_whisker: 1.0,
        upper_whisker: 3.0,
        outliers: vec![-5.0, 12.0],
    };
    let (x, y) = outlier_points(&stats, &options);
    assert_eq!(x, vec![3.0, 3.0]);
    assert_eq!(y, vec![0.0, 10.0]);
}

#[test]
fn factor_options_validate_known_values() {
    assert!(BoxplotOptions::parse(vec![
        Value::CharArray(CharArray::new_row("FactorDirection")),
        Value::CharArray(CharArray::new_row("data")),
        Value::CharArray(CharArray::new_row("FullFactors")),
        Value::CharArray(CharArray::new_row("off")),
        Value::CharArray(CharArray::new_row("FactorGap")),
        Value::Num(2.0),
        Value::CharArray(CharArray::new_row("FactorSeparator")),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor")),
    ])
    .is_ok());
    assert!(BoxplotOptions::parse(vec![
        Value::CharArray(CharArray::new_row("FactorDirection")),
        Value::CharArray(CharArray::new_row("sideways")),
    ])
    .is_err());
}

#[test]
fn renders_handle_vector_and_tick_labels() {
    let _guard = crate::builtins::plotting::lock_plot_test_context();
    crate::builtins::plotting::reset_plot_state();
    let options = BoxplotOptions::default();
    let boxes = build_box_stats(tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]), None, &options)
        .expect("stats");
    let handles = render_boxplot(&boxes, &options).expect("render");
    assert_eq!(handles.len(), 6);
    let figure = clone_figure(current_figure_handle()).expect("figure");
    let meta = figure.axes_metadata(0).expect("axes metadata");
    assert_eq!(meta.x_tick_labels.as_deref(), Some(&["1".to_string()][..]));
}
