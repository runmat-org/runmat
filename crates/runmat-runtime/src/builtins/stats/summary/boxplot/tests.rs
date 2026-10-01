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

    let ordered = BoxplotOptions {
        group_order: Some(vec![wide.to_string()]),
        ..Default::default()
    };
    let error = build_box_stats(
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Some(Value::Cell(
            CellArray::new(
                vec![
                    Value::Int(IntValue::U64(wide)),
                    Value::String(wide.to_string()),
                ],
                2,
                1,
            )
            .unwrap(),
        )),
        &ordered,
    )
    .expect_err("public text cannot resolve two distinct typed keys with one label");
    assert_eq!(
        error.identifier(),
        BOXPLOT_ERROR_INVALID_ARGUMENT.identifier
    );
}

#[test]
fn character_group_rows_follow_the_canonical_row_major_payload() {
    let row_major = CharArray::new(vec!['A', '1', 'B', '2'], 2, 2).unwrap();
    let column_major = CharArray::from_column_major(vec!['A', 'B', '1', '2'], vec![2, 2]).unwrap();
    assert_eq!(groups::char_rows(&row_major), vec!["A1", "B2"]);
    assert_eq!(groups::char_rows(&column_major), vec!["A1", "B2"]);
    for group in [row_major, column_major] {
        let boxes = build_box_stats(
            tensor(vec![1.0, 2.0], vec![2, 1]),
            Some(Value::CharArray(group)),
            &BoxplotOptions::default(),
        )
        .unwrap();
        assert_eq!(
            boxes.iter().map(|b| b.label.as_str()).collect::<Vec<_>>(),
            vec!["A1", "B2"]
        );
    }
}

#[test]
fn categorical_group_length_uses_all_codes_not_only_sample_prefix() {
    let mut group =
        runmat_value::ObjectInstance::new(runmat_types::standard::CATEGORICAL.to_string());
    group
        .properties
        .insert("Codes".to_string(), tensor(vec![1.0, 2.0, 1.0], vec![3, 1]));
    group.properties.insert(
        "Categories".to_string(),
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
    );
    let sample = tensor(vec![1.0, 2.0], vec![2, 1]);
    let error = admission::admit_host_visible(&sample, Some(&Value::Object(group.clone())))
        .expect_err("categorical group with extra code rejects at admission");
    assert_eq!(
        error.identifier(),
        BOXPLOT_ERROR_INVALID_ARGUMENT.identifier
    );
    let error = build_box_stats(
        sample,
        Some(Value::Object(group)),
        &BoxplotOptions::default(),
    )
    .expect_err("categorical group with extra code rejects at execution");
    assert_eq!(
        error.identifier(),
        BOXPLOT_ERROR_INVALID_ARGUMENT.identifier
    );
}

#[test]
fn owned_host_sample_passes_materialization_boundary_without_copy() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let original_payload = tensor.as_f64_slice().unwrap().as_ptr();
    let mut values = block_on(gather_values(vec![Value::Tensor(tensor)])).unwrap();
    let Value::Tensor(result) = values.remove(0) else {
        panic!("host tensor must retain its value kind");
    };
    assert_eq!(result.as_f64_slice().unwrap().as_ptr(), original_payload);
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
fn axis_limits_remain_finite_for_extreme_and_constant_finite_data() {
    let mut stats = BoxStats {
        label: "A".into(),
        position: 1.0,
        width: 0.5,
        color: Vec4::ONE,
        q1: -f64::MAX,
        median: 0.0,
        q3: f64::MAX,
        lower_whisker: -f64::MAX,
        upper_whisker: f64::MAX,
        outliers: Vec::new(),
        observation_count: 4,
    };
    let (_, data) = super::plot::axis_limits(&[stats.clone()], &BoxplotOptions::default())
        .expect("opposite finite extrema retain finite bounds");
    assert_eq!(data, Some((-f64::MAX, f64::MAX)));

    stats.q1 = f64::MAX;
    stats.median = f64::MAX;
    stats.lower_whisker = f64::MAX;
    let (_, data) = super::plot::axis_limits(&[stats.clone()], &BoxplotOptions::default())
        .expect("constant maximum has a representable neighboring bound");
    assert_eq!(data, Some((f64::MAX.next_down(), f64::MAX)));

    stats.q1 = -f64::MAX;
    stats.median = -f64::MAX;
    stats.q3 = -f64::MAX;
    stats.lower_whisker = -f64::MAX;
    stats.upper_whisker = -f64::MAX;
    let (_, data) = super::plot::axis_limits(&[stats.clone()], &BoxplotOptions::default())
        .expect("constant negative maximum has a representable neighboring bound");
    assert_eq!(data, Some((-f64::MAX, (-f64::MAX).next_up())));

    stats.position = f64::MAX;
    stats.width = f64::MAX;
    let error = super::plot::axis_limits(&[stats], &BoxplotOptions::default())
        .expect_err("unrepresentable category span rejects before graphics effects");
    assert_eq!(
        error.identifier(),
        BOXPLOT_ERROR_INVALID_ARGUMENT.identifier
    );
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

    let Some(label) = scalar_group_label(&Value::Tensor(tensor)).unwrap() else {
        panic!("expected exact integer label");
    };
    assert!(matches!(
        label.as_slice(),
        [crate::builtins::common::grouping_keys::KeyAtom::Integer(
            IntValue::U16(42)
        )]
    ));
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
        observation_count: 4,
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

#[test]
fn appearance_options_change_typed_plot_plans_and_interval_geometry() {
    let options = BoxplotOptions::parse(vec![
        Value::from("BoxStyle"),
        Value::from("filled"),
        Value::from("MedianStyle"),
        Value::from("target"),
        Value::from("Notch"),
        Value::from("marker"),
    ])
    .unwrap();
    assert_eq!(options.box_style, BoxStyle::Filled);
    assert_eq!(options.median_style, MedianStyle::Target);
    assert_eq!(options.notch_style, NotchStyle::Marker);
    let boxes =
        build_box_stats(tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]), None, &options).unwrap();
    let plans = super::plot::build_plot_plans(&boxes, &options).unwrap();
    assert_eq!(plans.len(), 10);
    assert_eq!(
        plans
            .iter()
            .filter(|p| matches!(p, PlotPlan::Patch(_)))
            .count(),
        1
    );
    assert_eq!(
        plans
            .iter()
            .filter(|p| matches!(p, PlotPlan::Scatter(_)))
            .count(),
        4
    );
    let (lower, upper) = appearance::notch_interval(&boxes[0]).unwrap();
    assert!((lower - 1.3225).abs() < 1.0e-12);
    assert!((upper - 3.6775).abs() < 1.0e-12);
    let (_, data) = super::plot::axis_limits(&boxes, &options).unwrap();
    let (min, max) = data.unwrap();
    assert!(min <= lower && max >= upper);
    let _guard = crate::builtins::plotting::lock_plot_test_context();
    crate::builtins::plotting::reset_plot_state();
    let handles = render_boxplot(&boxes, &options).unwrap();
    assert_eq!(handles.len(), plans.len());
    assert!(handles.iter().all(|handle| handle.is_finite()));
}

#[test]
fn plot_style_controls_defaults_but_later_explicit_appearance_wins() {
    let compact =
        BoxplotOptions::parse(vec![Value::from("PlotStyle"), Value::from("compact")]).unwrap();
    assert_eq!(compact.box_style, BoxStyle::Filled);
    assert_eq!(compact.median_style, MedianStyle::Target);
    assert_eq!(compact.label_orientation, LabelOrientation::Inline);
    assert_eq!(
        NotchStyle::On.effective(compact.compact),
        NotchStyle::Marker
    );
    let stats = statistics::stats_for_values("A".into(), vec![1.0, 2.0, 3.0, 4.0], 1.5).unwrap();
    let (outline_x, outline_y) = appearance::box_outline(&stats, NotchStyle::On).unwrap();
    assert_eq!(outline_x.len(), 11);
    assert_eq!(outline_y.len(), outline_x.len());
    assert_eq!(outline_x.first(), outline_x.last());
    assert_eq!(outline_y.first(), outline_y.last());
    let explicit = BoxplotOptions::parse(vec![
        Value::from("PlotStyle"),
        Value::from("compact"),
        Value::from("BoxStyle"),
        Value::from("outline"),
        Value::from("MedianStyle"),
        Value::from("line"),
    ])
    .unwrap();
    assert_eq!(explicit.box_style, BoxStyle::Outline);
    assert_eq!(explicit.median_style, MedianStyle::Line);
    let reversed = BoxplotOptions::parse(vec![
        Value::from("BoxStyle"),
        Value::from("outline"),
        Value::from("MedianStyle"),
        Value::from("line"),
        Value::from("PlotStyle"),
        Value::from("compact"),
    ])
    .unwrap();
    assert_eq!(reversed.box_style, BoxStyle::Outline);
    assert_eq!(reversed.median_style, MedianStyle::Line);
}

#[test]
fn label_orientation_changes_axes_rotation_without_changing_horizontal_labels() {
    let _guard = crate::builtins::plotting::lock_plot_test_context();
    let options =
        BoxplotOptions::parse(vec![Value::from("LabelOrientation"), Value::from("inline")])
            .unwrap();
    let boxes = build_box_stats(tensor(vec![1.0, 2.0, 3.0], vec![3, 1]), None, &options).unwrap();
    crate::builtins::plotting::reset_plot_state();
    render_boxplot(&boxes, &options).unwrap();
    let figure = clone_figure(current_figure_handle()).unwrap();
    assert_eq!(
        figure.axes_metadata(0).unwrap().x_tick_label_rotation,
        Some(90.0)
    );

    let horizontal = BoxplotOptions::parse(vec![
        Value::from("LabelOrientation"),
        Value::from("inline"),
        Value::from("Orientation"),
        Value::from("horizontal"),
    ])
    .unwrap();
    crate::builtins::plotting::reset_plot_state();
    render_boxplot(&boxes, &horizontal).unwrap();
    let figure = clone_figure(current_figure_handle()).unwrap();
    assert_eq!(
        figure.axes_metadata(0).unwrap().y_tick_label_rotation,
        Some(0.0)
    );
}

#[test]
fn unrepresentable_filled_vertices_reject_before_plot_state_changes() {
    let _guard = crate::builtins::plotting::lock_plot_test_context();
    crate::builtins::plotting::reset_plot_state();
    let options =
        BoxplotOptions::parse(vec![Value::from("BoxStyle"), Value::from("filled")]).unwrap();
    let boxes = build_box_stats(
        tensor(vec![f64::MAX, f64::MAX, f64::MAX], vec![3, 1]),
        None,
        &options,
    )
    .unwrap();
    let before = clone_figure(current_figure_handle()).unwrap();
    assert_eq!(before.plots().count(), 0);
    let error = render_boxplot(&boxes, &options)
        .expect_err("f32 patch boundary must fail before publication");
    assert_eq!(
        error.identifier(),
        BOXPLOT_ERROR_INVALID_ARGUMENT.identifier
    );
    let after = clone_figure(current_figure_handle()).unwrap();
    assert_eq!(after.plots().count(), 0);
}
