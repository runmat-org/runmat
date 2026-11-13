//! Scatterhist value, rendering, and admission regressions.
use super::*;
use crate::builtins::plotting::state::{current_figure_handle, decode_axes_handle};
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{clone_figure, reset_hold_state_for_run};
use runmat_plot::plots::PlotElement;
use runmat_value::{CellArray, IntegerStorage, StringArray};

fn vec_tensor(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn numeric_control_rank_boundaries_preserve_bandwidth_and_rgb_admission() {
    let scalar = Value::Tensor(Tensor::new(vec![0.5], Vec::new()).unwrap());
    assert_eq!(parse_bandwidth_cycle(&scalar).unwrap(), vec![[0.5, 0.5]]);
    assert!(parse_color_cycle(&scalar)
        .expect_err("rank-zero Color has no RGB columns")
        .message()
        .contains("RGB"));

    let bandwidth = Value::Tensor(Tensor::new(vec![0.5, 1.0], vec![2]).unwrap());
    assert_eq!(parse_bandwidth_cycle(&bandwidth).unwrap(), vec![[0.5, 1.0]]);
    let color = Value::Tensor(Tensor::new(vec![0.25, 0.5, 0.75], vec![3]).unwrap());
    assert_eq!(
        parse_color_cycle(&color).unwrap(),
        vec![Vec4::new(0.25, 0.5, 0.75, 1.0)]
    );
}

#[test]
fn categorical_group_must_match_full_observation_count() {
    let labels = Value::StringArray(
        StringArray::new(vec!["A".into(), "B".into(), "A".into()], vec![3, 1]).unwrap(),
    );
    let categorical = crate::builtins::table::categorical_from_args(
        crate::call::FunctionArgumentView::from_slice(&[labels]),
    )
    .unwrap();
    assert_eq!(
        group_labels(&categorical, 3).unwrap(),
        vec![
            Some(KeyAtom::Text("A".into())),
            Some(KeyAtom::Text("B".into())),
            Some(KeyAtom::Text("A".into()))
        ]
    );
    for count in [2, 4] {
        let error = group_labels(&categorical, count).expect_err("full category length must match");
        assert_eq!(
            error.identifier(),
            SCATTERHIST_ERROR_INVALID_ARGUMENT.identifier
        );
    }
}

#[test]
fn character_group_rows_follow_character_payload_layout() {
    let labels = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars(vec!['A', '1', 'B', '2'], 2, 2).unwrap(),
    );
    assert_eq!(
        group_labels(&labels, 2).unwrap(),
        vec![
            Some(KeyAtom::Text("A1".into())),
            Some(KeyAtom::Text("B2".into()))
        ]
    );
}

#[test]
fn scalar_and_logical_group_values_match_catalog_admission() {
    assert_eq!(
        group_labels(&Value::Num(1.0), 1).unwrap(),
        vec![Some(KeyAtom::Number(1.0))]
    );
    assert_eq!(
        group_labels(&Value::Int(IntValue::U64(1)), 1).unwrap(),
        vec![Some(KeyAtom::Integer(IntValue::U64(1)))]
    );
    assert_eq!(
        group_labels(&Value::Bool(true), 1).unwrap(),
        vec![Some(KeyAtom::Logical(true))]
    );
    assert_eq!(
        group_labels(&Value::String("one".into()), 1).unwrap(),
        vec![Some(KeyAtom::Text("one".into()))]
    );
    let logical =
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![1, 0, 1, 0], vec![4, 1]).unwrap());
    assert_eq!(
        group_labels(&logical, 4).unwrap(),
        vec![
            Some(KeyAtom::Logical(true)),
            Some(KeyAtom::Logical(false)),
            Some(KeyAtom::Logical(true)),
            Some(KeyAtom::Logical(false))
        ]
    );
    assert!(group_labels(&Value::Num(1.0), 2).is_err());
}

#[test]
fn multirow_character_option_name_is_rejected_before_chart_effects() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let figure = current_figure_handle();
    let before = clone_figure(figure).unwrap().plots().count();
    let name = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars(vec!['N', 'B', 'i', 'n', 's', ' '], 2, 3)
            .unwrap(),
    );
    let err = futures::executor::block_on(scatterhist_builtin(vec![
        vec_tensor(&[1.0, 2.0]),
        vec_tensor(&[2.0, 1.0]),
        name,
        Value::Num(2.0),
    ]))
    .expect_err("multirow option name cannot concatenate into NBins");
    assert_eq!(
        err.identifier(),
        SCATTERHIST_ERROR_INVALID_ARGUMENT.identifier
    );
    assert_eq!(clone_figure(figure).unwrap().plots().count(), before);
}

#[test]
fn scalar_string_array_option_values_decode_once_as_text() {
    let scalar =
        |text: &str| Value::StringArray(StringArray::new(vec![text.into()], vec![1, 1]).unwrap());
    assert_eq!(
        parse_kernel(&scalar("overlay")).unwrap(),
        KernelMode::Overlay
    );
    assert!(option_bool(&scalar("on"), "Legend").unwrap());
    assert_eq!(
        parse_style(&scalar("stairs")).unwrap(),
        (KernelMode::Off, HistStyle::Stairs)
    );
    assert_eq!(
        parse_location(&scalar("northeast")).unwrap(),
        Location::NorthEast
    );
    assert_eq!(parse_direction(&scalar("out")).unwrap(), Direction::Out);

    let multiple =
        Value::StringArray(StringArray::new(vec!["on".into(), "off".into()], vec![2, 1]).unwrap());
    assert!(parse_kernel(&multiple).is_err());
    assert!(option_bool(&multiple, "Legend").is_err());
    assert!(parse_style(&multiple).is_err());
    assert!(parse_location(&multiple).is_err());
    assert!(parse_direction(&multiple).is_err());
}

#[test]
fn automatic_kde_bandwidth_and_density_retain_large_and_small_finite_scales() {
    for scale in [1e200, 1e-200] {
        let samples = [scale, 2.0 * scale];
        let bandwidth = scott_bandwidth(&samples);
        assert!(bandwidth.is_finite() && bandwidth > 0.0, "{bandwidth}");
        let edges = [scale, 1.5 * scale, 2.0 * scale];
        let (centers, density) = kde_line(&samples, &edges, None).unwrap();
        assert_eq!(centers.len(), 2);
        assert!(density
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
    }

    let base = 1e200_f64;
    let next = f64::from_bits(base.to_bits() + 1);
    let expected = (next - base) / 2.0_f64.sqrt() * 1.06 * 2.0_f64.powf(-0.2);
    let actual = scott_bandwidth(&[base, next]);
    assert!(actual.is_finite() && actual > 0.0);
    assert!(((actual - expected) / expected).abs() < 1e-12);
    assert!(scott_bandwidth(&[-1e308, 1e308]).is_finite());
}

#[test]
fn singleton_automatic_kde_retains_existing_scott_bandwidth() {
    assert_eq!(scott_bandwidth(&[4.0]), 1.06);
    let (_, density) = kde_line(&[4.0], &[3.0, 4.0, 5.0], None).unwrap();
    assert_eq!(density.len(), 2);
    assert!(density
        .iter()
        .all(|value| value.is_finite() && *value > 0.0));
}

#[test]
fn kde_keeps_count_scaled_density_and_handles_tiny_bandwidth() {
    let (_, density) = kde_line(&[0.0, 0.0], &[-1.0, 0.0, 1.0], Some(1.0)).unwrap();
    let expected = 4.0 * (-0.125_f64).exp() / (2.0 * std::f64::consts::PI).sqrt();
    assert!((density[0] - expected).abs() < 1e-12);
    assert!((density[1] - expected).abs() < 1e-12);

    let (_, distant) = kde_line(&[0.0, 0.0], &[-1.0, 0.0, 1.0], Some(f64::MIN_POSITIVE)).unwrap();
    assert_eq!(distant, vec![0.0, 0.0]);
    assert!(kde_line(
        &[0.0, 0.0],
        &[-f64::from_bits(1), f64::from_bits(1)],
        Some(f64::from_bits(1)),
    )
    .is_err());
    let tiny = f64::from_bits(1);
    let (_, tail) = kde_line(&[0.0], &[37.0 * tiny, 39.0 * tiny], Some(tiny)).unwrap();
    assert!(tail[0].is_finite() && tail[0] > 0.0);
}

#[test]
fn unrepresentable_kde_geometry_fails_before_figure_mutation() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let figure = current_figure_handle();
    let before = clone_figure(figure).unwrap().plots().count();
    let options = ScatterhistOptions {
        kernel: KernelMode::On,
        bins: Some([2, 2]),
        ..Default::default()
    };
    let series = grouped_series(&[-f64::MAX, f64::MAX], &[-f64::MAX, f64::MAX], &options).unwrap();
    let error = render_scatterhist(&series, &options).expect_err("bandwidth exceeds f64 range");
    assert_eq!(
        error.identifier(),
        SCATTERHIST_ERROR_INVALID_ARGUMENT.identifier
    );
    assert_eq!(clone_figure(figure).unwrap().plots().count(), before);
}

#[test]
fn mixed_cell_groups_keep_typed_identity_until_rendering() {
    let labels = Value::Cell(
        CellArray::new(
            vec![
                Value::Int(IntValue::U64(1)),
                Value::String("1".into()),
                Value::String(String::new().into()),
                Value::String("Data".into()),
            ],
            4,
            1,
        )
        .unwrap(),
    );
    let group = group_labels(&labels, 4).unwrap();
    let mut options = ScatterhistOptions {
        group: Some(group),
        ..Default::default()
    };
    let series = grouped_series(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0], &options).unwrap();
    assert_eq!(series.len(), 4);
    assert_eq!(
        series
            .iter()
            .map(|item| item.scatter_x[0])
            .collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        series
            .iter()
            .map(|item| item.label.as_str())
            .collect::<Vec<_>>(),
        vec!["1", "1", "", "Data"]
    );
    options.group = None;
    let ungrouped = grouped_series(&[1.0], &[2.0], &options).unwrap();
    assert_eq!(ungrouped.len(), 1);
    assert_eq!(ungrouped[0].label, "Data");
}

#[test]
fn wide_integer_groups_do_not_round_or_merge_with_text() {
    let wide = (1u64 << 53) + 1;
    let keys = group_labels(
        &Value::Cell(
            CellArray::new(
                vec![
                    Value::Int(IntValue::U64(wide)),
                    Value::String(wide.to_string().into()),
                ],
                2,
                1,
            )
            .unwrap(),
        ),
        2,
    )
    .unwrap();
    assert_eq!(
        keys,
        vec![
            Some(KeyAtom::Integer(IntValue::U64(wide))),
            Some(KeyAtom::Text(wide.to_string()))
        ]
    );
    let series = grouped_series(
        &[1.0, 2.0],
        &[2.0, 1.0],
        &ScatterhistOptions {
            group: Some(keys),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(series.len(), 2);
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn first_unrepresentable_usize_double() -> f64 {
    if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    }
}

#[test]
fn histogram_boundaries_remain_finite_and_distinct_before_rendering() {
    let extremes = bin_edges((-f64::MAX, f64::MAX), 2).unwrap();
    assert_eq!(extremes[0], -f64::MAX);
    assert_eq!(extremes[2], f64::MAX);
    assert_eq!(extremes[1], 0.0);
    assert!(extremes.iter().all(|edge| edge.is_finite()));

    let positive = combined_limits([f64::MAX].into_iter()).unwrap();
    assert_eq!(positive.1, f64::MAX);
    assert!(positive.0.is_finite() && positive.0 < positive.1);
    let negative = combined_limits([-f64::MAX].into_iter()).unwrap();
    assert_eq!(negative.0, -f64::MAX);
    assert!(negative.1.is_finite() && negative.0 < negative.1);

    let next = f64::from_bits(1.0f64.to_bits() + 1);
    assert!(bin_edges((1.0, next), 2)
        .unwrap_err()
        .message
        .contains("not distinct"));
}

#[test]
fn unrepresentable_histogram_geometry_rejects_before_plot_effects() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let figure = current_figure_handle();
    let before = clone_figure(figure).unwrap().plots().count();
    let next = f64::from_bits(1.0f64.to_bits() + 1);
    let err = futures::executor::block_on(scatterhist_builtin(vec![
        vec_tensor(&[1.0, next]),
        vec_tensor(&[0.0, 1.0]),
        Value::String("NBins".into()),
        Value::Num(2.0),
    ]))
    .expect_err("bin boundaries should be unrepresentable");
    assert!(err.message.contains("not distinct"), "{}", err.message);
    assert_eq!(current_figure_handle(), figure);
    assert_eq!(clone_figure(figure).unwrap().plots().count(), before);
}

#[test]
fn scatterhist_numeric_parsers_read_typed_integer_storage_exactly() {
    let wide = u64::MAX - 1;
    assert_eq!(
        numeric_vector(
            int_tensor(IntegerStorage::U64(vec![wide, wide - 1]), vec![2, 1]),
            "x",
        )
        .unwrap(),
        vec![
            IntValue::U64(wide).to_f64(),
            IntValue::U64(wide - 1).to_f64()
        ]
    );
    assert_eq!(
        parse_bins(&int_tensor(IntegerStorage::U16(vec![12, 8]), vec![1, 2])).unwrap(),
        [12, 8]
    );
    assert!(option_bool(
        &int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
        "Legend",
    )
    .unwrap());
    assert_eq!(
        nonnegative_scalar_cycle(
            &int_tensor(IntegerStorage::U16(vec![4, 5]), vec![1, 2]),
            "MarkerSize",
        )
        .unwrap(),
        vec![4.0, 5.0]
    );
    assert_eq!(
        group_labels(
            &int_tensor(IntegerStorage::U64(vec![wide, wide - 1]), vec![2, 1]),
            2,
        )
        .unwrap(),
        vec![
            Some(KeyAtom::Integer(IntValue::U64(wide))),
            Some(KeyAtom::Integer(IntValue::U64(wide - 1)))
        ]
    );
}

#[test]
fn scatterhist_rejects_invalid_typed_integer_options() {
    let err = parse_bins(&int_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1])).unwrap_err();
    assert!(
        err.message.contains("greater than or equal to 2"),
        "{}",
        err.message
    );

    let err = option_bool(
        &int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
        "Legend",
    )
    .unwrap_err();
    assert!(err.message.contains("0 or 1"), "{}", err.message);
}

#[test]
fn bandwidth_uses_documented_two_by_k_and_retains_unambiguous_legacy_layout() {
    let documented =
        Value::Tensor(Tensor::new(vec![0.5, 0.15, 0.2, 0.25, 0.1, 0.35], vec![2, 3]).unwrap());
    assert_eq!(
        parse_bandwidth_cycle(&documented).unwrap(),
        vec![[0.5, 0.15], [0.2, 0.25], [0.1, 0.35]]
    );
    let legacy =
        Value::Tensor(Tensor::new(vec![0.5, 0.2, 0.1, 0.15, 0.25, 0.35], vec![3, 2]).unwrap());
    assert_eq!(
        parse_bandwidth_cycle(&legacy).unwrap(),
        vec![[0.5, 0.15], [0.2, 0.25], [0.1, 0.35]]
    );
    let square = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    assert_eq!(
        parse_bandwidth_cycle(&square).unwrap(),
        vec![[1.0, 2.0], [3.0, 4.0]]
    );
}

#[test]
fn scatterhist_rejects_unrepresentable_double_bin_counts_before_casting() {
    let err = parse_bins(&vec_tensor(&[first_unrepresentable_usize_double()])).unwrap_err();
    assert!(
        err.message.contains("no greater than 250"),
        "{}",
        err.message
    );
}

#[test]
fn scatterhist_accepts_typed_integer_data_and_options() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();

    let _ = futures::executor::block_on(scatterhist_builtin(vec![
        int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), vec![4, 1]),
        int_tensor(IntegerStorage::I16(vec![3, 2, 1, 0]), vec![4, 1]),
        Value::String("NBins".into()),
        int_tensor(IntegerStorage::U16(vec![2, 2]), vec![1, 2]),
        Value::String("Group".into()),
        int_tensor(IntegerStorage::U8(vec![1, 2, 1, 2]), vec![4, 1]),
        Value::String("MarkerSize".into()),
        int_tensor(IntegerStorage::U16(vec![4, 5]), vec![1, 2]),
        Value::String("LineWidth".into()),
        int_tensor(IntegerStorage::U16(vec![2, 3]), vec![1, 2]),
        Value::String("Color".into()),
        int_tensor(IntegerStorage::U8(vec![1, 0, 0, 0, 1, 0]), vec![2, 3]),
        Value::String("Legend".into()),
        int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
    ]))
    .unwrap();

    let figure = clone_figure(current_figure_handle()).unwrap();
    let scatter_plots = figure
        .plots()
        .filter_map(|plot| match plot {
            PlotElement::Scatter(scatter) => Some(scatter),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(scatter_plots.len(), 2);
    assert_eq!(scatter_plots[0].marker_size, 4.0);
    assert_eq!(scatter_plots[1].marker_size, 5.0);
    assert!(figure.axes_metadata(1).unwrap().legend_enabled);
}

#[test]
fn scatterhist_renders_three_axes_with_histograms() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();

    let handles = futures::executor::block_on(scatterhist_builtin(vec![
        vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
        vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
        Value::String("NBins".into()),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
    ]))
    .unwrap();
    let Value::Tensor(handles) = handles else {
        panic!("expected handle tensor");
    };
    assert_eq!(handles.shape(), &[3, 1]);
    let decoded = handles
        .materialize_f64()
        .iter()
        .map(|handle| decode_axes_handle(*handle).unwrap().1)
        .collect::<Vec<_>>();
    assert_eq!(decoded, vec![1, 3, 0]);

    let figure = clone_figure(current_figure_handle()).unwrap();
    assert_eq!(figure.axes_grid(), (2, 2));
    assert_eq!(figure.len(), 3);
    assert!(matches!(
        figure.plots().next().unwrap(),
        PlotElement::Scatter(_)
    ));
    let bars = figure
        .plots()
        .filter(|plot| matches!(plot, PlotElement::Bar(_)))
        .count();
    assert_eq!(bars, 2);
    for bar in figure.plots().filter_map(|plot| match plot {
        PlotElement::Bar(bar) => Some(bar),
        _ => None,
    }) {
        assert_eq!(bar.values().expect("host histogram values"), &[2.0, 2.0]);
        assert_eq!(
            bar.histogram_bin_edges().expect("histogram edges"),
            &[0.0, 0.5, 1.0]
        );
    }
    let axes = figure.plot_axes_indices();
    assert_eq!(axes, &[1, 3, 0]);
}

#[test]
fn scatterhist_documented_group_overlay_has_two_series_and_both_marginal_layers() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();

    let groups = StringArray::new(
        vec!["a".into(), "b".into(), "a".into(), "b".into()],
        vec![4, 1],
    )
    .unwrap();
    let handles = futures::executor::block_on(scatterhist_builtin(vec![
        vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
        vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
        Value::String("Group".into()),
        Value::StringArray(groups),
        Value::String("Kernel".into()),
        Value::String("overlay".into()),
        Value::String("Legend".into()),
        Value::String("on".into()),
    ]))
    .unwrap();
    let Value::Tensor(handles) = handles else {
        panic!("expected axes handle tensor");
    };
    assert_eq!(handles.shape(), &[3, 1]);

    let figure = clone_figure(current_figure_handle()).unwrap();
    assert_eq!(figure.axes_grid(), (2, 2));
    // Two scatter series, then four overlay lines on each marginal axis.
    assert_eq!(figure.plot_axes_indices(), &[1, 1, 3, 3, 3, 3, 0, 0, 0, 0]);
    let scatter = figure
        .plots()
        .filter_map(|plot| match plot {
            PlotElement::Scatter(scatter) => Some(scatter),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(scatter.len(), 2);
    assert_eq!(scatter[0].label.as_deref(), Some("a"));
    assert_eq!(scatter[1].label.as_deref(), Some("b"));
    assert_eq!(
        scatter[0].host_xy_f64().unwrap(),
        Some((vec![0.0, 0.8], vec![0.0, 0.9]))
    );
    assert_eq!(
        scatter[1].host_xy_f64().unwrap(),
        Some((vec![0.2, 1.0], vec![0.1, 1.0]))
    );
    // Each of two groups has a density line and a stair-count line on both marginals.
    let lines = figure
        .plots()
        .filter_map(|plot| match plot {
            PlotElement::Line(line) => Some(line),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 8);
    for (index, line) in lines.iter().enumerate() {
        let group = (index % 4) / 2;
        assert_eq!(line.label, scatter[group].label);
        assert_eq!(line.color, scatter[group].color);
        let (x, y) = line.host_xy_f64().unwrap().expect("host marginal line");
        assert_eq!(x.len(), y.len());
        assert!(x.len() >= 2);
        assert!(x.iter().chain(y.iter()).all(|value| value.is_finite()));
    }
    assert!(figure.axes_metadata(1).unwrap().legend_enabled);
}

#[test]
fn scatterhist_groups_and_kernel_lines() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();

    let groups = StringArray::new(
        vec!["a".into(), "b".into(), "a".into(), "b".into()],
        vec![4, 1],
    )
    .unwrap();
    let _ = futures::executor::block_on(scatterhist_builtin(vec![
        vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
        vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
        Value::String("Group".into()),
        Value::StringArray(groups),
        Value::String("Kernel".into()),
        Value::String("on".into()),
    ]))
    .unwrap();

    let figure = clone_figure(current_figure_handle()).unwrap();
    let scatter_count = figure
        .plots()
        .filter(|plot| matches!(plot, PlotElement::Scatter(_)))
        .count();
    let line_count = figure
        .plots()
        .filter(|plot| matches!(plot, PlotElement::Line(_)))
        .count();
    assert_eq!(scatter_count, 2);
    assert_eq!(line_count, 4);
    assert!(figure.axes_metadata(1).unwrap().legend_enabled);
}

#[test]
fn scatterhist_grouped_defaults_to_stairs_and_cycles_styles() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();

    let groups = StringArray::new(
        vec!["a".into(), "b".into(), "a".into(), "b".into()],
        vec![4, 1],
    )
    .unwrap();
    let line_styles = CellArray::new(
        vec![Value::String("-".into()), Value::String("--".into())],
        1,
        2,
    )
    .unwrap();
    let _ = futures::executor::block_on(scatterhist_builtin(vec![
        vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
        vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
        Value::String("Group".into()),
        Value::StringArray(groups),
        Value::String("Color".into()),
        Value::String("kr".into()),
        Value::String("Marker".into()),
        Value::String("o+".into()),
        Value::String("MarkerSize".into()),
        Value::Tensor(Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap()),
        Value::String("LineStyle".into()),
        Value::Cell(line_styles),
        Value::String("LineWidth".into()),
        Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
    ]))
    .unwrap();

    let figure = clone_figure(current_figure_handle()).unwrap();
    let scatter_plots = figure
        .plots()
        .filter_map(|plot| match plot {
            PlotElement::Scatter(scatter) => Some(scatter),
            _ => None,
        })
        .collect::<Vec<_>>();
    let line_plots = figure
        .plots()
        .filter_map(|plot| match plot {
            PlotElement::Line(line) => Some(line),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(scatter_plots.len(), 2);
    assert_eq!(line_plots.len(), 4);
    assert_eq!(
        figure
            .plots()
            .filter(|plot| matches!(plot, PlotElement::Bar(_)))
            .count(),
        0
    );
    assert_eq!(scatter_plots[0].color, Vec4::new(0.0, 0.0, 0.0, 1.0));
    assert_eq!(scatter_plots[0].marker_style, MarkerStyle::Circle);
    assert_eq!(scatter_plots[0].marker_size, 4.0);
    assert_eq!(scatter_plots[1].color, Vec4::new(1.0, 0.0, 0.0, 1.0));
    assert_eq!(scatter_plots[1].marker_style, MarkerStyle::Plus);
    assert_eq!(scatter_plots[1].marker_size, 5.0);
    assert_eq!(line_plots[0].line_style, LineStyle::Solid);
    assert_eq!(line_plots[0].line_width, 2.0);
    assert_eq!(line_plots[1].line_style, LineStyle::Dashed);
    assert_eq!(line_plots[1].line_width, 3.0);
}

#[test]
fn scatterhist_keeps_marginal_nan_semantics_independent() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();

    let _ = futures::executor::block_on(scatterhist_builtin(vec![
        vec_tensor(&[1.0, 2.0, f64::NAN]),
        vec_tensor(&[10.0, f64::NAN, 30.0]),
        Value::String("NBins".into()),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
    ]))
    .unwrap();

    let figure = clone_figure(current_figure_handle()).unwrap();
    let mut scatter_points = 0usize;
    let mut x_count = 0.0;
    let mut y_count = 0.0;
    for (plot, axes) in figure.plots().zip(figure.plot_axes_indices()) {
        match (plot, *axes) {
            (PlotElement::Scatter(scatter), 1) => {
                scatter_points += scatter.host_xy_f64().unwrap().unwrap().0.len()
            }
            (PlotElement::Bar(bar), 3) => {
                x_count = bar.values().unwrap().iter().map(|v| v.abs()).sum()
            }
            (PlotElement::Bar(bar), 0) => {
                y_count = bar.values().unwrap().iter().map(|v| v.abs()).sum()
            }
            _ => {}
        }
    }
    assert_eq!(scatter_points, 1);
    assert_eq!(x_count, 2.0);
    assert_eq!(y_count, 2.0);
}

#[test]
fn scatterhist_rejects_invalid_bins_and_supports_stairs_overlay_options() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();

    let err = futures::executor::block_on(scatterhist_builtin(vec![
        vec_tensor(&[0.0, 1.0]),
        vec_tensor(&[0.0, 1.0]),
        Value::String("NBins".into()),
        Value::Num(1.0),
    ]))
    .unwrap_err();
    assert!(err.message.contains("greater than or equal to 2"));

    let err = futures::executor::block_on(scatterhist_builtin(vec![
        Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap()),
        vec_tensor(&[0.0, 1.0, 2.0, 3.0]),
    ]))
    .unwrap_err();
    assert!(err.message.contains("x must be a vector"));

    let err = futures::executor::block_on(scatterhist_builtin(vec![
        vec_tensor(&[0.0, 1.0]),
        vec_tensor(&[0.0, 1.0]),
        Value::String("Style".into()),
        Value::String("overlay".into()),
    ]))
    .unwrap_err();
    assert!(err.message.contains("unsupported Style"));

    let _ = futures::executor::block_on(scatterhist_builtin(vec![
        vec_tensor(&[0.0, 0.2, 0.8, 1.0]),
        vec_tensor(&[0.0, 0.1, 0.9, 1.0]),
        Value::String("Style".into()),
        Value::String("stairs".into()),
        Value::String("LineStyle".into()),
        Value::String("--".into()),
        Value::String("MarkerSize".into()),
        Value::Num(0.0),
    ]))
    .unwrap();
    let figure = clone_figure(current_figure_handle()).unwrap();
    assert_eq!(
        figure
            .plots()
            .filter(|plot| matches!(plot, PlotElement::Line(_)))
            .count(),
        2
    );
}
