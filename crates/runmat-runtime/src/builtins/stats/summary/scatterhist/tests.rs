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
        vec![Some(wide.to_string()), Some((wide - 1).to_string())]
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
    assert_eq!(handles.shape, vec![3, 1]);
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
    let axes = figure.plot_axes_indices();
    assert_eq!(axes, &[1, 3, 0]);
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
