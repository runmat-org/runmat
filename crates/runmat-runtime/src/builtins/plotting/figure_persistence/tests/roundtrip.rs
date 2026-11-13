use super::*;

#[test]
fn savefig_round_trips_single_figure() {
    let _guard = setup();
    block_on(plot_builtin(vec![
        Value::Tensor(tensor(&[1.0, 2.0, 3.0])),
        Value::Tensor(tensor(&[1.0, 4.0, 9.0])),
    ]))
    .expect("plot");

    let path = temp_path("single");
    let _ = std::fs::remove_file(&path);
    block_on(savefig_builtin(vec![Value::String(
        path.to_string_lossy().into_owned().into(),
    )]))
    .expect("savefig");
    reset_plot_state();
    let value = block_on(openfig_builtin(vec![Value::String(
        path.to_string_lossy().into_owned().into(),
    )]))
    .expect("openfig");
    assert!(matches!(value, Value::Num(_)));
    assert_eq!(figure_handles().len(), 1);
    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "plot-core")]
#[test]
fn savefig_round_trips_exact_native_float_sources_and_surface_hole() {
    use runmat_plot::plots::{Figure, LinePlot, NumericPlotData, PlotElement, SurfacePlot};
    use runmat_value::NumericStorage;

    let _guard = setup();
    let double_bits = [
        0x3ff8_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0042,
    ];
    let single_bits = [
        0x3fc0_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0042,
    ];
    let hole_bits = 0x7ff8_0000_0000_0091;
    let mut figure = Figure::new();
    figure.add_line_plot(
        LinePlot::from_numeric_data(
            NumericPlotData::new(
                NumericStorage::F64(double_bits.map(f64::from_bits).to_vec()),
                vec![1, double_bits.len()],
            )
            .unwrap(),
            NumericPlotData::new(
                NumericStorage::F32(single_bits.map(f32::from_bits).to_vec()),
                vec![1, single_bits.len()],
            )
            .unwrap(),
        )
        .unwrap(),
    );
    figure.add_surface_plot(
        SurfacePlot::new(
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![vec![1.0, f64::from_bits(hole_bits)], vec![2.0, 3.0]],
        )
        .unwrap(),
    );
    let _original = crate::builtins::plotting::import_figure(figure);

    let path = temp_path("native_float_sources");
    let _ = std::fs::remove_file(&path);
    block_on(savefig_builtin(vec![Value::String(
        path.to_string_lossy().into_owned().into(),
    )]))
    .expect("savefig native sources");
    reset_plot_state();
    let opened = block_on(openfig_builtin(vec![Value::String(
        path.to_string_lossy().into_owned().into(),
    )]))
    .expect("openfig native sources");
    assert!(matches!(opened, Value::Num(_)));
    let rebuilt = clone_figure(current_figure_handle()).expect("reopened figure");
    let mut plots = rebuilt.plots();
    let Some(PlotElement::Line(line)) = plots.next() else {
        panic!("expected reopened line")
    };
    let (Some(x), Some(y)) = line.source_data() else {
        panic!("expected native line sources")
    };
    assert_eq!(x.shape(), &[1, double_bits.len()]);
    assert_eq!(y.shape(), &[1, single_bits.len()]);
    let NumericStorage::F64(x_values) = x.storage() else {
        panic!("expected double X source")
    };
    let NumericStorage::F32(y_values) = y.storage() else {
        panic!("expected single Y source")
    };
    assert_eq!(
        x_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        double_bits
    );
    assert_eq!(
        y_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        single_bits
    );
    let Some(PlotElement::Surface(surface)) = plots.next() else {
        panic!("expected reopened surface")
    };
    let (_, _, Some(z)) = surface.source_data() else {
        panic!("expected native surface Z source")
    };
    assert_eq!(z.shape(), &[2, 2]);
    let NumericStorage::F64(z_values) = z.storage() else {
        panic!("expected double surface Z source")
    };
    assert_eq!(z_values[1].to_bits(), hole_bits);
    assert!(plots.next().is_none());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn savefig_round_trips_multiple_figures() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = setup();
    let first = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("figure one");
    block_on(plot_builtin(vec![Value::Tensor(tensor(&[1.0, 2.0, 3.0]))])).expect("plot one");
    let second = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "next".into(),
        )]),
    )
    .expect("figure two");
    block_on(plot_builtin(vec![Value::Tensor(tensor(&[3.0, 2.0, 1.0]))])).expect("plot two");
    let handles =
        Value::Tensor(Tensor::new(vec![first, second], vec![1, 2]).expect("figure handle row"));

    let path = temp_path("multi");
    let _ = std::fs::remove_file(&path);
    block_on(savefig_builtin(vec![
        handles,
        Value::String(path.to_string_lossy().into_owned().into()),
        Value::String("-v7.3".into()),
    ]))
    .expect("savefig multi");
    clear_figure(None).expect("clear");
    let value = block_on(openfig_builtin(vec![
        Value::String(path.to_string_lossy().into_owned().into()),
        Value::String("new".into()),
        Value::String("invisible".into()),
    ]))
    .expect("openfig");
    let Value::Tensor(tensor) = value else {
        panic!("expected handle tensor");
    };
    assert_eq!(tensor.materialize_f64().len(), 2);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn savefig_rejects_typed_integer_handle_arrays_before_io() {
    let _guard = setup();
    let first = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("figure one");
    let handles = Tensor::new_integer(
        runmat_value::IntegerStorage::I64(vec![first as i64]),
        vec![1, 1],
    )
    .expect("handle tensor");

    let path = temp_path("integer_handles");
    let _ = std::fs::remove_file(&path);
    let error = block_on(savefig_builtin(vec![
        Value::Tensor(handles),
        Value::String(path.to_string_lossy().into_owned().into()),
    ]))
    .expect_err("typed integer graphics handles must reject");
    assert_eq!(
        error.identifier(),
        Some("RunMat:figurePersistence:InvalidArgument")
    );
    assert!(!path.exists());
}

#[test]
fn hgsave_deduplicates_handles_resolving_to_same_figure() {
    let _guard = setup();
    let fig = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("figure");
    block_on(plot_builtin(vec![Value::Tensor(tensor(&[1.0, 2.0, 3.0]))])).expect("plot");
    let handles =
        Value::Tensor(Tensor::new(vec![fig, fig], vec![1, 2]).expect("figure handle row"));

    let path = temp_path("dedupe");
    let _ = std::fs::remove_file(&path);
    block_on(hgsave_builtin(vec![
        handles,
        Value::String(path.to_string_lossy().into_owned().into()),
    ]))
    .expect("hgsave duplicate handles");
    reset_plot_state();
    let value = block_on(hgload_builtin(vec![Value::String(
        path.to_string_lossy().into_owned().into(),
    )]))
    .expect("hgload");
    assert!(matches!(value, Value::Num(_)));
    assert_eq!(figure_handles().len(), 1);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn saveas_png_uses_renderer_fast_path() {
    let _guard = setup();
    block_on(plot_builtin(vec![Value::Tensor(tensor(&[1.0, 2.0, 3.0]))])).expect("plot");
    let mut path = std::env::temp_dir();
    path.push(format!("runmat_saveas_png_{}.png", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let handle = current_figure_handle().as_u32() as f64;
    block_on(saveas_builtin(vec![
        Value::Num(handle),
        Value::String(path.to_string_lossy().into_owned().into()),
        Value::String("png".into()),
    ]))
    .expect("saveas png");
    let bytes = std::fs::read(&path).expect("read png");
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    let _ = std::fs::remove_file(&path);
}
