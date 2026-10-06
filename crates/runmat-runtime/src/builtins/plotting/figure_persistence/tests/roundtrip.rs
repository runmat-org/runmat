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
        path.to_string_lossy().into_owned(),
    )]))
    .expect("savefig");
    reset_plot_state();
    let value = block_on(openfig_builtin(vec![Value::String(
        path.to_string_lossy().into_owned(),
    )]))
    .expect("openfig");
    assert!(matches!(value, Value::Num(_)));
    assert_eq!(figure_handles().len(), 1);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn savefig_round_trips_multiple_figures() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = setup();
    let first = figure_builtin(vec![]).expect("figure one");
    block_on(plot_builtin(vec![Value::Tensor(tensor(&[1.0, 2.0, 3.0]))])).expect("plot one");
    let second = figure_builtin(vec![Value::String("next".into())]).expect("figure two");
    block_on(plot_builtin(vec![Value::Tensor(tensor(&[3.0, 2.0, 1.0]))])).expect("plot two");
    let handles =
        Value::Tensor(Tensor::new(vec![first, second], vec![1, 2]).expect("figure handle row"));

    let path = temp_path("multi");
    let _ = std::fs::remove_file(&path);
    block_on(savefig_builtin(vec![
        handles,
        Value::String(path.to_string_lossy().into_owned()),
        Value::String("-v7.3".into()),
    ]))
    .expect("savefig multi");
    clear_figure(None).expect("clear");
    let value = block_on(openfig_builtin(vec![
        Value::String(path.to_string_lossy().into_owned()),
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
    let first = figure_builtin(vec![]).expect("figure one");
    let handles = Tensor::new_integer(
        runmat_value::IntegerStorage::I64(vec![first as i64]),
        vec![1, 1],
    )
    .expect("handle tensor");

    let path = temp_path("integer_handles");
    let _ = std::fs::remove_file(&path);
    let error = block_on(savefig_builtin(vec![
        Value::Tensor(handles),
        Value::String(path.to_string_lossy().into_owned()),
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
    let fig = figure_builtin(vec![]).expect("figure");
    block_on(plot_builtin(vec![Value::Tensor(tensor(&[1.0, 2.0, 3.0]))])).expect("plot");
    let handles =
        Value::Tensor(Tensor::new(vec![fig, fig], vec![1, 2]).expect("figure handle row"));

    let path = temp_path("dedupe");
    let _ = std::fs::remove_file(&path);
    block_on(hgsave_builtin(vec![
        handles,
        Value::String(path.to_string_lossy().into_owned()),
    ]))
    .expect("hgsave duplicate handles");
    reset_plot_state();
    let value = block_on(hgload_builtin(vec![Value::String(
        path.to_string_lossy().into_owned(),
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
        Value::String(path.to_string_lossy().into_owned()),
        Value::String("png".into()),
    ]))
    .expect("saveas png");
    let bytes = std::fs::read(&path).expect("read png");
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    let _ = std::fs::remove_file(&path);
}
