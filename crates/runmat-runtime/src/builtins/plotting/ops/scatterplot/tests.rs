use super::*;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{
    clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
};
use futures::executor::block_on;
use runmat_builtins::SCATTERPLOT_DESCRIPTOR;
use runmat_plot::plots::{scatter::MarkerStyle, PlotElement};
use runmat_value::ComplexTensor;
use runmat_value::{IntegerStorage, NumericStorage};

fn setup_plot_tests() {
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
}

fn run_scatterplot(x: Value, rest: Vec<Value>) -> BuiltinResult<f64> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    block_on(super::scatterplot_builtin(None, x, rest))
}

fn complex_tensor(data: &[(f64, f64)]) -> ComplexTensor {
    ComplexTensor::new(data.to_vec(), vec![data.len(), 1]).expect("complex tensor")
}

fn poisoned_integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
    Tensor::new_integer(storage, shape).expect("integer tensor")
}

fn all_integer_scalar_storages(value: u8) -> [IntegerStorage; 8] {
    [
        IntegerStorage::I8(vec![value as i8]),
        IntegerStorage::I16(vec![value as i16]),
        IntegerStorage::I32(vec![value as i32]),
        IntegerStorage::I64(vec![value as i64]),
        IntegerStorage::U8(vec![value]),
        IntegerStorage::U16(vec![value as u16]),
        IntegerStorage::U32(vec![value as u32]),
        IntegerStorage::U64(vec![value as u64]),
    ]
}

#[test]
fn scatterplot_decimates_from_zero_based_offset() {
    let data = complex_tensor(&[(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)]);
    let (x, y, limits) =
        extract_host_points(Value::ComplexTensor(data), 2, 1).expect("decimated points");
    assert_eq!(x.as_f64_slice(), Some(&[2.0, 4.0][..]));
    assert_eq!(y.as_f64_slice(), Some(&[20.0, 40.0][..]));
    assert_eq!(limits, Some((-42.0, 42.0)));
}

#[test]
fn scatterplot_samples_and_scalars_read_typed_integer_storage() {
    let samples = poisoned_integer_tensor(IntegerStorage::I16(vec![1, -2, 3]), vec![3, 1]);
    let (x, y, _) = extract_host_points(Value::Tensor(samples), 1, 0).expect("integer samples");
    assert_eq!(x.as_f64_slice(), Some(&[1.0, -2.0, 3.0][..]));
    assert_eq!(y.as_f64_slice(), Some(&[0.0, 0.0, 0.0][..]));

    let decimation = poisoned_integer_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]);
    assert_eq!(
        parse_nonnegative_integer(&Value::Tensor(decimation), "n").expect("decimation"),
        2
    );
}

#[test]
fn scatterplot_materializes_native_single_at_the_plotting_boundary() {
    let samples = Tensor::from_numeric_storage(NumericStorage::F32(vec![1.25, -2.5]), vec![2, 1])
        .expect("single samples");
    let (x, y, limits) = extract_host_points(Value::Tensor(samples), 1, 0).expect("single samples");
    assert_eq!(x.as_f64_slice(), Some(&[1.25, -2.5][..]));
    assert_eq!(y.as_f64_slice(), Some(&[0.0, 0.0][..]));
    assert_eq!(limits, Some((-2.625, 2.625)));
}

#[test]
fn scatterplot_count_parser_reads_all_integer_storages_without_mirrors() {
    for storage in all_integer_scalar_storages(2) {
        let value = Value::Tensor(poisoned_integer_tensor(storage, vec![1, 1]));
        assert_eq!(parse_nonnegative_integer(&value, "n").unwrap(), 2);
    }
}

#[test]
fn scatterplot_figure_parser_rejects_all_integer_storages_without_mirrors() {
    for storage in all_integer_scalar_storages(1) {
        let value = Value::Tensor(poisoned_integer_tensor(storage, vec![1, 1]));
        assert!(parse_figure_handle(&value).is_err());
    }
}

#[test]
fn scatterplot_rejects_offset_equal_to_decimation() {
    let err = ScatterplotOptions::parse(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Num(2.0), Value::Num(2.0)],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("offset must be less than n"));
}

#[test]
fn scatterplot_rejects_invalid_marker_at_parse_time() {
    let err = ScatterplotOptions::parse(crate::builtins::common::test_support::builtin_arguments(
        vec![Value::Num(1.0), Value::Num(0.0), Value::Num(7.0)],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:scatterplot:InvalidArgument"));
    assert!(!err.to_string().contains("PlotFailed"));
}

#[test]
fn scatterplot_rejects_unknown_figure_at_parse_time() {
    let err = ScatterplotOptions::parse(crate::builtins::common::test_support::builtin_arguments(
        vec![
            Value::Num(1.0),
            Value::Num(0.0),
            Value::String("x".into()),
            Value::Num(f64::from(u32::MAX)),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:scatterplot:InvalidArgument"));
    assert!(!err.to_string().contains("PlotFailed"));
}

#[test]
fn scatterplot_rejects_nonscalar_figure_at_parse_time() {
    let figure = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor");
    let err = ScatterplotOptions::parse(crate::builtins::common::test_support::builtin_arguments(
        vec![
            Value::Num(1.0),
            Value::Num(0.0),
            Value::String("x".into()),
            Value::Tensor(figure),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:scatterplot:InvalidArgument"));
    assert!(!err.to_string().contains("PlotFailed"));
}

#[test]
fn scatterplot_rejects_unrepresentable_integer_options_before_cast() {
    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };
    assert!(parse_nonnegative_integer(&Value::Num(boundary), "n").is_err());
}

#[test]
fn scatterplot_smoke_renders_complex_samples() {
    let _guard = lock_plot_registry();
    setup_plot_tests();
    let data = complex_tensor(&[(1.0, -1.0), (0.5, 0.5), (-1.0, 1.0)]);
    let _ = run_scatterplot(Value::ComplexTensor(data), Vec::new()).unwrap();
    let fig = clone_figure(current_figure_handle()).unwrap();
    let PlotElement::Scatter(plot) = fig.plots().next().unwrap() else {
        panic!("expected scatter plot")
    };
    let (x, y) = plot.host_xy_f64().unwrap().unwrap();
    assert_eq!(x, vec![1.0, 0.5, -1.0]);
    assert_eq!(y, vec![-1.0, 0.5, 1.0]);
    assert!(fig.axis_equal);
    assert!(fig.grid_enabled);
}

#[test]
fn scatterplot_forwards_marker_to_scatter() {
    let _guard = lock_plot_registry();
    setup_plot_tests();
    let data = complex_tensor(&[(1.0, 1.0), (2.0, 2.0)]);
    let _ = run_scatterplot(
        Value::ComplexTensor(data),
        vec![Value::Num(1.0), Value::Num(0.0), Value::String("s".into())],
    )
    .unwrap();
    let fig = clone_figure(current_figure_handle()).unwrap();
    let PlotElement::Scatter(plot) = fig.plots().next().unwrap() else {
        panic!("expected scatter plot")
    };
    assert_eq!(plot.marker_style, MarkerStyle::Square);
}

#[test]
fn scatterplot_accepts_trailing_existing_figure_handle() {
    let _guard = lock_plot_registry();
    setup_plot_tests();
    let fig_handle = current_figure_handle();
    select_figure(fig_handle);
    let data = complex_tensor(&[(1.0, 2.0), (3.0, 4.0)]);
    let _ = run_scatterplot(
        Value::ComplexTensor(data),
        vec![
            Value::Num(1.0),
            Value::Num(0.0),
            Value::String("x".into()),
            Value::Num(f64::from(fig_handle.as_u32())),
        ],
    )
    .unwrap();
    let fig = clone_figure(fig_handle).unwrap();
    assert_eq!(fig.plot_axes_indices(), &[0]);
}

#[test]
fn scatterplot_descriptor_lists_matlab_call_forms() {
    let labels: Vec<&str> = SCATTERPLOT_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"h = scatterplot(x)"));
    assert!(labels.contains(&"h = scatterplot(x, n)"));
    assert!(labels.contains(&"h = scatterplot(x, n, offset)"));
    assert!(labels.contains(&"h = scatterplot(x, n, offset, marker)"));
    assert!(labels.contains(&"h = scatterplot(x, n, offset, marker, scatfig)"));
}

#[test]
fn scatterplot_is_registered_with_descriptor() {
    let entry = runmat_builtins::builtin_catalog_entry_by_name("scatterplot")
        .expect("scatterplot registered in the typed catalog");
    assert_eq!(entry.category, "communications/plotting");
    assert!(entry
        .descriptor
        .signatures
        .iter()
        .any(|signature| signature.label == "h = scatterplot(x, n, offset, marker, scatfig)"));
}
