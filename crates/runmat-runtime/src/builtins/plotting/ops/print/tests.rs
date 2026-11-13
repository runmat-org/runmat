use super::*;
use crate::builtins::plotting::plot::plot_builtin;
use crate::builtins::plotting::state::{clear_figure, reset_hold_state_for_run};
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use futures::executor::block_on;
use runmat_value::Tensor;

#[path = "tests/dispatcher.rs"]
mod dispatcher;

fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
    let guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    guard
}

fn tensor(data: &[f64]) -> Tensor {
    Tensor::new(data.to_vec(), vec![data.len()]).expect("print test vector")
}

fn unique_temp_path(stem: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let thread = std::thread::current();
    let thread_name = thread.name().unwrap_or("thread");
    let safe_thread_name: String = thread_name
        .chars()
        .map(|ch| match ch {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '_',
            ch if ch.is_control() => '_',
            ch => ch,
        })
        .collect();
    path.push(format!(
        "runmat_print_test_{}_{}_{}",
        stem,
        std::process::id(),
        safe_thread_name
    ));
    path
}

#[test]
fn print_figure_handle_arg_reads_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor =
        Tensor::new_integer(runmat_value::IntegerStorage::U32(vec![5]), vec![1, 1]).unwrap();

    assert_eq!(
        figure_handle_arg(&Value::Tensor(tensor)).unwrap(),
        Some(FigureHandle::from(5))
    );
}

#[test]
fn print_descriptor_signatures_cover_png_forms() {
    let labels: Vec<&str> = PRINT_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"print(filename)"));
    assert!(labels.contains(&"print(fig, filename, ___)"));
    assert!(labels.contains(&"ok = print(filename, ___)"));
}

#[test]
fn print_rejects_unsupported_device() {
    let _guard = setup();
    let err = block_on(print_builtin_direct(
        None,
        vec![Value::from("plot"), Value::from("-dsvg")],
    ))
    .expect_err("svg is not supported yet");
    assert!(err.message().contains("currently supports '-dpng'"));
}

#[test]
fn nested_container_is_not_materialized_as_a_print_control() {
    let nested = runmat_value::CellArray::new(vec![Value::from("plot.png")], 1, 1)
        .expect("one-cell fixture");
    let error = preflight_host_arguments(crate::call::FunctionArgumentView::from_slice(&[
        Value::Cell(nested),
    ]))
    .expect_err("container is not a scalar text operand");
    assert_eq!(error.identifier(), PRINT_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn output_admission_precedes_render_and_file_effects() {
    let _guard = setup();
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let one = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let error = block_on(print_builtin_direct(
        None,
        vec![Value::from("never-created.png")],
    ))
    .expect_err("status output is an extension");
    assert_eq!(
        error.identifier(),
        PRINT_STATUS_OUTPUT_EXTENSION.error_identifier
    );
    drop(one);
    let _two = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(print_builtin_direct(
        None,
        vec![Value::from("never-created.png")],
    ))
    .expect_err("excess outputs");
    assert_eq!(error.identifier(), PRINT_ERROR_TOO_MANY_OUTPUTS.identifier);
}

#[test]
fn print_adds_png_extension_and_writes_png() {
    let _guard = setup();
    let _fs = runmat_filesystem::provider_override_lock();
    block_on(plot_builtin(vec![
        Value::Tensor(tensor(&[0.0, 1.0, 2.0, 3.0])),
        Value::Tensor(tensor(&[0.0, 1.0, 4.0, 9.0])),
    ]))
    .expect("plot");

    let stem = unique_temp_path("basic");
    let output = stem.with_extension("png");
    let _ = std::fs::remove_file(&output);
    let _statement = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    block_on(print_builtin_direct(
        None,
        vec![
            Value::String(stem.to_string_lossy().into_owned().into()),
            Value::from("-dpng"),
            Value::from("-r75"),
        ],
    ))
    .expect("print");

    let bytes = std::fs::read(&output).expect("read exported png");
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    assert!(bytes.len() > 1000);
    let _ = std::fs::remove_file(&output);
}

#[cfg(feature = "plot-core")]
#[test]
fn captured_scene_renders_after_ambient_figure_is_cleared() {
    let _guard = setup();
    block_on(plot_builtin(vec![
        Value::Tensor(tensor(&[0.0, 1.0, 2.0])),
        Value::Tensor(tensor(&[1.0, 3.0, 2.0])),
    ]))
    .expect("plot");
    let selected = current_figure_handle();
    let mut scene = clone_figure(selected).expect("captured plotted scene");
    // Explicit export must retain hidden figures and all subplot content,
    // independently of automatic presentation eligibility or ambient state.
    scene.set_name("hidden subplot export");
    scene.set_visible(false);
    scene.set_subplot_grid(1, 2);
    scene.add_bar_chart_on_axes(
        runmat_plot::plots::BarChart::new(vec!["first".into(), "second".into()], vec![3.0, 1.0])
            .expect("bar data"),
        1,
    );
    assert_eq!(scene.len(), 2);
    assert_eq!(scene.axes_grid(), (1, 2));
    assert!(!scene.visible);
    clear_figure(None).expect("clear ambient figure");
    let bytes = block_on(render_png(scene, 800, 600)).expect("captured scene export");
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    assert!(bytes.len() > 1000);
}

#[test]
fn command_style_order_accepts_device_then_filename() {
    let _guard = setup();
    let request = parse_print_args(
        crate::call::FunctionArgumentView::from_slice(&[
            Value::from("-dpng"),
            Value::from("command_style_name"),
        ]),
        current_figure_handle(),
    )
    .expect("parse");
    assert_eq!(
        request.output_path().expect("output path"),
        PathBuf::from("command_style_name.png")
    );
    assert_eq!(request.device, PrintDevice::Png);
}

#[test]
fn print_appends_png_to_filename_component_only() {
    let request = parse_print_args(
        crate::call::FunctionArgumentView::from_slice(&[
            Value::from("exports/report"),
            Value::from("-dpng"),
        ]),
        current_figure_handle(),
    )
    .expect("parse");
    assert_eq!(
        request.output_path().expect("output path"),
        PathBuf::from("exports/report.png")
    );
}

#[test]
fn all_host_text_forms_retain_the_absolute_filename_before_gather() {
    let filename = "/exports/plot.png";
    let controls = [
        Value::String(filename.into()),
        Value::CharArray(runmat_value::CharArray::new_row(filename)),
        Value::StringArray(runmat_value::StringArray::new_2d(vec![filename.into()], 1, 1).unwrap()),
    ];
    for control in controls {
        assert_eq!(
            preflight_host_arguments(crate::call::FunctionArgumentView::from_slice(&[control]))
                .expect("host preflight"),
            Some(PathBuf::from(filename))
        );
    }
}

#[test]
fn repeated_resolution_uses_last_token_in_runtime_parser() {
    let request = parse_print_args(
        crate::call::FunctionArgumentView::from_slice(&[
            Value::from("plot.png"),
            Value::from("-r75"),
            Value::from("-r300"),
        ]),
        current_figure_handle(),
    )
    .expect("last resolution wins");
    assert_eq!((request.width, request.height), (1600, 1200));
}

#[test]
fn integer_alias_admission_and_exactness_precede_filesystem_capture() {
    let _guard = setup();
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    for integer in [
        runmat_value::IntValue::U8(1),
        runmat_value::IntValue::U64(u64::MAX),
    ] {
        let error = block_on(print_builtin_direct(
            None,
            vec![Value::Int(integer), Value::from("/never-mounted/plot.png")],
        ))
        .expect_err("integer alias requires compatibility admission");
        assert_eq!(
            error.identifier(),
            PRINT_INTEGER_FIGURE_EXTENSION.error_identifier
        );
    }
    drop(_matlab);
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(print_builtin_direct(
        None,
        vec![
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            Value::from("/never-mounted/plot.png"),
        ],
    ))
    .expect_err("wide integer must fail before filesystem capture");
    assert!(error.message().contains("exactly representable"));
}

#[cfg(feature = "plot-core")]
#[test]
fn pre_cancelled_public_print_preserves_existing_target() {
    let _guard = setup();
    let _fs = runmat_filesystem::provider_override_lock();
    block_on(plot_builtin(vec![
        Value::Tensor(tensor(&[0.0, 1.0])),
        Value::Tensor(tensor(&[1.0, 2.0])),
    ]))
    .expect("plot before cancellation");
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir("/"));
    provider
        .write_project_path("/plot.png", b"old")
        .expect("old target");
    let _selected = runmat_filesystem::replace_provider(provider.clone());
    let source = runmat_execution::cancellation::CancellationSource::new();
    source.cancel(runmat_execution::CancellationReason::User);
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let _statement = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(print_builtin_direct(
        Some(runtime),
        vec![Value::from("/plot.png")],
    ))
    .expect_err("pre-cancelled call");
    assert!(error
        .identifier()
        .is_some_and(|id| id.ends_with(":Cancelled")));
    assert_eq!(provider.read_project_path("/plot.png").unwrap(), b"old");
}

#[test]
fn print_rejects_directory_like_png_output_path() {
    let request = parse_print_args(
        crate::call::FunctionArgumentView::from_slice(&[Value::from("/"), Value::from("-dpng")]),
        current_figure_handle(),
    )
    .expect("parse");
    let err = request
        .output_path()
        .expect_err("root path has no filename component");
    assert!(err
        .message()
        .contains("filename must include a file name before appending '.png'"));
}
