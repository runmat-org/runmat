use super::*;
use crate::builtins::plotting::get::get_builtin;
use crate::builtins::plotting::plot::plot_builtin;
use crate::builtins::plotting::scatter::scatter_builtin;
use crate::builtins::plotting::state::{encode_axes_handle, PlotTestLockGuard};
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{
    clear_figure, clone_figure, configure_subplot, current_axes_state, current_figure_handle,
    reset_hold_state_for_run,
};
use futures::executor::block_on;
use runmat_builtins::{
    REFLINE_ERROR_INVALID_ARGUMENT, REFLINE_EXPLICIT_GPU_COEFFICIENT_EXTENSION,
    REFLINE_INTEGER_COEFFICIENT_EXTENSION, REFLINE_LOGICAL_COEFFICIENT_EXTENSION,
};
use runmat_value::{IntegerStorage, Tensor};

fn setup() -> PlotTestLockGuard {
    let guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    guard
}

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn x_data(handle: f64) -> Vec<f64> {
    let value = get_builtin(vec![Value::Num(handle), Value::String("XData".into())]).unwrap();
    Tensor::try_from(&value).unwrap().materialize_f64()
}

fn y_data(handle: f64) -> Vec<f64> {
    let value = get_builtin(vec![Value::Num(handle), Value::String("YData".into())]).unwrap();
    Tensor::try_from(&value).unwrap().materialize_f64()
}

#[test]
fn refline_without_coefficients_adds_least_squares_line() {
    let _guard = setup();
    block_on(scatter_builtin(None, {
        let mut coordinates = vec![
            tensor(vec![1.0, 2.0, 3.0], 1, 3),
            tensor(vec![2.0, 4.0, 6.0], 1, 3),
        ];
        coordinates.extend(Vec::new());
        crate::builtins::common::test_support::builtin_arguments(coordinates)
    }))
    .unwrap();
    let handle = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let Value::Num(handle) = handle else {
        panic!("expected line handle");
    };
    assert_eq!(x_data(handle), vec![1.0, 3.0]);
    assert_eq!(y_data(handle), vec![2.0, 6.0]);
}

#[test]
fn refline_accepts_coeff_vector_and_slope_intercept_forms() {
    let _guard = setup();
    let handle = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            vec![2.0, -1.0],
            1,
            2,
        )]),
    ))
    .unwrap();
    let Value::Num(handle) = handle else {
        panic!("expected line handle");
    };
    assert_eq!(y_data(handle), vec![-1.0, 1.0]);

    let handle = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(-0.5),
            Value::Num(3.0),
        ]),
    ))
    .unwrap();
    let Value::Num(handle) = handle else {
        panic!("expected line handle");
    };
    assert_eq!(y_data(handle), vec![3.0, 2.5]);
}

#[test]
fn invalid_style_does_not_retarget_axes() {
    let _guard = setup();
    configure_subplot(1, 2, 1).unwrap();
    let figure = current_figure_handle();
    let target = encode_axes_handle(figure, 0);
    let before = current_axes_state().active_index;
    let err = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(target),
            tensor(vec![1.0, 0.0], 1, 2),
            Value::String("Color".into()),
        ]),
    ))
    .expect_err("a name without its value must fail before axes mutation");
    assert_eq!(err.identifier(), REFLINE_ERROR_INVALID_ARGUMENT.identifier);
    assert_eq!(current_axes_state().active_index, before);
}

#[test]
fn refline_reads_typed_integer_coefficients_exactly() {
    let _guard = setup();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let handle = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![int_tensor(
            IntegerStorage::I16(vec![2, -1]),
            1,
            2,
        )]),
    ))
    .unwrap();
    let Value::Num(handle) = handle else {
        panic!("expected line handle");
    };
    assert_eq!(y_data(handle), vec![-1.0, 1.0]);
}

#[test]
fn refline_integer_extension_is_gated_and_inexact_values_reject() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![int_tensor(
            IntegerStorage::I16(vec![2, -1]),
            1,
            2,
        )]),
    ))
    .expect_err("strict mode rejects integer coefficients");
    assert_eq!(
        error.identifier(),
        REFLINE_INTEGER_COEFFICIENT_EXTENSION.error_identifier
    );
    drop(strict);

    let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![int_tensor(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, 0]),
            1,
            2,
        )]),
    ))
    .expect_err("inexact integer coefficients reject");
    assert!(error.message().contains("exactly representable as double"));
    drop(extensions);
}

#[test]
fn refline_strict_mode_gates_explicit_gpu_before_provider_access() {
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 2],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle;
    let handle = handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(handle)]),
    ))
    .expect_err("strict mode rejects explicit GPU coefficients before gather");
    assert_eq!(
        error.identifier(),
        REFLINE_EXPLICIT_GPU_COEFFICIENT_EXTENSION.error_identifier
    );
}

#[test]
fn refline_logical_coefficients_are_a_gated_extension() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Bool(true),
            Value::Bool(false),
        ]),
    ))
    .expect_err("strict mode rejects logical coefficients");
    assert_eq!(
        error.identifier(),
        REFLINE_LOGICAL_COEFFICIENT_EXTENSION.error_identifier
    );
}

#[test]
fn refline_spans_existing_plot_or_explicit_axes_limits() {
    let _guard = setup();
    let _ = block_on(plot_builtin(vec![
        tensor(vec![2.0, 4.0], 1, 2),
        tensor(vec![10.0, 20.0], 1, 2),
    ]))
    .unwrap();
    let handle = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(0.0),
        ]),
    ))
    .unwrap();
    let Value::Num(handle) = handle else {
        panic!("expected line handle");
    };
    assert_eq!(x_data(handle), vec![2.0, 4.0]);
    assert_eq!(y_data(handle), vec![2.0, 4.0]);

    let _ =
        crate::builtins::plotting::xlim::xlim_builtin(vec![tensor(vec![-1.0, 3.0], 1, 2)]).unwrap();
    let handle = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(1.0),
        ]),
    ))
    .unwrap();
    let Value::Num(handle) = handle else {
        panic!("expected line handle");
    };
    assert_eq!(x_data(handle), vec![-1.0, 3.0]);
    assert_eq!(y_data(handle), vec![-1.0, 7.0]);
}

#[test]
fn refline_explicit_finite_endpoints_survive_intermediate_product_overflow() {
    let _guard = setup();
    crate::builtins::plotting::xlim::xlim_builtin(vec![tensor(vec![2.0, 2.5], 1, 2)]).unwrap();
    let handle = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0e308),
            Value::Num(-1.0e308),
        ]),
    ))
    .expect("finite fused endpoints");
    let Value::Num(handle) = handle else {
        panic!("expected line handle")
    };
    assert_eq!(x_data(handle), vec![2.0, 2.5]);
    assert_eq!(y_data(handle), vec![1.0e308, 1.5e308]);
}

#[test]
fn refline_targets_axes_and_accepts_style_args() {
    let _guard = setup();
    configure_subplot(1, 2, 0).unwrap();
    let _ = block_on(plot_builtin(vec![
        tensor(vec![100.0, 200.0], 1, 2),
        tensor(vec![10.0, 20.0], 1, 2),
    ]))
    .unwrap();
    configure_subplot(1, 2, 1).unwrap();
    let _ = block_on(plot_builtin(vec![
        tensor(vec![2.0, 4.0], 1, 2),
        tensor(vec![10.0, 20.0], 1, 2),
    ]))
    .unwrap();
    let fig = current_figure_handle();
    let ax = encode_axes_handle(fig, 1);
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let supplied = crate::builtins::common::test_support::builtin_arguments(vec![
        Value::Num(0.0),
        Value::Num(2.0),
        Value::String("--r".into()),
        Value::String("DisplayName".into()),
        Value::String("threshold".into()),
    ]);
    let args = crate::call::BuiltinArguments::new(
        supplied
            .into_function_arguments()
            .prepend_routing_in_context(Value::Num(ax), &runtime)
            .unwrap(),
    );
    let handle = block_on(refline_builtin(Some(runtime), args)).unwrap();
    let Value::Num(handle) = handle else {
        panic!("expected line handle");
    };
    assert_eq!(x_data(handle), vec![2.0, 4.0]);
    let style = get_builtin(vec![Value::Num(handle), Value::String("LineStyle".into())]).unwrap();
    assert_eq!(style, Value::String("--".into()));
    let name = get_builtin(vec![
        Value::Num(handle),
        Value::String("DisplayName".into()),
    ])
    .unwrap();
    assert_eq!(name, Value::String("threshold".into()));
    let figure = clone_figure(fig).unwrap();
    assert_eq!(figure.len(), 3);
}

#[test]
fn refline_rejects_bad_coefficients() {
    let _guard = setup();
    let err = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(
            vec![1.0, 2.0, 3.0],
            1,
            3,
        )]),
    ))
    .unwrap_err();
    assert!(err.message.contains("coefficients"));

    let err = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(f64::NAN),
            Value::Num(0.0),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("finite"));
}

#[test]
#[cfg(feature = "wgpu")]
fn refline_least_squares_reads_gpu_scatter_source_data() {
    use crate::builtins::plotting::state::{append_active_plot, PlotRenderOptions};
    use runmat_plot::core::{BoundingBox, GpuVertexBuffer};
    use runmat_plot::gpu::scatter2::Scatter2GpuInputs;
    use runmat_plot::gpu::ScalarType;
    use runmat_plot::plots::scatter::ScatterGpuStyle;
    use runmat_plot::plots::ScatterPlot;

    let _guard = setup();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        tracing::warn!("Skipping refline GPU scatter regression: no WGPU provider");
        return;
    };
    let context = crate::builtins::plotting::context::ensure_context_from_provider()
        .expect("shared plotting context");

    let x_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("x tensor");
    let y_tensor = Tensor::new(vec![2.0, 4.0, 6.0], vec![1, 3]).expect("y tensor");
    let x_handle =
        crate::builtins::common::gpu_helpers::upload_tensor(provider, &x_tensor).expect("gpu x");
    let y_handle =
        crate::builtins::common::gpu_helpers::upload_tensor(provider, &y_tensor).expect("gpu y");
    let x = Value::GpuTensor(x_handle.clone());
    let y = Value::GpuTensor(y_handle.clone());
    let x_ref = runmat_accelerate_api::export_wgpu_buffer(&x_handle)
        .unwrap()
        .expect("export x");
    let y_ref = runmat_accelerate_api::export_wgpu_buffer(&y_handle)
        .unwrap()
        .expect("export y");
    let dummy_vertices = context
        .access
        .with_shared(|emission| {
            runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("refline-lsline-gpu-scatter-test-dummy-vertices"),
                    size: 16,
                    usage: wgpu::BufferUsages::VERTEX,
                    mapped_at_creation: false,
                },
            ))
        })
        .expect("shared WGPU device access");
    let scatter = ScatterPlot::from_gpu_buffer(
        GpuVertexBuffer::new(dummy_vertices, 0),
        0,
        BoundingBox::new(
            glam::Vec3::new(1.0, 2.0, 0.0),
            glam::Vec3::new(3.0, 6.0, 0.0),
        ),
        ScatterGpuStyle {
            color: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            edge_color: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            edge_thickness: 1.0,
            marker_size: 12.0,
            marker_style: runmat_plot::plots::MarkerStyle::Circle,
            filled: false,
            has_per_point_sizes: false,
            has_per_point_colors: false,
            edge_from_vertex_colors: false,
        },
    )
    .with_gpu_source_inputs(Scatter2GpuInputs {
        x_buffer: x_ref.clone().into(),
        y_buffer: y_ref.clone().into(),
        len: x_ref.len as u32,
        scalar: ScalarType::from_is_f64(
            x_ref.precision == runmat_accelerate_api::ProviderPrecision::F64,
        ),
    });
    let mut scatter = Some(scatter);
    append_active_plot(
        ReferenceLineOwner::Refline.name(),
        PlotRenderOptions::default(),
        move |figure, axes| {
            let scatter = scatter.take().ok_or_else(|| {
                crate::build_runtime_error("refline: test scatter already inserted")
                    .with_builtin("refline")
                    .with_identifier(runmat_builtins::REFLINE_ERROR_INTERNAL.identifier.unwrap())
                    .build()
            })?;
            figure.add_scatter_plot_on_axes(scatter, axes);
            Ok(())
        },
    )
    .expect("insert GPU scatter plot");

    let handle = block_on(refline_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("gpu-backed refline");
    let Value::Num(handle) = handle else {
        panic!("expected line handle");
    };
    assert_eq!(x_data(handle), vec![1.0, 3.0]);
    assert_eq!(y_data(handle), vec![2.0, 6.0]);

    if let Value::GpuTensor(handle) = x {
        drop(handle);
    }
    if let Value::GpuTensor(handle) = y {
        drop(handle);
    }
    runmat_accelerate::simple_provider::register_inprocess_provider();
}
