use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual native WGPU adapter"]
fn resident_only_line_metadata_and_invalid_controls_do_not_read_coordinates() {
    use runmat_plot::gpu::{line::LineGpuInputs, GpuInputBuffer, ScalarType};
    use runmat_plot::plots::{LineGpuStyle, LinePlot, LineStyle};
    use std::sync::Arc;

    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let handle = host_line();

    let instance = Arc::new(wgpu::Instance::default());
    let adapter = Arc::new(
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("this hardware gate requires an actual WGPU adapter"),
    );
    let (device, queue) =
        block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .expect("this hardware gate requires an actual WGPU device");
    let access = runmat_accelerate_api::WgpuDeviceAccess::new(device, queue);
    let short_source = access
        .with_shared(|emission| {
            runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("graphics-property-short-resident-source"),
                    size: 4,
                    usage: wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            ))
        })
        .expect("create resident-only source");
    runmat_plot::install_shared_wgpu_context(runmat_plot::SharedWgpuContext {
        instance,
        access,
        adapter_info: adapter.get_info(),
        limits: adapter.limits(),
        features: adapter.features(),
        adapter,
    });
    let (child, _, _) = plot_child_element_snapshot(handle).expect("line snapshot");
    let PlotChildHandleState::Line(state) = child else {
        panic!("expected line child");
    };
    let inputs = LineGpuInputs {
        x_buffer: GpuInputBuffer::from(short_source.clone()),
        y_buffer: GpuInputBuffer::from(short_source),
        len: 2,
        scalar: ScalarType::F32,
    };
    let style = LineGpuStyle {
        color: glam::Vec4::ONE,
        line_width: 1.0,
        line_style: LineStyle::Solid,
        marker: None,
        handle_visibility: "on".into(),
    };
    crate::builtins::plotting::state::update_plot_element(state.figure, state.plot_index, |plot| {
        *plot = PlotElement::Line(LinePlot::from_gpu_xy(
            inputs,
            style,
            runmat_plot::core::BoundingBox::default(),
            None,
        ));
    })
    .expect("install resident-only line");

    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
    assert_eq!(
        block_on(get_properties_async(target, Some("Type"), "get")).expect("metadata"),
        Value::String("line".into())
    );
    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
    block_on(set_properties_async(
        target,
        &[Value::String("LineWidth".into()), Value::Num(3.0)],
        "set",
    ))
    .expect("style-only set must not read coordinates");
    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
    let invalid = block_on(set_properties_async(
        target,
        &[Value::String("XData".into()), Value::String("bad".into())],
        "set",
    ))
    .expect_err("invalid host control must fail before readback");
    assert!(!crate::builtins::plotting::error::plot_data_export_failure(
        &invalid
    ));

    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
    let readback = block_on(get_properties_async(target, Some("XData"), "get"))
        .expect_err("undersized resident source must fail when coordinates are requested");
    assert!(crate::builtins::plotting::error::plot_data_export_failure(
        &readback
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual native WGPU adapter"]
fn exported_resident_line_retains_borrowed_allocations_until_coordinate_publication() {
    use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};
    use runmat_plot::gpu::{line::LineGpuInputs, GpuInputBuffer, ScalarType};
    use runmat_plot::plots::{LineGpuStyle, LinePlot, LineStyle};

    let _accel_guard = crate::builtins::common::test_support::accel_test_lock();
    let _plot_guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let handle = host_line();
    let provider =
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default())
            .expect("actual WGPU device required");
    crate::builtins::plotting::context::ensure_context_from_provider()
        .expect("install the provider's shared device access");
    let x = provider
        .upload(&HostTensorView {
            data: &[1.25, -3.5],
            shape: &[1, 2],
        })
        .expect("upload x");
    let y = provider
        .upload(&HostTensorView {
            data: &[4.75, -2.0],
            shape: &[1, 2],
        })
        .expect("upload y");
    let x_wire = x.wire_descriptor();
    let y_wire = y.wire_descriptor();
    let x_export = runmat_accelerate_api::export_wgpu_buffer(&x)
        .unwrap()
        .expect("owned x export");
    let y_export = runmat_accelerate_api::export_wgpu_buffer(&y)
        .unwrap()
        .expect("owned y export");
    let scalar = ScalarType::from_is_f64(x_export.precision == ProviderPrecision::F64);
    let (child, _, _) = plot_child_element_snapshot(handle).expect("line snapshot");
    let PlotChildHandleState::Line(state) = child else {
        panic!("expected line child");
    };
    crate::builtins::plotting::state::update_plot_element(state.figure, state.plot_index, |plot| {
        *plot = PlotElement::Line(LinePlot::from_gpu_xy(
            LineGpuInputs {
                x_buffer: GpuInputBuffer::from(x_export),
                y_buffer: GpuInputBuffer::from(y_export),
                len: 2,
                scalar,
            },
            LineGpuStyle {
                color: glam::Vec4::ONE,
                line_width: 1.0,
                line_style: LineStyle::Solid,
                marker: None,
                handle_visibility: "on".into(),
            },
            runmat_plot::core::BoundingBox::default(),
            None,
        ));
    })
    .expect("install provider-owned line source");
    let receiver = crate::builtins::plotting::state::plot_child_graphics_receiver(handle as u64)
        .unwrap()
        .unwrap();
    let object = crate::builtins::plotting::state::graphics_receiver_object(receiver)
        .unwrap()
        .unwrap();
    drop(x);
    drop(y);
    assert!(provider.acquire_live(&x_wire).is_ok());
    assert!(provider.acquire_live(&y_wire).is_ok());

    let Value::Tensor(x_before) = block_on(crate::builtins::plotting::get::get_builtin_async(
        crate::builtins::common::test_support::builtin_arguments(vec![
            object.clone(),
            Value::String("XData".into()),
        ]),
    ))
    .expect("exact source read") else {
        panic!("expected typed source")
    };
    assert_eq!(x_before.materialize_f64(), vec![1.25, -3.5]);
    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
    let Value::Tensor(y_before) =
        block_on(get_properties_async(target, Some("YData"), "get")).expect("exported y")
    else {
        panic!("expected typed y tensor");
    };
    assert_eq!(y_before.materialize_f64(), vec![4.75, -2.0]);
    assert!(provider.acquire_live(&x_wire).is_ok());
    assert!(provider.acquire_live(&y_wire).is_ok());

    block_on(crate::builtins::plotting::set::set_builtin_async(vec![
        object,
        Value::String("XData".into()),
        Value::Tensor(Tensor::new(vec![8.0, 9.0], vec![1, 2]).expect("new x")),
        Value::String("YData".into()),
        Value::Tensor(Tensor::new(vec![-6.0, 7.0], vec![1, 2]).expect("new y")),
    ]))
    .expect("exact replacement coordinates");
    assert!(provider.acquire_live(&x_wire).is_err());
    assert!(provider.acquire_live(&y_wire).is_err());
    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
    let Value::Tensor(x_after) =
        block_on(get_properties_async(target, Some("XData"), "get")).expect("new x")
    else {
        panic!("expected new x tensor");
    };
    assert_eq!(x_after.materialize_f64(), vec![8.0, 9.0]);
    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
    let Value::Tensor(y_after) =
        block_on(get_properties_async(target, Some("YData"), "get")).expect("new y")
    else {
        panic!("expected new y tensor");
    };
    assert_eq!(y_after.materialize_f64(), vec![-6.0, 7.0]);
}
