use super::*;
use runmat_accelerate_api::{WgpuBufferOwner, WgpuDeviceAccess};
use runmat_plot::gpu::{
    line3::Line3GpuInputs, scatter2::Scatter2GpuInputs, GpuInputBuffer, ScalarType,
};
use runmat_plot::plots::{
    scatter::ScatterGpuStyle, Line3Plot, LineStyle, MarkerStyle, ScatterPlot,
};
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll};
use wgpu::util::DeviceExt;

pub(super) async fn install_browser_source(values: &[f32]) -> (WgpuBufferOwner, WgpuDeviceAccess) {
    let instance = Arc::new(wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    }));
    let adapter = Arc::new(
        instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("this browser gate requires WebGPU"),
    );
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .expect("this browser gate requires a WebGPU device");
    let access = WgpuDeviceAccess::new(device, queue);
    let source = browser_buffer(&access, values);
    runmat_plot::install_shared_wgpu_context(runmat_plot::SharedWgpuContext {
        instance,
        access: access.clone(),
        adapter_info: adapter.get_info(),
        limits: adapter.limits(),
        features: adapter.features(),
        adapter,
    });
    (source, access)
}

fn browser_buffer(access: &WgpuDeviceAccess, values: &[f32]) -> WgpuBufferOwner {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    WgpuBufferOwner::new(
        access
            .with_shared(|emission| {
                let buffer =
                    emission
                        .device()
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("graphics-property-resident-coordinate"),
                            contents: &bytes,
                            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        });
                emission.queue().submit(std::iter::empty());
                buffer
            })
            .expect("WebGPU coordinate buffer"),
    )
}

async fn assert_pending_drop_and_stale(
    handle: f64,
    coordinate: &str,
    original: &[f64],
    expected_type: &str,
) {
    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("plot handle");
    assert_eq!(
        get_properties_async(target, Some("Type"), "get")
            .await
            .expect("metadata-only property"),
        Value::String(expected_type.into())
    );
    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("plot handle");
    set_properties_async(
        target,
        &[Value::String("LineWidth".into()), Value::Num(3.0)],
        "set",
    )
    .await
    .expect("style-only update");
    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("plot handle");
    let invalid = set_properties_async(
        target,
        &[
            Value::String(coordinate.into()),
            Value::String("invalid".into()),
        ],
        "set",
    )
    .await
    .expect_err("invalid coordinate control");
    assert!(!crate::builtins::plotting::error::plot_data_export_failure(
        &invalid
    ));

    let replacement = [
        Value::String(coordinate.into()),
        Value::Tensor(Tensor::new(vec![8.0, 9.0], vec![1, 2]).expect("replacement")),
    ];
    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("plot handle");
    let waker = futures::task::noop_waker();
    let mut task = Context::from_waker(&waker);
    let mut dropped = Box::pin(set_properties_async(target, &replacement, "set"));
    assert!(matches!(dropped.as_mut().poll(&mut task), Poll::Pending));
    drop(dropped);

    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("plot handle");
    let Value::Tensor(value) = get_properties_async(target, Some(coordinate), "get")
        .await
        .expect("dropped readback keeps source live")
    else {
        panic!("resident coordinate must remain a tensor");
    };
    assert_eq!(value.materialize_f64(), original);

    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("plot handle");
    let mut stale = Box::pin(set_properties_async(target, &replacement, "set"));
    assert!(matches!(stale.as_mut().poll(&mut task), Poll::Pending));
    let (child, _, _) = plot_child_element_snapshot(handle).expect("plot snapshot");
    let state = match child {
        PlotChildHandleState::Scatter(state) | PlotChildHandleState::Line3(state) => state,
        _ => panic!("expected resident Scatter or Line3"),
    };
    crate::builtins::plotting::state::update_plot_element(state.figure, state.plot_index, |_| {})
        .expect("intervening mutation");
    let error = stale.await.expect_err("stale readback cannot publish");
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    let mut stale_revision = false;
    while let Some(current) = source {
        stale_revision |= current
            .downcast_ref::<crate::builtins::plotting::state::FigureError>()
            .is_some_and(|error| {
                matches!(
                    error,
                    crate::builtins::plotting::state::FigureError::StalePlotSnapshot
                )
            });
        source = current.source();
    }
    assert!(stale_revision, "typed stale-revision cause must survive");
    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("plot handle");
    let Value::Tensor(value) = get_properties_async(target, Some(coordinate), "get")
        .await
        .expect("original data after stale update")
    else {
        panic!("resident coordinate must remain a tensor");
    };
    assert_eq!(value.materialize_f64(), original);
}

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_scatter_resident_property_lifecycle() {
    use runmat_plot::core::{BoundingBox, GpuVertexBuffer};

    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let scatter = host_scatter();
    let (x, access) = install_browser_source(&[1.25, -3.5]).await;
    let y = browser_buffer(&access, &[4.75, -2.0]);
    let (child, _, _) = plot_child_element_snapshot(scatter).expect("scatter snapshot");
    let PlotChildHandleState::Scatter(state) = child else {
        panic!("expected scatter child");
    };
    let vertices = WgpuBufferOwner::new(
        access
            .with_shared(|emission| {
                emission.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("scatter-property-vertex-placeholder"),
                    size: 16,
                    usage: wgpu::BufferUsages::VERTEX,
                    mapped_at_creation: false,
                })
            })
            .expect("WebGPU vertex buffer"),
    );
    crate::builtins::plotting::state::update_plot_element(state.figure, state.plot_index, |plot| {
        *plot = PlotElement::Scatter(
            ScatterPlot::from_gpu_buffer(
                GpuVertexBuffer::new(vertices, 0),
                2,
                BoundingBox::default(),
                ScatterGpuStyle {
                    color: glam::Vec4::ONE,
                    edge_color: glam::Vec4::ONE,
                    edge_thickness: 1.0,
                    marker_size: 8.0,
                    marker_style: MarkerStyle::Circle,
                    filled: false,
                    has_per_point_sizes: false,
                    has_per_point_colors: false,
                    edge_from_vertex_colors: false,
                },
            )
            .with_gpu_source_inputs(Scatter2GpuInputs {
                x_buffer: GpuInputBuffer::from(x.clone()),
                y_buffer: GpuInputBuffer::from(y.clone()),
                len: 2,
                scalar: ScalarType::F32,
            }),
        );
    })
    .expect("install resident scatter");
    assert_pending_drop_and_stale(scatter, "XData", &[1.25, -3.5], "scatter").await;
    let target = resolve_plot_handle(&Value::Num(scatter), "get").expect("scatter handle");
    let Value::Tensor(y_data) = get_properties_async(target, Some("YData"), "get")
        .await
        .expect("resident scatter Y")
    else {
        panic!("resident scatter Y must be a tensor");
    };
    assert_eq!(y_data.materialize_f64(), vec![4.75, -2.0]);
}

#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_line3_resident_property_lifecycle() {
    use runmat_plot::core::BoundingBox;

    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let line3 = host_line3();
    let (x, access) = install_browser_source(&[1.25, -3.5]).await;
    let y = browser_buffer(&access, &[4.75, -2.0]);
    let z = browser_buffer(&access, &[6.5, 0.5]);
    let (child, _, _) = plot_child_element_snapshot(line3).expect("Line3 snapshot");
    let PlotChildHandleState::Line3(state) = child else {
        panic!("expected Line3 child");
    };
    crate::builtins::plotting::state::update_plot_element(state.figure, state.plot_index, |plot| {
        *plot = PlotElement::Line3(Line3Plot::from_gpu_xyz(
            Line3GpuInputs {
                x_buffer: GpuInputBuffer::from(x.clone()),
                y_buffer: GpuInputBuffer::from(y.clone()),
                z_buffer: GpuInputBuffer::from(z.clone()),
                len: 2,
                scalar: ScalarType::F32,
            },
            glam::Vec4::ONE,
            1.0,
            LineStyle::Solid,
            BoundingBox::default(),
        ));
    })
    .expect("install resident Line3");
    assert_pending_drop_and_stale(line3, "ZData", &[6.5, 0.5], "line").await;
    for (name, expected) in [("XData", vec![1.25, -3.5]), ("YData", vec![4.75, -2.0])] {
        let target = resolve_plot_handle(&Value::Num(line3), "get").expect("Line3 handle");
        let Value::Tensor(data) = get_properties_async(target, Some(name), "get")
            .await
            .expect("resident Line3 coordinate")
        else {
            panic!("resident Line3 coordinate must be a tensor");
        };
        assert_eq!(data.materialize_f64(), expected);
    }
}
