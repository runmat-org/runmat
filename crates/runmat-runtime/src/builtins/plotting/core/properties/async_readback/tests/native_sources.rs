use super::*;
use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};
use runmat_plot::gpu::{
    line3::Line3GpuInputs, scatter2::Scatter2GpuInputs, GpuInputBuffer, ScalarType,
};
use runmat_plot::plots::{Line3Plot, LineStyle, ScatterPlot};

fn resident_data(handle: f64, name: &str) -> Vec<f64> {
    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("plot handle");
    let Value::Tensor(tensor) =
        block_on(get_properties_async(target, Some(name), "get")).expect("resident data")
    else {
        panic!("expected typed resident coordinate");
    };
    tensor.materialize_f64()
}

fn assert_selective_readback(handle: f64, coordinate: &str, expected_type: &str) {
    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("plot handle");
    assert_eq!(
        block_on(get_properties_async(target, Some("Type"), "get")).expect("metadata"),
        Value::String(expected_type.into())
    );
    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("plot handle");
    block_on(set_properties_async(
        target,
        &[Value::String("LineWidth".into()), Value::Num(3.0)],
        "set",
    ))
    .expect("style-only update must not read coordinates");
    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("plot handle");
    let invalid = block_on(set_properties_async(
        target,
        &[
            Value::String(coordinate.into()),
            Value::String("invalid".into()),
        ],
        "set",
    ))
    .expect_err("known-invalid coordinate must fail before readback");
    assert!(!crate::builtins::plotting::error::plot_data_export_failure(
        &invalid
    ));
    let target = resolve_plot_handle(&Value::Num(handle), "get").expect("plot handle");
    let readback = block_on(get_properties_async(target, Some(coordinate), "get"))
        .expect_err("declared length exceeds exported allocation");
    assert!(crate::builtins::plotting::error::plot_data_export_failure(
        &readback
    ));
}

#[test]
#[ignore = "requires an actual native WGPU adapter"]
fn scatter_exported_source_preserves_exact_values_and_borrowed_leases() {
    use runmat_plot::core::{BoundingBox, GpuVertexBuffer};
    use runmat_plot::plots::{scatter::ScatterGpuStyle, MarkerStyle};

    let _accel_guard = crate::builtins::common::test_support::accel_test_lock();
    let _plot_guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let handle = host_scatter();
    let provider =
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default())
            .expect("actual WGPU device required");
    let context = crate::builtins::plotting::context::ensure_context_from_provider()
        .expect("provider shared device");
    let x = provider
        .upload(&HostTensorView {
            data: &[1.25, -3.5],
            shape: &[1, 2],
        })
        .expect("x");
    let y = provider
        .upload(&HostTensorView {
            data: &[4.75, -2.0],
            shape: &[1, 2],
        })
        .expect("y");
    let x_wire = x.wire_descriptor();
    let y_wire = y.wire_descriptor();
    let x_export = runmat_accelerate_api::export_wgpu_buffer(&x)
        .unwrap()
        .expect("owned x export");
    let y_export = runmat_accelerate_api::export_wgpu_buffer(&y)
        .unwrap()
        .expect("owned y export");
    let scalar = ScalarType::from_is_f64(x_export.precision == ProviderPrecision::F64);
    let x_buffer = GpuInputBuffer::from(x_export);
    let y_buffer = GpuInputBuffer::from(y_export);
    let dummy_vertices = context
        .access
        .with_shared(|emission| {
            runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("scatter-property-vertex-placeholder"),
                    size: 16,
                    usage: wgpu::BufferUsages::VERTEX,
                    mapped_at_creation: false,
                },
            ))
        })
        .expect("shared device vertex buffer");
    let style = ScatterGpuStyle {
        color: glam::Vec4::ONE,
        edge_color: glam::Vec4::ONE,
        edge_thickness: 1.0,
        marker_size: 8.0,
        marker_style: MarkerStyle::Circle,
        filled: false,
        has_per_point_sizes: false,
        has_per_point_colors: false,
        edge_from_vertex_colors: false,
    };
    let make_plot = |len| {
        ScatterPlot::from_gpu_buffer(
            GpuVertexBuffer::new(dummy_vertices.clone(), 0),
            2,
            BoundingBox::default(),
            style,
        )
        .with_gpu_source_inputs(Scatter2GpuInputs {
            x_buffer: x_buffer.clone(),
            y_buffer: y_buffer.clone(),
            len,
            scalar,
        })
    };
    let (child, _, _) = plot_child_element_snapshot(handle).expect("scatter snapshot");
    let PlotChildHandleState::Scatter(state) = child else {
        panic!("expected scatter child");
    };
    crate::builtins::plotting::state::update_plot_element(state.figure, state.plot_index, |plot| {
        *plot = PlotElement::Scatter(make_plot(3));
    })
    .expect("install mismatched source extent");
    drop(x);
    drop(y);
    assert_selective_readback(handle, "XData", "scatter");
    assert!(provider.acquire_live(&x_wire).is_ok());
    assert!(provider.acquire_live(&y_wire).is_ok());

    crate::builtins::plotting::state::update_plot_element(state.figure, state.plot_index, |plot| {
        *plot = PlotElement::Scatter(make_plot(2));
    })
    .expect("install exact source extent");
    drop(x_buffer);
    drop(y_buffer);
    assert_eq!(resident_data(handle, "XData"), vec![1.25, -3.5]);
    assert_eq!(resident_data(handle, "YData"), vec![4.75, -2.0]);
    assert!(provider.acquire_live(&x_wire).is_ok());
    assert!(provider.acquire_live(&y_wire).is_ok());
    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("scatter handle");
    block_on(set_properties_async(
        target,
        &[
            Value::String("XData".into()),
            Value::Tensor(Tensor::new(vec![8.0, 9.0], vec![1, 2]).expect("new x")),
            Value::String("YData".into()),
            Value::Tensor(Tensor::new(vec![-6.0, 7.0], vec![1, 2]).expect("new y")),
        ],
        "set",
    ))
    .expect("replace scatter coordinates");
    assert!(provider.acquire_live(&x_wire).is_err());
    assert!(provider.acquire_live(&y_wire).is_err());
    assert_eq!(resident_data(handle, "XData"), vec![8.0, 9.0]);
    assert_eq!(resident_data(handle, "YData"), vec![-6.0, 7.0]);
}

#[test]
#[ignore = "requires an actual native WGPU adapter"]
fn line3_exported_source_preserves_exact_values_and_borrowed_leases() {
    let _accel_guard = crate::builtins::common::test_support::accel_test_lock();
    let _plot_guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let handle = host_line3();
    let provider =
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default())
            .expect("actual WGPU device required");
    crate::builtins::plotting::context::ensure_context_from_provider()
        .expect("provider shared device");
    let upload = |values: &[f64]| {
        provider
            .upload(&HostTensorView {
                data: values,
                shape: &[1, 2],
            })
            .expect("coordinate upload")
    };
    let x = upload(&[1.25, -3.5]);
    let y = upload(&[4.75, -2.0]);
    let z = upload(&[6.5, 0.5]);
    let wires = [
        x.wire_descriptor(),
        y.wire_descriptor(),
        z.wire_descriptor(),
    ];
    let x_export = runmat_accelerate_api::export_wgpu_buffer(&x)
        .unwrap()
        .expect("owned x export");
    let y_export = runmat_accelerate_api::export_wgpu_buffer(&y)
        .unwrap()
        .expect("owned y export");
    let z_export = runmat_accelerate_api::export_wgpu_buffer(&z)
        .unwrap()
        .expect("owned z export");
    let scalar = ScalarType::from_is_f64(x_export.precision == ProviderPrecision::F64);
    let x_buffer = GpuInputBuffer::from(x_export);
    let y_buffer = GpuInputBuffer::from(y_export);
    let z_buffer = GpuInputBuffer::from(z_export);
    let make_plot = |len| {
        Line3Plot::from_gpu_xyz(
            Line3GpuInputs {
                x_buffer: x_buffer.clone(),
                y_buffer: y_buffer.clone(),
                z_buffer: z_buffer.clone(),
                len,
                scalar,
            },
            glam::Vec4::ONE,
            1.0,
            LineStyle::Solid,
            runmat_plot::core::BoundingBox::default(),
        )
    };
    let (child, _, _) = plot_child_element_snapshot(handle).expect("plot3 snapshot");
    let PlotChildHandleState::Line3(state) = child else {
        panic!("expected plot3 child");
    };
    crate::builtins::plotting::state::update_plot_element(state.figure, state.plot_index, |plot| {
        *plot = PlotElement::Line3(make_plot(3));
    })
    .expect("install mismatched source extent");
    drop(x);
    drop(y);
    drop(z);
    assert_selective_readback(handle, "ZData", "line");
    for wire in &wires {
        assert!(provider.acquire_live(wire).is_ok());
    }

    crate::builtins::plotting::state::update_plot_element(state.figure, state.plot_index, |plot| {
        *plot = PlotElement::Line3(make_plot(2));
    })
    .expect("install exact source extent");
    drop(x_buffer);
    drop(y_buffer);
    drop(z_buffer);
    assert_eq!(resident_data(handle, "XData"), vec![1.25, -3.5]);
    assert_eq!(resident_data(handle, "YData"), vec![4.75, -2.0]);
    assert_eq!(resident_data(handle, "ZData"), vec![6.5, 0.5]);
    for wire in &wires {
        assert!(provider.acquire_live(wire).is_ok());
    }
    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("plot3 handle");
    block_on(set_properties_async(
        target,
        &[
            Value::String("XData".into()),
            Value::Tensor(Tensor::new(vec![8.0, 9.0], vec![1, 2]).expect("new x")),
            Value::String("YData".into()),
            Value::Tensor(Tensor::new(vec![-6.0, 7.0], vec![1, 2]).expect("new y")),
            Value::String("ZData".into()),
            Value::Tensor(Tensor::new(vec![10.0, 11.0], vec![1, 2]).expect("new z")),
        ],
        "set",
    ))
    .expect("replace plot3 coordinates");
    for wire in &wires {
        assert!(provider.acquire_live(wire).is_err());
    }
    assert_eq!(resident_data(handle, "XData"), vec![8.0, 9.0]);
    assert_eq!(resident_data(handle, "YData"), vec![-6.0, 7.0]);
    assert_eq!(resident_data(handle, "ZData"), vec![10.0, 11.0]);
}
