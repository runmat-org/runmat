//! Async plot-source materialization at the graphics property boundary.
//! Snapshotting releases the registry guard before a provider readback awaits.
use super::super::state::{plot_child_element_snapshot, PlotChildHandleState};
use super::*;
use runmat_plot::plots::figure::PlotElement;

async fn line_xy(
    line: &runmat_plot::plots::LinePlot,
    builtin: &'static str,
) -> BuiltinResult<PlotXY> {
    line.export_numeric_xy_data_typed()
        .await
        .map_err(|err| map_figure_error(builtin, err))?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: line has no source data")))
}

async fn scatter_xy(
    scatter: &runmat_plot::plots::ScatterPlot,
    builtin: &'static str,
) -> BuiltinResult<PlotXY> {
    scatter
        .export_numeric_xy_data_typed()
        .await
        .map_err(|err| map_figure_error(builtin, err))?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: scatter has no source data")))
}

async fn line3_xyz(
    line: &runmat_plot::plots::Line3Plot,
    builtin: &'static str,
) -> BuiltinResult<PlotXYZ> {
    line.export_numeric_xyz_data_typed()
        .await
        .map_err(|err| map_figure_error(builtin, err))?
        .ok_or_else(|| plotting_error(builtin, format!("{builtin}: plot3 has no source data")))
}

pub async fn get_properties_async(
    handle: PlotHandle,
    property: Option<&str>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let PlotHandle::PlotChild(id, state) = &handle else {
        return get_properties(handle, property, builtin);
    };
    if !matches!(
        state.as_ref(),
        PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_)
    ) {
        return get_properties(handle, property, builtin);
    }
    let (child, plot, _) =
        plot_child_element_snapshot(*id).map_err(|err| map_figure_error(builtin, err))?;
    match (child, plot) {
        (PlotChildHandleState::Line(state), PlotElement::Line(line)) => {
            if let SourcePropertyRead::Ready(value) =
                line_property_without_data(&state, property, builtin, &line)?
            {
                return Ok(value);
            }
            let (x, y) = line_xy(&line, builtin).await?;
            line_property_with_data(&state, property, builtin, line, x, y)
        }
        (PlotChildHandleState::Scatter(state), PlotElement::Scatter(scatter)) => {
            if let SourcePropertyRead::Ready(value) =
                scatter_property_without_data(&state, property, builtin, &scatter)?
            {
                return Ok(value);
            }
            let (x, y) = scatter_xy(&scatter, builtin).await?;
            scatter_property_with_data(&state, property, builtin, scatter, x, y)
        }
        (PlotChildHandleState::Line3(state), PlotElement::Line3(line))
            if line3_property_needs_data(property) =>
        {
            let data = line3_xyz(&line, builtin).await?;
            line3_property_with_data(&state, property, builtin, line, Some(data))
        }
        (PlotChildHandleState::Line3(state), PlotElement::Line3(line)) => {
            line3_property_with_data(&state, property, builtin, line, None)
        }
        (
            PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_),
            _,
        ) => Err(plotting_error(
            builtin,
            format!("{builtin}: invalid plot handle"),
        )),
        _ => get_properties(handle, property, builtin),
    }
}

pub async fn set_properties_async(
    handle: PlotHandle,
    args: &[Value],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let PlotHandle::PlotChild(id, state) = &handle else {
        return set_properties(handle, args, builtin);
    };
    if !matches!(
        state.as_ref(),
        PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_)
    ) {
        return set_properties(handle, args, builtin);
    }
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return Err(plotting_error(
            builtin,
            format!("{builtin}: property/value arguments must come in pairs"),
        ));
    }
    let mut pairs = Vec::with_capacity(args.len() / 2);
    for pair in args.chunks_exact(2) {
        pairs.push((property_name(&pair[0], builtin)?, &pair[1]));
    }
    let (child, plot, revision) =
        plot_child_element_snapshot(*id).map_err(|err| map_figure_error(builtin, err))?;
    let guard = PlotUpdateGuard {
        child_handle: *id,
        revision,
    };
    match (child, plot) {
        (PlotChildHandleState::Line(state), PlotElement::Line(line)) => {
            let prepared = prepare_line_data_updates(&pairs, builtin)?;
            if prepared.is_empty() {
                return apply_line_style_only(&state, &pairs, builtin, guard);
            }
            let data = line_xy(&line, builtin).await?;
            apply_line_properties_with_data(&state, &pairs, builtin, data, prepared, Some(guard))
        }
        (PlotChildHandleState::Scatter(state), PlotElement::Scatter(scatter)) => {
            let prepared = prepare_scatter_data_updates(&pairs, builtin)?;
            if prepared.is_empty() {
                return apply_scatter_style_only(&state, &pairs, builtin, guard);
            }
            let data = scatter_xy(&scatter, builtin).await?;
            apply_scatter_properties_with_data(
                &state,
                &pairs,
                builtin,
                scatter,
                data,
                prepared,
                Some(guard),
            )
        }
        (PlotChildHandleState::Line3(state), PlotElement::Line3(line)) => {
            let prepared = prepare_line3_data_updates(&pairs, builtin)?;
            if prepared.is_empty() {
                return apply_line3_style_only(&state, &pairs, builtin, guard);
            }
            let data = line3_xyz(&line, builtin).await?;
            apply_line3_properties_with_data(&state, &pairs, builtin, data, prepared, Some(guard))
        }
        (
            PlotChildHandleState::Line(_)
            | PlotChildHandleState::Scatter(_)
            | PlotChildHandleState::Line3(_),
            _,
        ) => Err(plotting_error(
            builtin,
            format!("{builtin}: invalid plot handle"),
        )),
        _ => set_properties(handle, args, builtin),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::line::line_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use futures::executor::block_on;

    fn host_line() -> f64 {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("x"));
        let y = Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![1, 2]).expect("y"));
        let Value::Num(handle) = block_on(line_builtin(vec![x, y])).expect("line") else {
            panic!("expected graphics handle");
        };
        handle
    }

    fn x_data(handle: f64) -> Value {
        let plot = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
        block_on(get_properties_async(plot, Some("XData"), "get")).expect("XData")
    }

    #[test]
    fn async_get_preserves_host_numeric_source_and_unpolled_set_has_no_effect() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        crate::builtins::plotting::reset_plot_state();
        let handle = host_line();
        let before = x_data(handle);
        let plot = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
        let args = [
            Value::String("XData".into()),
            Value::Tensor(Tensor::new(vec![8.0, 9.0], vec![1, 2]).expect("replacement")),
        ];
        let future = set_properties_async(plot, &args, "set");
        drop(future);
        assert_eq!(x_data(handle), before);
    }

    #[test]
    fn stale_snapshot_refuses_atomic_property_update() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        crate::builtins::plotting::reset_plot_state();
        let handle = host_line();
        let (child, _, revision) = plot_child_element_snapshot(handle).expect("snapshot");
        let PlotChildHandleState::Line(state) = child else {
            panic!("expected line child");
        };
        let figure = state.figure;
        let plot_index = state.plot_index;
        super::super::super::state::update_plot_element(figure, plot_index, |_| {})
            .expect("intervening mutation");
        let mut called = false;
        let error = super::super::super::state::update_plot_element_if_current(
            handle,
            figure,
            plot_index,
            revision,
            |_| called = true,
        )
        .expect_err("stale revision");
        assert!(matches!(
            error,
            super::super::super::state::FigureError::StalePlotSnapshot
        ));
        assert!(!called);
    }

    #[test]
    fn invalid_async_set_does_not_mutate_host_line() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        crate::builtins::plotting::reset_plot_state();
        let handle = host_line();
        let before = x_data(handle);
        let plot = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
        let args = [
            Value::String("XData".into()),
            Value::Tensor(Tensor::new(vec![8.0, 9.0, 10.0], vec![1, 3]).expect("mismatch")),
        ];
        let error =
            block_on(set_properties_async(plot, &args, "set")).expect_err("mismatched lengths");
        assert!(error.message.contains("same number of elements"));
        assert_eq!(x_data(handle), before);
    }

    #[test]
    fn invalid_coordinate_input_is_rejected_by_host_preflight() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        crate::builtins::plotting::reset_plot_state();
        let handle = host_line();
        let before = x_data(handle);
        let plot = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
        let args = [
            Value::String("XData".into()),
            Value::String("invalid".into()),
        ];
        let error = block_on(set_properties_async(plot, &args, "set"))
            .expect_err("host-invalid coordinate must fail");
        assert!(error.message.contains("XData must be numeric"));
        assert_eq!(x_data(handle), before);
    }

    #[test]
    fn metadata_queries_and_style_only_updates_do_not_require_coordinate_materialization() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        crate::builtins::plotting::reset_plot_state();
        let handle = host_line();
        let (child, plot, _) = plot_child_element_snapshot(handle).expect("snapshot");
        let (PlotChildHandleState::Line(state), PlotElement::Line(line)) = (child, plot) else {
            panic!("expected line");
        };
        assert!(matches!(
            line_property_without_data(&state, Some("Type"), "get", &line).expect("metadata"),
            SourcePropertyRead::Ready(Value::String(value)) if value == "line"
        ));
        assert!(matches!(
            line_property_without_data(&state, Some("XData"), "get", &line)
                .expect("known coordinate property"),
            SourcePropertyRead::RequiresCoordinates
        ));
        assert!(line_property_without_data(&state, Some("NotAProperty"), "get", &line).is_err());

        let before = x_data(handle);
        let target = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
        block_on(set_properties_async(
            target,
            &[Value::String("LineWidth".into()), Value::Num(3.0)],
            "set",
        ))
        .expect("style update");
        assert_eq!(x_data(handle), before);
        let target = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
        assert_eq!(
            block_on(get_properties_async(target, Some("LineWidth"), "get")).expect("style"),
            Value::Num(3.0)
        );
    }

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
        let short_source = Arc::new(
            access
                .with_shared(|emission| {
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("graphics-property-short-resident-source"),
                        size: 4,
                        usage: wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                })
                .expect("create resident-only source"),
        );
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
        super::super::super::state::update_plot_element(state.figure, state.plot_index, |plot| {
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
        assert!(!super::super::super::error::plot_data_export_failure(
            &invalid
        ));

        let target = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
        let readback = block_on(get_properties_async(target, Some("XData"), "get"))
            .expect_err("undersized resident source must fail when coordinates are requested");
        assert!(super::super::super::error::plot_data_export_failure(
            &readback
        ));
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test(async)]
    async fn browser_resident_line_pending_set_drop_and_stale_revision() {
        use runmat_plot::gpu::{line::LineGpuInputs, GpuInputBuffer, ScalarType};
        use runmat_plot::plots::{LineGpuStyle, LinePlot, LineStyle};
        use std::future::Future;
        use std::sync::Arc;
        use std::task::{Context, Poll};

        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        crate::builtins::plotting::reset_plot_state();
        let handle = host_line();

        let instance = Arc::new(wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        }));
        let adapter = Arc::new(
            instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .expect("browser hardware gate requires a WebGPU adapter"),
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("browser hardware gate requires a WebGPU device");
        let access = runmat_accelerate_api::WgpuDeviceAccess::new(device, queue);
        let source = Arc::new(
            access
                .with_shared(|emission| {
                    emission.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("browser-graphics-property-resident-source"),
                        size: 8,
                        usage: wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                })
                .expect("create resident source"),
        );
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
        let figure = state.figure;
        let plot_index = state.plot_index;
        super::super::super::state::update_plot_element(figure, plot_index, |plot| {
            *plot = PlotElement::Line(LinePlot::from_gpu_xy(
                LineGpuInputs {
                    x_buffer: GpuInputBuffer::from(source.clone()),
                    y_buffer: GpuInputBuffer::from(source.clone()),
                    len: 2,
                    scalar: ScalarType::F32,
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
        .expect("install resident-only line");

        let replacement = [
            Value::String("XData".into()),
            Value::Tensor(Tensor::new(vec![8.0, 9.0], vec![1, 2]).expect("replacement")),
        ];
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        let target = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
        let mut dropped = Box::pin(set_properties_async(target, &replacement, "set"));
        assert!(matches!(dropped.as_mut().poll(&mut context), Poll::Pending));
        drop(dropped);
        let (_, plot, _) = plot_child_element_snapshot(handle).expect("source remains live");
        assert!(matches!(plot, PlotElement::Line(ref line) if line.has_gpu_source_data()));
        let target = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
        let original = get_properties_async(target, Some("XData"), "get")
            .await
            .expect("cancelled readback remains reusable");
        assert_ne!(original, replacement[1]);

        let target = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
        let mut stale = Box::pin(set_properties_async(target, &replacement, "set"));
        assert!(matches!(stale.as_mut().poll(&mut context), Poll::Pending));
        super::super::super::state::update_plot_element(figure, plot_index, |_| {})
            .expect("intervening mutation");
        let error = stale
            .await
            .expect_err("stale async update must not publish");
        let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&error);
        let mut stale_revision = false;
        while let Some(current) = source {
            stale_revision |= current
                .downcast_ref::<super::super::super::state::FigureError>()
                .is_some_and(|error| {
                    matches!(
                        error,
                        super::super::super::state::FigureError::StalePlotSnapshot
                    )
                });
            source = current.source();
        }
        assert!(
            stale_revision,
            "must preserve the typed stale-revision cause"
        );
        let target = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
        assert_eq!(
            get_properties_async(target, Some("XData"), "get")
                .await
                .expect("original data remains readable"),
            original
        );
    }
}
