use super::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_resident_line_pending_set_drop_and_stale_revision() {
    use runmat_plot::gpu::{line::LineGpuInputs, GpuInputBuffer, ScalarType};
    use runmat_plot::plots::{LineGpuStyle, LinePlot, LineStyle};
    use std::future::Future;
    use std::task::{Context, Poll};

    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    crate::builtins::plotting::reset_plot_state();
    let handle = host_line();

    let (source, _access) = super::browser_sources::install_browser_source(&[1.25, -3.5]).await;
    let (child, _, _) = plot_child_element_snapshot(handle).expect("line snapshot");
    let PlotChildHandleState::Line(state) = child else {
        panic!("expected line child");
    };
    let figure = state.figure;
    let plot_index = state.plot_index;
    crate::builtins::plotting::state::update_plot_element(figure, plot_index, |plot| {
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
    let Value::Tensor(ref original_tensor) = original else {
        panic!("expected resident single tensor");
    };
    assert_eq!(original_tensor.materialize_f64(), vec![1.25, -3.5]);

    let target = resolve_plot_handle(&Value::Num(handle), "set").expect("handle");
    let mut stale = Box::pin(set_properties_async(target, &replacement, "set"));
    assert!(matches!(stale.as_mut().poll(&mut context), Poll::Pending));
    crate::builtins::plotting::state::update_plot_element(figure, plot_index, |_| {})
        .expect("intervening mutation");
    let error = stale
        .await
        .expect_err("stale async update must not publish");
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
