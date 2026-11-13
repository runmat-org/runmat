//! Real selected invocation contexts, never constructed mutation proofs.
use super::*;
use crate::builtins::plotting::state::{
    axes_graphics_receiver_for_scene, capture_figure_persistence_snapshot,
    figure_graphics_receiver, reset_plot_state, select_figure, FigureHandle,
};
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::execution::RuntimeExecutionService;
use runmat_value::{Tensor, Value};
use std::cell::RefCell;
use std::future::Future;

fn source() -> (
    FigureHandle,
    GraphicsReceiverRef,
    GraphicsCallbackRolePayload,
) {
    reset_plot_state();
    let figure = FigureHandle::from(46);
    select_figure(figure);
    let receiver = figure_graphics_receiver(figure).unwrap().unwrap();
    let snapshot = capture_figure_persistence_snapshot(&[figure], 1, usize::MAX).unwrap();
    let scene = snapshot.entries[0].scene_lease.clone();
    let axes = axes_graphics_receiver_for_scene(&receiver, &scene, 0)
        .unwrap()
        .unwrap();
    let payload = GraphicsCallbackRolePayload::Action {
        figure: receiver.clone(),
        scene,
        axes,
    };
    (figure, receiver, payload)
}

fn owner() -> (RuntimeGraphicsCallbackQueue, RuntimeContext) {
    let base = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let queue = RuntimeGraphicsCallbackQueue::new_for_selected(
        &base,
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(1).unwrap(),
    );
    let ports = base
        .service_ports()
        .clone()
        .with_graphics_callback_queue(queue.downgrade());
    (queue, base.with_service_ports(ports))
}

#[test]
fn selected_callback_ordinary_mutations_publish_only_their_own_successor() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    for name in ["set", "plot", "cla", "axes"] {
        let (figure, receiver, payload) = source();
        let (queue, selected) = owner();
        let axes = match &payload {
            GraphicsCallbackRolePayload::Action { axes, .. } => axes.clone(),
            _ => unreachable!(),
        };
        let callback: Rc<GraphicsSelectedBoundCallbackResponder> = Rc::new(move |runtime, _, _| {
            let axes = axes.clone();
            Box::pin(async move {
                let args = match name {
                    "set" => vec![
                        crate::builtins::plotting::state::graphics_receiver_object(axes)
                            .unwrap()
                            .unwrap(),
                        Value::String("XLim".into()),
                        Value::Tensor(Tensor::new(vec![0.0, 4.0], vec![1, 2]).unwrap()),
                    ],
                    "plot" => vec![
                        crate::builtins::plotting::state::graphics_receiver_object(axes)
                            .unwrap()
                            .unwrap(),
                        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
                    ],
                    "cla" | "axes" => vec![
                        crate::builtins::plotting::state::graphics_receiver_object(axes)
                            .unwrap()
                            .unwrap(),
                    ],
                    _ => unreachable!(),
                };
                crate::call_builtin_async_with_outputs_in_context(&runtime, name, &args, 0).await?;
                Ok(GraphicsCallbackResult::Completed)
            })
        });
        let kind = GraphicsCallbackKind::PanActionPre;
        let ingress = queue
            .replace_selected_graphics_bound_property_response(selected, receiver, kind, callback)
            .unwrap();
        let ticket = ingress
            .request_bound(
                GraphicsCallbackHostEvent {
                    figure: figure.as_u32(),
                    kind,
                    position: None,
                },
                payload,
            )
            .unwrap();
        futures::executor::block_on(queue.process())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            futures::executor::block_on(ticket),
            Ok(GraphicsCallbackResult::Completed),
            "{name}"
        );
    }
}

#[test]
fn dropping_pending_callback_revokes_retained_invocation_authority() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    let (figure, receiver, payload) = source();
    let (queue, selected) = owner();
    let retained = Rc::new(RefCell::new(None));
    let callback: Rc<GraphicsSelectedBoundCallbackResponder> = {
        let retained = retained.clone();
        Rc::new(move |runtime, _, _| {
            *retained.borrow_mut() = Some(runtime);
            Box::pin(futures::future::pending())
        })
    };
    let kind = GraphicsCallbackKind::PanActionPre;
    let ingress = queue
        .replace_selected_graphics_bound_property_response(selected, receiver, kind, callback)
        .unwrap();
    let ticket = ingress
        .request_bound(
            GraphicsCallbackHostEvent {
                figure: figure.as_u32(),
                kind,
                position: None,
            },
            payload,
        )
        .unwrap();
    let mut processing = Box::pin(queue.process());
    futures::executor::block_on(futures::future::poll_fn(|cx| {
        assert!(processing.as_mut().poll(cx).is_pending());
        std::task::Poll::Ready(())
    }));
    let runtime = retained
        .borrow_mut()
        .take()
        .expect("actual selected responder ran");
    drop(processing);
    let property = crate::builtins::plotting::state::PropertyMutationTarget::alias()
        .with_runtime(Some(&runtime));
    assert!(
        crate::builtins::plotting::state::set_current_x_limits_for_property(
            &property,
            Some((0.0, 10.0))
        )
        .is_err()
    );
    assert_eq!(
        ticket.disposition(),
        GraphicsCallbackDisposition::Indeterminate
    );
}
