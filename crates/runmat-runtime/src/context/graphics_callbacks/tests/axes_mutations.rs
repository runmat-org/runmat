//! Actual canonical axes calls carry the callback invocation's own-successor
//! proof through their shared state owners; ordinary aliases cannot supply it.

use super::*;
use crate::builtins::plotting::state::{
    axes_graphics_receiver_for_scene, capture_figure_persistence_snapshot, current_figure_handle,
    encode_axes_handle, figure_graphics_receiver, figure_handle_exists, graphics_receiver_object,
    reset_plot_state, select_figure, FigureHandle,
};
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};

#[test]
fn canonical_axes_controls_keep_authentic_callback_successors() {
    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    for name in [
        "xlim",
        "xscale",
        "xticklabels",
        "colororder",
        "subplot",
        "gca",
    ] {
        for exact in [false, true] {
            reset_plot_state();
            let figure = FigureHandle::from(84);
            select_figure(figure);
            let receiver = figure_graphics_receiver(figure).unwrap().unwrap();
            let snapshot = capture_figure_persistence_snapshot(&[figure], 1, usize::MAX).unwrap();
            let scene = snapshot.entries[0].scene_lease.clone();
            let axes = axes_graphics_receiver_for_scene(&receiver, &scene, 0)
                .unwrap()
                .unwrap();
            let target = if exact {
                graphics_receiver_object(axes.clone()).unwrap().unwrap()
            } else {
                Value::Num(encode_axes_handle(figure, 0))
            };
            let args = match name {
                "xlim" => vec![
                    target.clone(),
                    Value::Tensor(runmat_value::Tensor::new(vec![2.0, 9.0], vec![1, 2]).unwrap()),
                ],
                "xscale" if exact => vec![target.clone(), Value::String("log".into())],
                "xscale" => vec![Value::String("log".into())],
                "xticklabels" if exact => {
                    vec![target.clone(), Value::String("chosen label".into())]
                }
                "xticklabels" => vec![Value::String("chosen label".into())],
                "colororder" => vec![target, Value::String("gem".into())],
                "subplot" => vec![Value::Num(1.0), Value::Num(1.0), Value::Num(1.0)],
                "gca" if exact => {
                    vec![graphics_receiver_object(receiver.clone()).unwrap().unwrap()]
                }
                "gca" => Vec::new(),
                _ => unreachable!("closed fixture operations"),
            };
            // Exact targets must not rediscover their coordinates through the
            // globally current Figure. Subplot intentionally has no target form.
            let current = if exact && name != "subplot" {
                let other = FigureHandle::from(85);
                select_figure(other);
                other
            } else {
                figure
            };
            let base = selected();
            // The explicit Figure argument is a declared RunMat extension;
            // other forms keep the fixture's conservative MATLAB policy.
            base.set_runmat_extensions_enabled(name == "gca" && exact);
            let queue = RuntimeGraphicsCallbackQueue::new_for_selected(
                &base,
                NonZeroUsize::new(1).unwrap(),
                NonZeroUsize::new(1).unwrap(),
            );
            let ports = base
                .service_ports()
                .clone()
                .with_graphics_callback_queue(queue.downgrade());
            let selected = base.with_service_ports(ports);
            let calls = Rc::new(Cell::new(0));
            let callback: Rc<GraphicsSelectedBoundCallbackResponder> = {
                let calls = calls.clone();
                Rc::new(move |runtime, _, _| {
                    let args = args.clone();
                    let calls = calls.clone();
                    Box::pin(async move {
                        // A second call must admit this invocation's completed
                        // successor, not replay against the original scene.
                        for _ in 0..2 {
                            crate::call_builtin_async_with_outputs_in_context(
                                &runtime, name, &args, 0,
                            )
                            .await?;
                            calls.set(calls.get() + 1);
                        }
                        Ok(GraphicsCallbackResult::Completed)
                    })
                })
            };
            let kind = GraphicsCallbackKind::PanActionPre;
            let ingress = queue
                .replace_selected_graphics_bound_property_response(
                    selected,
                    receiver.clone(),
                    kind,
                    callback,
                )
                .unwrap();
            let ticket = ingress
                .request_bound(
                    GraphicsCallbackHostEvent {
                        figure: figure.as_u32(),
                        kind,
                        position: None,
                    },
                    GraphicsCallbackRolePayload::Action {
                        figure: receiver,
                        scene,
                        axes,
                    },
                )
                .unwrap();
            futures::executor::block_on(queue.process())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                futures::executor::block_on(ticket),
                Ok(GraphicsCallbackResult::Completed),
                "{name}"
            );
            assert_eq!(calls.get(), 2, "{name}");
            assert_eq!(
                current_figure_handle(),
                current,
                "{name} reselected its target"
            );
        }
    }

    // Failed control/output admission must not manufacture the implicit Figure,
    // even through the actual context-bearing registered entry point.
    reset_plot_state();
    let implicit = current_figure_handle();
    assert!(!figure_handle_exists(implicit));
    let runtime = selected();
    for (name, args, outputs) in [
        ("xlim", vec![Value::String("invalid-mode".into())], 0),
        (
            "xticks",
            vec![Value::Tensor(
                runmat_value::Tensor::new(vec![3.0, 2.0], vec![1, 2]).unwrap(),
            )],
            0,
        ),
        ("gca", Vec::new(), 2),
    ] {
        assert!(
            futures::executor::block_on(crate::call_builtin_async_with_outputs_in_context(
                &runtime, name, &args, outputs
            ),)
            .is_err(),
            "{name}"
        );
        assert!(
            !figure_handle_exists(implicit),
            "{name} created state before admission"
        );
    }
}
