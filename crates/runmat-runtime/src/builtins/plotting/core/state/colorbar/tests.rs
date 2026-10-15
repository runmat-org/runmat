use super::*;
use crate::builtins::plotting::properties::{get_properties, resolve_plot_handle, set_properties};
use runmat_value::Value;

fn setup() -> PlotTestLockGuard {
    let guard = crate::builtins::plotting::tests::lock_plot_registry();
    reset_plot_state();
    guard
}

#[test]
fn repeated_publication_keeps_identity_and_hidden_is_not_deleted() {
    let _guard = setup();
    let axes = current_axes_state();
    let bar = ensure_colorbar_for_axes(axes.handle, axes.active_index).unwrap();
    let repeated = ensure_colorbar_for_axes(axes.handle, axes.active_index).unwrap();
    assert_eq!(bar, repeated);
    let next = registry().next_plot_child_handle;
    let figure = clone_figure(axes.handle).unwrap();
    notify_with_figure(axes.handle, &figure, FigureEventKind::Updated);
    assert_eq!(registry().next_plot_child_handle, next);
    set_colorbar_visible_for_property(&PropertyMutationTarget::alias(), bar, false).unwrap();
    let resolved = resolve_plot_handle(&Value::Num(bar), "get").unwrap();
    assert_eq!(
        get_properties(resolved.clone(), Some("Visible"), "get").unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        colorbar_handle_for_axes(axes.handle, axes.active_index),
        Some(bar)
    );
    assert!(
        !clone_figure(axes.handle)
            .unwrap()
            .axes_metadata(axes.active_index)
            .unwrap()
            .colorbar_enabled
    );
    set_properties(
        resolved,
        &[Value::String("Visible".into()), Value::Bool(true)],
        "set",
    )
    .unwrap();
    assert!(
        clone_figure(axes.handle)
            .unwrap()
            .axes_metadata(axes.active_index)
            .unwrap()
            .colorbar_enabled
    );
    set_colorbar_enabled_for_axes(axes.handle, axes.active_index, false).unwrap();
    assert!(plot_child_handle_snapshot(bar).is_err());
    let replacement = ensure_colorbar_for_axes(axes.handle, axes.active_index).unwrap();
    assert!(replacement > bar);
    assert!(plot_child_handle_snapshot(bar).is_err());
}

#[test]
fn imported_and_builder_colorbars_are_registered_before_reentrant_observation() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let _guard = setup();
    let enabled = Arc::new(AtomicBool::new(true));
    let observed = Arc::new(AtomicUsize::new(0));
    let enabled_observer = enabled.clone();
    let observed_callback = observed.clone();
    install_figure_observer(Arc::new(move |view| {
        if !enabled_observer.load(Ordering::SeqCst) {
            return;
        }
        if let Some(figure) = view.figure {
            for (index, meta) in figure.axes_metadata.iter().enumerate() {
                if meta.colorbar_enabled {
                    let bar = colorbar_handle_for_axes(view.handle, index)
                        .expect("identity before callback");
                    assert!(plot_child_handle_snapshot(bar).is_ok());
                    observed_callback.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
    }))
    .unwrap();
    // Disable this persistent observer even if a later assertion unwinds.
    struct Disable(Arc<AtomicBool>);
    impl Drop for Disable {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    let _disable = Disable(enabled);
    let mut figure = default_figure("", "", "");
    figure.set_axes_colorbar_enabled(0, true);
    let imported = import_figure(figure);
    let bar = colorbar_handle_for_axes(imported, 0).unwrap();
    clear_figure(Some(imported)).unwrap();
    assert!(plot_child_handle_snapshot(bar).is_err());
    render_active_plot(
        "colorbar-test",
        PlotRenderOptions::default(),
        |figure, axes| {
            figure.set_axes_colorbar_enabled(axes, true);
            Ok(())
        },
    )
    .unwrap();
    let replacement = colorbar_handle_for_axes(imported, 0).unwrap();
    assert!(replacement > bar);
    assert!(observed.load(Ordering::SeqCst) >= 2);
    close_figure(Some(imported)).unwrap();
    assert!(plot_child_handle_snapshot(replacement).is_err());
}

#[test]
fn figure_parent_and_children_are_consistent_and_cla_keeps_colorbar() {
    let _guard = setup();
    let axes = current_axes_state();
    let bar = ensure_colorbar_for_axes(axes.handle, axes.active_index).unwrap();
    let child = resolve_plot_handle(&Value::Num(bar), "get").unwrap();
    assert_eq!(
        get_properties(child, Some("Parent"), "get").unwrap(),
        Value::Num(axes.handle.as_u32() as f64)
    );
    let figure = resolve_plot_handle(&Value::Num(axes.handle.as_u32() as f64), "get").unwrap();
    let Value::Tensor(children) = get_properties(figure, Some("Children"), "get").unwrap() else {
        panic!("numeric children");
    };
    assert!(children.materialize_f64().contains(&bar));
    assert!(!plot_child_handles_for_axes(axes.handle, axes.active_index).contains(&bar));
    clear_current_axes();
    assert_eq!(
        colorbar_handle_for_axes(axes.handle, axes.active_index),
        Some(bar)
    );
    assert!(validate_plot_child_copy_source(bar).is_err());
}

#[test]
fn removed_axes_invalidate_only_associated_colorbars() {
    let _guard = setup();
    let mut figure = default_figure("", "", "");
    figure.set_axes_colorbar_enabled(0, true);
    figure.set_axes_colorbar_enabled(1, true);
    let handle = import_figure(figure);
    let retained = colorbar_handle_for_axes(handle, 0).unwrap();
    let removed = colorbar_handle_for_axes(handle, 1).unwrap();
    let snapshot = {
        let mut reg = registry();
        let state = reg.figures.get_mut(&handle).unwrap();
        state.figure.axes_metadata.truncate(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &snapshot, FigureEventKind::Updated);
    assert!(plot_child_handle_snapshot(removed).is_err());
    assert_eq!(colorbar_handle_for_axes(handle, 0), Some(retained));
}
