use super::*;
use crate::builtins::plotting::state::{self, FigureEventKind};
use crate::builtins::plotting::tests::lock_plot_registry;
use std::sync::{Arc, Mutex};

#[test]
fn gcf_creates_once_preserves_selection_and_recreates_after_close() {
    let _lock = lock_plot_registry();
    state::reset_plot_state();
    let events = Arc::new(Mutex::new(Vec::new()));
    // Observer registry owns callbacks permanently; keep only a weak witness.
    let weak = Arc::downgrade(&events);
    state::install_figure_observer(Arc::new(move |view| {
        if let Some(events) = weak.upgrade() {
            events.lock().unwrap().push((view.handle, view.kind));
        }
    }))
    .unwrap();
    assert!(state::current_figure_handle_if_exists().is_none());
    let first = gcf_builtin().unwrap();
    let handle = state::current_figure_handle_if_exists().unwrap();
    assert_eq!(first, handle.as_u32() as f64);
    let (_, extra_axes) = state::create_axes_for_figure(Some(handle)).unwrap();
    assert!(extra_axes > 0);
    let axes = state::current_axes_state();
    let axes_handles = state::axes_handles_for_figure(handle).unwrap();
    let revision = state::current_figure_revision(handle);
    let events_before_gcf = events.lock().unwrap().clone();
    assert_eq!(
        events_before_gcf.first(),
        Some(&(handle, FigureEventKind::Created))
    );
    assert_eq!(
        events_before_gcf.last(),
        Some(&(handle, FigureEventKind::Updated))
    );
    assert_eq!(gcf_builtin().unwrap(), first);
    assert_eq!(state::current_axes_state().active_index, axes.active_index);
    assert_eq!(state::current_figure_revision(handle), revision);
    assert_eq!(
        state::axes_handles_for_figure(handle).unwrap(),
        axes_handles
    );
    assert_eq!(*events.lock().unwrap(), events_before_gcf);

    let other = state::new_figure_handle();
    let before = events.lock().unwrap().len();
    assert_eq!(gcf_builtin().unwrap(), other.as_u32() as f64);
    assert_eq!(events.lock().unwrap().len(), before);
    state::close_figure(Some(other)).unwrap();
    state::close_figure(Some(handle)).unwrap();
    assert!(state::current_figure_handle_if_exists().is_none());
    gcf_builtin().unwrap();
    assert!(state::current_figure_handle_if_exists().is_some());
    assert_eq!(
        events.lock().unwrap().last().unwrap().1,
        FigureEventKind::Created
    );
}

#[test]
fn gcf_output_count_precedes_creation_and_discard_still_creates() {
    let _lock = lock_plot_registry();
    state::reset_plot_state();
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        assert_eq!(
            gcf_builtin().unwrap_err().identifier(),
            Some("RunMat:gcf:TooManyOutputs")
        );
        assert!(state::current_figure_handle_if_exists().is_none());
    }
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    gcf_builtin().unwrap();
    assert!(state::current_figure_handle_if_exists().is_some());
}
