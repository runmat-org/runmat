//! Live Figure/Axes object authority owned by the shared plot registry.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_RECEIVER_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, thiserror::Error)]
#[error("graphics receiver identity space exhausted")]
pub struct GraphicsReceiverIssueError;

fn issue_with(
    next_generation: &AtomicU64,
    kind: GraphicsReceiverKind,
) -> Result<GraphicsReceiverRef, GraphicsReceiverIssueError> {
    let generation = next_generation
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |next| {
            next.checked_add(1)
        })
        .map_err(|_| GraphicsReceiverIssueError)?;
    Ok(GraphicsReceiverRef::issued(kind, generation))
}

pub(super) fn issue_graphics_receiver(
    kind: GraphicsReceiverKind,
) -> Result<GraphicsReceiverRef, GraphicsReceiverIssueError> {
    issue_with(&NEXT_RECEIVER_GENERATION, kind)
}

/// Resolve the exact object lease; numeric coordinates alone confer no authority.
pub fn graphics_receiver_valid(receiver: &GraphicsReceiverRef) -> bool {
    let reg = registry();
    match receiver.kind() {
        GraphicsReceiverKind::Figure { figure } => reg
            .figures
            .get(&FigureHandle::from(figure))
            .is_some_and(|state| {
                state
                    .figure_receiver
                    .as_ref()
                    .is_some_and(|owned| owned.shares_lease(receiver))
            }),
        GraphicsReceiverKind::Axes { figure, index } => reg
            .figures
            .get(&FigureHandle::from(figure))
            .is_some_and(|state| {
                index < axes_count(state)
                    && state
                        .axes_receivers
                        .get(&index)
                        .is_some_and(|owned| owned.shares_lease(receiver))
            }),
    }
}

pub fn figure_graphics_receiver(
    handle: FigureHandle,
) -> Result<Option<GraphicsReceiverRef>, GraphicsReceiverIssueError> {
    registry()
        .figures
        .get(&handle)
        .map(|state| {
            state
                .figure_receiver
                .clone()
                .ok_or(GraphicsReceiverIssueError)
        })
        .transpose()
}

pub fn axes_graphics_receiver(
    handle: FigureHandle,
    index: usize,
) -> Result<Option<GraphicsReceiverRef>, GraphicsReceiverIssueError> {
    let mut reg = registry();
    let Some(state) = reg.figures.get_mut(&handle) else {
        return Ok(None);
    };
    if index >= axes_count(state) {
        return Ok(None);
    }
    if let Some(receiver) = state.axes_receivers.get(&index) {
        return Ok(Some(receiver.clone()));
    }
    let receiver = issue_graphics_receiver(GraphicsReceiverKind::Axes {
        figure: handle.as_u32(),
        index,
    })?;
    state.axes_receivers.insert(index, receiver.clone());
    Ok(Some(receiver))
}

/// Construct the canonical handle-object receiver for a selected live lease.
/// The returned Value owns only the lease; future operations revalidate it.
pub fn graphics_receiver_object(
    receiver: GraphicsReceiverRef,
) -> Result<Option<runmat_value::Value>, runmat_gc::GcError> {
    if !graphics_receiver_valid(&receiver) {
        return Ok(None);
    }
    let class = match receiver.kind() {
        GraphicsReceiverKind::Figure { .. } => runmat_types::standard::FIGURE,
        GraphicsReceiverKind::Axes { .. } => runmat_types::standard::AXES,
    };
    let object = runmat_value::ObjectInstance::new(class).with_graphics_receiver(receiver);
    let target = runmat_gc::gc_allocate(runmat_value::Value::Object(object))?;
    Ok(Some(runmat_value::Value::HandleObject(
        runmat_value::HandleRef {
            class_name: class.into(),
            target,
            valid: true,
        },
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};

    fn setup() -> super::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_plot_state();
        guard
    }

    #[test]
    fn generation_exhaustion_is_typed_and_never_wraps() {
        let issuer = AtomicU64::new(u64::MAX - 1);
        let kind = GraphicsReceiverKind::Figure { figure: 1 };
        let final_receiver = issue_with(&issuer, kind).expect("final identity");
        assert_eq!(final_receiver.generation(), u64::MAX - 1);
        assert!(issue_with(&issuer, kind).is_err());
        assert_eq!(issuer.load(Ordering::Acquire), u64::MAX);
    }

    #[test]
    fn figure_lease_retires_on_close_and_numeric_reuse() {
        let _guard = setup();
        let handle = FigureHandle::from(17);
        select_figure(handle);
        let original = figure_graphics_receiver(handle)
            .expect("issue receiver")
            .expect("selected figure");
        assert!(graphics_receiver_valid(&original));
        close_figure(Some(handle)).expect("close");
        assert!(!graphics_receiver_valid(&original));
        select_figure(handle);
        let replacement = figure_graphics_receiver(handle)
            .expect("issue replacement")
            .expect("replacement");
        assert!(!original.shares_lease(&replacement));
        assert!(graphics_receiver_valid(&replacement));
        reset_plot_state();
        assert!(!graphics_receiver_valid(&replacement));
    }

    #[test]
    fn forged_coordinates_do_not_authenticate_axes() {
        let _guard = setup();
        let handle = FigureHandle::from(18);
        select_figure(handle);
        let selected = axes_graphics_receiver(handle, 0)
            .expect("issue axes")
            .expect("default axes");
        assert!(graphics_receiver_valid(&selected));
        let forged = GraphicsReceiverRef::issued(selected.kind(), selected.generation());
        assert!(!graphics_receiver_valid(&forged));
        clear_axes_target(handle, 0, true).expect("reset live axes");
        assert!(graphics_receiver_valid(&selected));
        clear_figure(Some(handle)).expect("clear");
        assert!(!graphics_receiver_valid(&selected));
    }

    #[test]
    fn receiver_object_has_canonical_class_and_retires_with_figure() {
        let _guard = setup();
        let figure = FigureHandle::from(19);
        select_figure(figure);
        let receiver = figure_graphics_receiver(figure)
            .expect("issue receiver")
            .expect("figure receiver");
        let value = graphics_receiver_object(receiver)
            .expect("allocate receiver")
            .expect("live receiver");
        let runmat_value::Value::HandleObject(handle) = value else {
            panic!("graphics receiver must be a handle object")
        };
        assert_eq!(handle.class_name, runmat_types::standard::FIGURE);
        assert_eq!(
            runmat_gc::gc_with_value(&handle.target, |value| match value {
                runmat_value::Value::Object(object) => object.persistence_payload_kind(),
                _ => panic!("receiver target is not an object"),
            })
            .expect("live target"),
            runmat_value::ObjectPersistencePayloadKind::GraphicsReceiver
        );
        assert!(crate::is_handle_valid(&handle));
        close_figure(Some(figure)).expect("close");
        assert!(!crate::is_handle_valid(&handle));
    }

    #[test]
    fn live_object_property_routes_do_not_erase_lease_to_numeric_authority() {
        let _guard = setup();
        let figure = FigureHandle::from(20);
        select_figure(figure);
        let value = graphics_receiver_object(
            figure_graphics_receiver(figure)
                .expect("issue receiver")
                .expect("figure receiver"),
        )
        .expect("allocate")
        .expect("live");
        let denied_set = crate::builtins::plotting::set::set_builtin(vec![
            value.clone(),
            runmat_value::Value::String("Name".into()),
            runmat_value::Value::String("selected".into()),
        ])
        .expect_err("object setter must retain lease through registry mutation");
        assert_eq!(
            denied_set.identifier(),
            Some("RunMat:graphics:ReceiverPropertyUnavailable")
        );
        let denied_get = crate::builtins::plotting::get::get_builtin(vec![
            value.clone(),
            runmat_value::Value::String("Name".into()),
        ])
        .expect_err("object getter must retain lease through registry read");
        assert_eq!(
            denied_get.identifier(),
            Some("RunMat:graphics:ReceiverPropertyUnavailable")
        );
        let denied_dot = futures::executor::block_on(crate::object::resolve::load_member(
            value.clone(),
            "Name".into(),
            false,
            None,
        ))
        .expect_err("dot read cannot erase the lease");
        assert_eq!(
            denied_dot.identifier(),
            Some("RunMat:graphics:ReceiverPropertyUnavailable")
        );
        assert_ne!(
            crate::builtins::plotting::get::get_builtin(vec![
                runmat_value::Value::Num(f64::from(figure.as_u32())),
                runmat_value::Value::String("Name".into()),
            ])
            .expect("ordinary numeric route unchanged"),
            runmat_value::Value::String("selected".into())
        );
        let runmat_value::Value::HandleObject(handle) = value else {
            panic!("object carrier")
        };
        assert!(crate::is_handle_valid(&handle));
        assert!(super::super::figure_handle_exists(figure));
        close_figure(Some(figure)).expect("close");
        assert!(!crate::is_handle_valid(&handle));
    }
}
