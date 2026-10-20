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
        GraphicsReceiverKind::PlotChild { id } => reg
            .plot_children
            .entries
            .get(&id)
            .and_then(|entry| entry.receiver.as_ref())
            .is_some_and(|owned| owned.shares_lease(receiver)),
    }
}

/// Issue or borrow the exact identity already owned by one admitted child.
/// The numeric child key is never accepted as proof of object authority.
pub fn plot_child_graphics_receiver(
    id: u64,
) -> Result<Option<GraphicsReceiverRef>, GraphicsReceiverIssueError> {
    let mut reg = registry();
    let Some(entry) = reg.plot_children.entries.get_mut(&id) else {
        return Ok(None);
    };
    if let Some(receiver) = &entry.receiver {
        return Ok(Some(receiver.clone()));
    }
    let receiver = issue_graphics_receiver(GraphicsReceiverKind::PlotChild { id })?;
    entry.receiver = Some(receiver.clone());
    Ok(Some(receiver))
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
    axes_graphics_receiver_in_registry(&mut reg, handle, index)
}

pub(super) fn axes_graphics_receiver_in_registry(
    reg: &mut PlotRegistry,
    handle: FigureHandle,
    index: usize,
) -> Result<Option<GraphicsReceiverRef>, GraphicsReceiverIssueError> {
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

/// Resolve an axes object hit-tested against an immutable presented Figure.
/// The pinned scene lease and Figure receiver must both still be selected by
/// the same registry entry; a numeric revision or axes index alone cannot
/// retarget a callback after mutation, close, or handle reuse.
pub fn axes_graphics_receiver_for_scene(
    figure_receiver: &GraphicsReceiverRef,
    scene_lease: &GraphicsSceneLease,
    index: usize,
) -> Result<Option<GraphicsReceiverRef>, GraphicsReceiverIssueError> {
    let GraphicsReceiverKind::Figure { figure } = figure_receiver.kind() else {
        return Ok(None);
    };
    if scene_lease.figure() != figure {
        return Ok(None);
    }
    let mut reg = registry();
    let Some(state) = reg.figures.get_mut(&FigureHandle::from(figure)) else {
        return Ok(None);
    };
    if !state.scene_lease.shares_lease(scene_lease)
        || !state
            .figure_receiver
            .as_ref()
            .is_some_and(|owned| owned.shares_lease(figure_receiver))
        || index >= axes_count(state)
    {
        return Ok(None);
    }
    if let Some(receiver) = state.axes_receivers.get(&index) {
        return Ok(Some(receiver.clone()));
    }
    let receiver = issue_graphics_receiver(GraphicsReceiverKind::Axes { figure, index })?;
    state.axes_receivers.insert(index, receiver.clone());
    Ok(Some(receiver))
}

/// Authenticate a pinned scene and receiver under the same registry guard.
#[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
pub(crate) fn figure_scene_is_current(
    receiver: &GraphicsReceiverRef,
    scene: &GraphicsSceneLease,
) -> bool {
    let GraphicsReceiverKind::Figure { figure } = receiver.kind() else {
        return false;
    };
    if scene.figure() != figure {
        return false;
    }
    registry()
        .figures
        .get(&FigureHandle::from(figure))
        .is_some_and(|state| {
            state.scene_lease.shares_lease(scene)
                && state
                    .figure_receiver
                    .as_ref()
                    .is_some_and(|owned| owned.shares_lease(receiver))
        })
}

/// Construct the canonical handle-object receiver for a selected live lease.
/// The returned Value owns only the lease; future operations revalidate it.
pub fn graphics_receiver_object(
    receiver: GraphicsReceiverRef,
) -> Result<Option<runmat_value::Value>, runmat_gc::GcError> {
    let Some(class) = graphics_receiver_class(&receiver) else {
        return Ok(None);
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

/// Derive class only from the exact live typed registry entry, never Type text.
pub(crate) fn graphics_receiver_class(
    receiver: &GraphicsReceiverRef,
) -> Option<runmat_types::StaticClassIdentity> {
    let registry = registry();
    match receiver.kind() {
        GraphicsReceiverKind::Figure { figure } => registry
            .figures
            .get(&FigureHandle::from(figure))
            .filter(|state| {
                state
                    .figure_receiver
                    .as_ref()
                    .is_some_and(|owned| owned.shares_lease(receiver))
            })
            .map(|_| runmat_types::standard::FIGURE),
        GraphicsReceiverKind::Axes { figure, index } => registry
            .figures
            .get(&FigureHandle::from(figure))
            .filter(|state| {
                index < axes_count(state)
                    && state
                        .axes_receivers
                        .get(&index)
                        .is_some_and(|owned| owned.shares_lease(receiver))
            })
            .map(|_| runmat_types::standard::AXES),
        GraphicsReceiverKind::PlotChild { id } => registry
            .plot_children
            .entries
            .get(&id)
            .filter(|entry| {
                entry
                    .receiver
                    .as_ref()
                    .is_some_and(|owned| owned.shares_lease(receiver))
            })
            .map(|entry| entry.state.class_identity()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use runmat_value::Value;

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
        let selection = GraphicsPropertySelection::capture(original.clone()).unwrap();
        selection.ensure_current().unwrap();
        assert!(graphics_receiver_valid(&original));
        close_figure(Some(handle)).expect("close");
        assert!(!graphics_receiver_valid(&original));
        assert!(selection.ensure_current().is_err());
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
    fn child_lease_is_exact_and_retires_on_clear_and_key_reuse() {
        let _guard = setup();
        let figure = FigureHandle::from(19);
        select_figure(figure);
        let id = register_line_handle(figure, 0, 0) as u64;
        let original = plot_child_graphics_receiver(id)
            .expect("issue child")
            .expect("registered child");
        let repeated = plot_child_graphics_receiver(id)
            .expect("borrow child")
            .expect("registered child");
        let selection = GraphicsPropertySelection::capture(original.clone()).unwrap();
        assert_eq!(
            graphics_receiver_class(&original),
            Some(runmat_types::standard::GRAPHICS_LINE)
        );
        selection.ensure_current().unwrap();
        assert!(original.shares_lease(&repeated));
        assert!(graphics_receiver_valid(&original));
        assert!(!graphics_receiver_valid(&GraphicsReceiverRef::issued(
            original.kind(),
            original.generation(),
        )));

        let other_id = register_line_handle(figure, 0, 1) as u64;
        let (original_state, other_state) = {
            let reg = registry();
            (
                reg.plot_children.entries[&id].state.clone(),
                reg.plot_children.entries[&other_id].state.clone(),
            )
        };
        {
            let mut reg = registry();
            let other = reg
                .plot_children
                .find_mut(|state| state.plot_index() == Some(1))
                .unwrap();
            let PlotChildHandleState::Line(other) = other else {
                panic!("line state");
            };
            other.plot_index = 2;
            assert!(
                Arc::ptr_eq(&original_state, &reg.plot_children.entries[&id].state),
                "borrowed predicates must not copy unrelated selected state"
            );
            assert!(!Arc::ptr_eq(
                &other_state,
                &reg.plot_children.entries[&other_id].state
            ));
        }
        assert_eq!(
            other_state.plot_index(),
            Some(1),
            "the retained snapshot is immutable"
        );
        selection.ensure_current().unwrap();
        {
            let mut reg = registry();
            let original = reg.plot_children.get_mut(&id).unwrap();
            let PlotChildHandleState::Line(original) = original else {
                panic!("line state");
            };
            original.plot_index = 3;
        }
        assert!(
            graphics_receiver_valid(&original),
            "metadata replacement does not forge a new receiver"
        );
        assert!(
            selection.ensure_current().is_err(),
            "backing identity fences metadata changes even without a scene revision"
        );

        clear_figure(Some(figure)).expect("clear figure");
        assert!(!graphics_receiver_valid(&original));
        assert!(selection.ensure_current().is_err());
        assert!(plot_child_graphics_receiver(id).expect("lookup").is_none());

        let mut reg = registry();
        reg.plot_children.insert(
            id,
            PlotChildHandleState::Line(SimplePlotHandleState {
                figure,
                axes_index: 0,
                plot_index: 0,
            }),
        );
        drop(reg);
        let replacement = plot_child_graphics_receiver(id)
            .expect("issue replacement")
            .expect("replacement child");
        assert!(!original.shares_lease(&replacement));
        assert!(!graphics_receiver_valid(&original));
        assert!(graphics_receiver_valid(&replacement));
    }

    #[test]
    fn copied_function_children_keep_callable_and_distinct_receiver_until_last_plot_retires() {
        let _guard = setup();
        let original_roots = callbacks::direct_root_count();

        for contour in [false, true] {
            let mut source_scene = Figure::new();
            let plot_index = if contour {
                source_scene.add_contour_plot_on_axes(
                    runmat_plot::plots::ContourPlot::from_vertices(
                        Vec::new(),
                        0.0,
                        runmat_plot::core::BoundingBox::default(),
                    ),
                    0,
                )
            } else {
                source_scene.add_surface_plot_on_axes(
                    runmat_plot::plots::SurfacePlot::new(vec![0.0], vec![0.0], vec![vec![0.0]])
                        .expect("minimal surface"),
                    0,
                )
            };
            let source_figure = import_figure(source_scene);
            let target_figure = import_figure(Figure::new());
            let token = callbacks::retain(Value::Num(31.0), "copyobj").expect("root callable");
            let source_handle = if contour {
                register_function_contour_handle(
                    source_figure,
                    0,
                    plot_index,
                    2,
                    (0.0, 1.0),
                    (0.0, 1.0),
                    FunctionSurfaceFunctionRef::OwnedCallable(token),
                    false,
                )
            } else {
                register_function_surface_handle(
                    source_figure,
                    0,
                    plot_index,
                    2,
                    (0.0, 1.0),
                    (0.0, 1.0),
                    FunctionSurfaceFunctionState::Explicit(
                        FunctionSurfaceFunctionRef::OwnedCallable(token),
                    ),
                )
            };
            let source_receiver = plot_child_graphics_receiver(source_handle as u64)
                .expect("source receiver")
                .expect("registered source");
            let copied_handle =
                copy_plot_child_to_parent(source_handle, CopyParentTarget::Axes(target_figure, 0))
                    .expect("copy plot child");
            let copied_receiver = plot_child_graphics_receiver(copied_handle as u64)
                .expect("copy receiver")
                .expect("registered copy");
            assert!(!source_receiver.shares_lease(&copied_receiver));

            close_figure(Some(source_figure)).expect("retire source");
            assert!(!graphics_receiver_valid(&source_receiver));
            assert!(graphics_receiver_valid(&copied_receiver));
            let copied_state = plot_child_handle_snapshot(copied_handle).expect("copied child");
            let copied_token = match &copied_state {
                PlotChildHandleState::FunctionSurface(state) => {
                    let FunctionSurfaceFunctionState::Explicit(
                        FunctionSurfaceFunctionRef::OwnedCallable(token),
                    ) = &state.function
                    else {
                        panic!("copied surface callable")
                    };
                    token
                }
                PlotChildHandleState::FunctionContour(state) => {
                    let FunctionSurfaceFunctionRef::OwnedCallable(token) = &state.function else {
                        panic!("copied contour callable")
                    };
                    token
                }
                _ => panic!("copied function child"),
            };
            assert_eq!(
                callbacks::resolve(copied_token, "copyobj").unwrap(),
                Value::Num(31.0)
            );
            drop(copied_state);
            close_figure(Some(target_figure)).expect("retire copied child");
            assert!(!graphics_receiver_valid(&copied_receiver));
            assert_eq!(callbacks::direct_root_count(), original_roots);
        }
    }

    #[test]
    fn pinned_scene_hit_test_rejects_mutation_and_numeric_handle_reuse() {
        let _guard = setup();
        let handle = FigureHandle::from(27);
        select_figure(handle);
        let receiver = figure_graphics_receiver(handle)
            .expect("issue receiver")
            .expect("selected figure");
        let snapshot = capture_figure_persistence_snapshot(&[handle], 1, usize::MAX)
            .expect("pin selected scene");
        let scene = &snapshot.entries[0].scene_lease;
        assert!(axes_graphics_receiver_for_scene(&receiver, scene, 0)
            .expect("issue axes")
            .is_some());

        set_figure_name(handle, "changed".into()).expect("mutate figure");
        assert!(axes_graphics_receiver_for_scene(&receiver, scene, 0)
            .expect("stale scene checked")
            .is_none());
        let current =
            capture_figure_persistence_snapshot(&[handle], 1, usize::MAX).expect("pin successor");
        assert!(
            axes_graphics_receiver_for_scene(&receiver, &current.entries[0].scene_lease, 0)
                .expect("current scene checked")
                .is_some()
        );

        close_figure(Some(handle)).expect("close figure");
        select_figure(handle);
        assert!(
            axes_graphics_receiver_for_scene(&receiver, &current.entries[0].scene_lease, 0)
                .expect("reused numeric handle checked")
                .is_none()
        );
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
        let selected_get = crate::builtins::plotting::get::get_builtin(vec![
            value.clone(),
            runmat_value::Value::String("Name".into()),
        ])
        .expect("object getter retains lease through registry read");
        assert_eq!(
            selected_get,
            Value::String(
                clone_figure(figure)
                    .unwrap()
                    .name
                    .unwrap_or_default()
                    .into()
            )
        );
        let selected_dot = futures::executor::block_on(crate::object::resolve::load_member(
            value.clone(),
            "Name".into(),
            false,
            None,
        ))
        .expect("dot read uses the exact property owner");
        assert_eq!(selected_dot, selected_get);
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
