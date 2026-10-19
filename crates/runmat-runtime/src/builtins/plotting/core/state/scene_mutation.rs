//! One exact scene successor per Figure per registry transaction.

use super::{
    validate_axes_index, FigureError, FigureHandle, FigureState, GraphicsSceneLease, PlotRegistry,
};
use std::collections::hash_map::Entry;

impl FigureState {
    fn replace_scene_lease(&mut self, successor: GraphicsSceneLease) -> GraphicsSceneLease {
        debug_assert_eq!(self.scene_lease.figure(), successor.figure());
        std::mem::replace(&mut self.scene_lease, successor)
    }
}

/// Query access may create the ordinary default Figure but does not retire an
/// existing immutable scene selection.
pub(super) fn get_state_for_query(
    registry: &mut PlotRegistry,
    handle: FigureHandle,
) -> &mut FigureState {
    registry
        .figures
        .entry(handle)
        .or_insert_with(|| FigureState::new(handle))
}

/// Change the style cursor/order used by a *later* plot. These fields are not
/// part of the currently rendered Figure and cannot change an admitted scene
/// snapshot; the later plot insertion obtains a scene successor itself.
pub(super) fn get_state_for_future_plot_style(
    registry: &mut PlotRegistry,
    handle: FigureHandle,
) -> &mut FigureState {
    get_state_for_query(registry, handle)
}

/// Admit a mutable Figure before the caller can change scene or interaction
/// state. A strong predecessor is parked until the registry guard releases.
pub(super) fn get_state_mut(registry: &mut PlotRegistry, handle: FigureHandle) -> &mut FigureState {
    let PlotRegistry {
        figures,
        scene_transaction_lease,
        extra_scene_transaction_leases,
        ..
    } = registry;
    match figures.entry(handle) {
        Entry::Occupied(entry) => {
            // A compound mutation may request the same Figure more than once
            // while one registry guard is held. Its first successor already
            // fences every later write in that transaction.
            if scene_transaction_lease
                .as_ref()
                .is_some_and(|lease| lease.figure() == handle.as_u32())
                || extra_scene_transaction_leases
                    .iter()
                    .any(|lease| lease.figure() == handle.as_u32())
            {
                return entry.into_mut();
            }
            let successor = GraphicsSceneLease::candidate(handle.as_u32());
            if scene_transaction_lease.is_some() {
                extra_scene_transaction_leases.reserve(1);
            }
            let state = entry.into_mut();
            let retired = state.replace_scene_lease(successor);
            if let Some(previous) = scene_transaction_lease.replace(retired) {
                extra_scene_transaction_leases.push(previous);
            }
            state
        }
        Entry::Vacant(entry) => {
            if scene_transaction_lease.is_some() {
                extra_scene_transaction_leases.reserve(1);
            }
            let state = entry.insert(FigureState::new(handle));
            // A new Figure has no predecessor, but later accesses under this
            // guard belong to the same transaction and must not remint it.
            let selected = state.scene_lease.clone();
            if let Some(previous) = scene_transaction_lease.replace(selected) {
                extra_scene_transaction_leases.push(previous);
            }
            state
        }
    }
}

pub(super) fn existing_state_mut(
    registry: &mut PlotRegistry,
    handle: FigureHandle,
) -> Result<&mut FigureState, FigureError> {
    if !registry.figures.contains_key(&handle) {
        return Err(FigureError::InvalidHandle(handle.as_u32()));
    }
    Ok(get_state_mut(registry, handle))
}

pub(super) fn existing_axes_state_mut(
    registry: &mut PlotRegistry,
    handle: FigureHandle,
    axes_index: usize,
) -> Result<&mut FigureState, FigureError> {
    let state = registry
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    validate_axes_index(state, axes_index)?;
    Ok(get_state_mut(registry, handle))
}
