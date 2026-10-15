//! Authenticate role geometry against one exact live immutable scene selection.
use super::*;
use runmat_value::{GraphicsCallbackKind, GraphicsCallbackRolePayload};

pub(crate) fn callback_payload_valid(
    registered: &GraphicsReceiverRef,
    kind: GraphicsCallbackKind,
    payload: &GraphicsCallbackRolePayload,
) -> bool {
    callback_payload_valid_at_scene(registered, kind, payload, payload.figure_and_scene().1)
}

pub(crate) fn callback_payload_valid_at_scene(
    registered: &GraphicsReceiverRef,
    kind: GraphicsCallbackKind,
    payload: &GraphicsCallbackRolePayload,
    admitted_scene: &GraphicsSceneLease,
) -> bool {
    if !payload.matches_role(kind) {
        return false;
    }
    let (figure, _) = payload.figure_and_scene();
    if !figure.shares_lease(registered) {
        return false;
    }
    let GraphicsReceiverKind::Figure { figure: number } = figure.kind() else {
        return false;
    };
    let reg = registry();
    let Some(state) = reg.figures.get(&FigureHandle::from(number)) else {
        return false;
    };
    if !state
        .figure_receiver
        .as_ref()
        .is_some_and(|owned| owned.shares_lease(figure))
        || !state.scene_lease.shares_lease(admitted_scene)
    {
        return false;
    }
    let axes_index = |axes: &GraphicsReceiverRef| match axes.kind() {
        GraphicsReceiverKind::Axes { figure, index }
            if figure == number
                && index < axes_count(state)
                && state
                    .axes_receivers
                    .get(&index)
                    .is_some_and(|owned| owned.shares_lease(axes)) =>
        {
            Some(index)
        }
        _ => None,
    };
    match payload {
        GraphicsCallbackRolePayload::Action { axes, .. } => axes_index(axes).is_some(),
        GraphicsCallbackRolePayload::Filter { axes, target, .. } => {
            let Some(index) = axes_index(axes) else {
                return false;
            };
            match target.kind() {
                GraphicsReceiverKind::Figure { .. } => target.shares_lease(figure),
                GraphicsReceiverKind::Axes { .. } => target.shares_lease(axes),
                GraphicsReceiverKind::PlotChild { id } => {
                    reg.plot_children.entries.get(&id).is_some_and(|entry| {
                        entry
                            .receiver
                            .as_ref()
                            .is_some_and(|owned| owned.shares_lease(target))
                            && entry.state.figure_axes() == (FigureHandle::from(number), index)
                    })
                }
            }
        }
    }
}
