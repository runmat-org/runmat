//! Ordinary implicit limits retain their exact source through linked updates.

use super::super::*;

fn set_current_axis_manual_limits(
    property: &PropertyMutationTarget,
    axis: LinkAxesAxis,
    limits: Option<(f64, f64)>,
) -> Result<(), FigureError> {
    let updates = {
        let mut reg = property.registry(PropertyLocation::CurrentFigure)?;
        let handle = reg.current;
        let (axes, x, y) = reg
            .figures
            .get(&handle)
            .map(|state| {
                let metadata = state.figure.axes_metadata(state.active_axes);
                (
                    state.active_axes,
                    metadata.and_then(|meta| meta.x_limits),
                    metadata.and_then(|meta| meta.y_limits),
                )
            })
            .unwrap_or((0, None, None));
        let (x, y) = match axis {
            LinkAxesAxis::X => (limits, y),
            LinkAxesAxis::Y => (x, limits),
        };
        get_state_mut(&mut reg, handle);
        let updates = set_axes_limits_with_links(&mut reg, handle, axes, x, y)?;
        super::PreparedFigureUpdates::capture(property, &reg, updates)?
    };
    updates.publish(property)?;
    Ok(())
}

pub(crate) fn set_current_x_limits_for_property(
    property: &PropertyMutationTarget,
    limits: Option<(f64, f64)>,
) -> Result<(), FigureError> {
    set_current_axis_manual_limits(property, LinkAxesAxis::X, limits)
}

pub(crate) fn set_current_y_limits_for_property(
    property: &PropertyMutationTarget,
    limits: Option<(f64, f64)>,
) -> Result<(), FigureError> {
    set_current_axis_manual_limits(property, LinkAxesAxis::Y, limits)
}
