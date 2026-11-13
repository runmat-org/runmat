//! One exact axes selection after caller-owned control/output preflight.
use super::*;

pub(crate) struct ResolvedAxesTarget {
    authority: PropertyMutationTarget,
    handle: FigureHandle,
    index: usize,
}

impl ResolvedAxesTarget {
    pub(crate) fn authority(&self) -> &PropertyMutationTarget {
        &self.authority
    }
    pub(crate) fn coordinates(&self) -> (FigureHandle, usize) {
        (self.handle, self.index)
    }
    pub(super) fn into_authority(self) -> PropertyMutationTarget {
        self.authority
    }
}

pub(crate) fn resolve_axes_target(
    authority: PropertyMutationTarget,
    explicit: Option<(FigureHandle, usize)>,
) -> Result<ResolvedAxesTarget, FigureError> {
    let location = explicit.map_or(PropertyLocation::CurrentFigure, |(handle, index)| {
        PropertyLocation::Axes(handle, index)
    });
    resolve(authority, location)
}

pub(crate) fn resolve_current_axes_for_figure(
    authority: PropertyMutationTarget,
    handle: FigureHandle,
) -> Result<ResolvedAxesTarget, FigureError> {
    resolve(authority, PropertyLocation::Figure(handle))
}

fn resolve(
    authority: PropertyMutationTarget,
    location: PropertyLocation,
) -> Result<ResolvedAxesTarget, FigureError> {
    let (handle, index, selection) = {
        let mut reg = authority.registry(location)?;
        let (handle, explicit_index, implicit) = match location {
            PropertyLocation::CurrentFigure => (reg.current, None, true),
            PropertyLocation::Figure(handle) => (handle, None, !authority.is_selected()),
            PropertyLocation::Axes(handle, index) => (handle, Some(index), false),
            _ => return Err(FigureError::InvalidAxesHandle),
        };
        let state = if implicit {
            get_state_for_query(&mut reg, handle)
        } else {
            reg.figures
                .get_mut(&handle)
                .ok_or(FigureError::InvalidHandle(handle.as_u32()))?
        };
        let index = explicit_index.unwrap_or(state.active_axes);
        validate_axes_index(state, index)?;
        let receiver =
            graphics_receiver::axes_graphics_receiver_in_registry(&mut reg, handle, index)?
                .ok_or(FigureError::InvalidAxesHandle)?;
        let selection = GraphicsPropertySelection::capture_in_registry(receiver, &reg)?;
        (handle, index, selection)
    };
    Ok(ResolvedAxesTarget {
        authority: authority.selecting(selection),
        handle,
        index,
    })
}
