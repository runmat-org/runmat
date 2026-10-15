//! Borrowed property grammar; authority remains in the selected state owner.

use super::super::state::{FigureOwner, GraphicsPropertySelection, PropertyMutationTarget};
use super::*;

pub(crate) enum PropertySetOutcome {
    Complete,
    Presentation(PropertyPresentation),
}

pub(crate) enum PropertyPresentation {
    CompatibilityAlias(FigureHandle, FigureOwner),
    Selected {
        selection: GraphicsPropertySelection,
        figure: FigureOwner,
    },
}

impl PropertySetOutcome {
    pub(super) fn presentation(
        authority: &PropertyMutationTarget,
        handle: FigureHandle,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        let figure = authority
            .figure_snapshot(handle)
            .map_err(|error| map_figure_error(builtin, error))?;
        let source = match authority
            .selection()
            .map_err(|error| map_figure_error(builtin, error))?
        {
            Some(selection) => PropertyPresentation::Selected { selection, figure },
            None => PropertyPresentation::CompatibilityAlias(handle, figure),
        };
        Ok(Self::Presentation(source))
    }

    pub(super) fn complete_compatibility(self, builtin: &'static str) -> BuiltinResult<()> {
        match self {
            Self::Complete => Ok(()),
            Self::Presentation(PropertyPresentation::CompatibilityAlias(handle, figure)) => {
                present_figure_update(builtin, handle, figure.as_ref().clone()).map(|_| ())
            }
            Self::Presentation(PropertyPresentation::Selected { .. }) => {
                Err(receiver::property_unavailable(builtin))
            }
        }
    }
}

pub(super) enum PropertySetHandle<'a> {
    Root,
    Figure(FigureHandle),
    Axes(FigureHandle, usize),
    Ruler(FigureHandle, usize, PlotObjectKind),
    Text(FigureHandle, usize, PlotObjectKind),
    Legend(FigureHandle, usize),
    PlotChild(f64, &'a super::super::state::PlotChildHandleState),
}

impl<'a> From<&'a PlotHandle> for PropertySetHandle<'a> {
    fn from(handle: &'a PlotHandle) -> Self {
        match handle {
            PlotHandle::Root => Self::Root,
            PlotHandle::Figure(figure) => Self::Figure(*figure),
            PlotHandle::Axes(figure, index) => Self::Axes(*figure, *index),
            PlotHandle::Ruler(figure, index, kind) => Self::Ruler(*figure, *index, *kind),
            PlotHandle::Text(figure, index, kind) => Self::Text(*figure, *index, *kind),
            PlotHandle::Legend(figure, index) => Self::Legend(*figure, *index),
            PlotHandle::PlotChild(id, state) => Self::PlotChild(*id, state),
        }
    }
}
