//! Physical source selection for the existing property renderers.

use super::super::state::GraphicsPropertySnapshot;
use super::*;
use runmat_plot::plots::Figure;

#[derive(Clone, Copy)]
pub(super) enum PropertyReadSource<'a> {
    CompatibilityAlias,
    Selected(&'a GraphicsPropertySnapshot),
}

impl<'a> PropertyReadSource<'a> {
    pub(super) fn figure(self, handle: FigureHandle) -> Option<Cow<'a, Figure>> {
        match self {
            Self::CompatibilityAlias => super::super::state::clone_figure(handle).map(Cow::Owned),
            Self::Selected(snapshot) => (snapshot.selection.figure_handle() == handle)
                .then(|| Cow::Borrowed(snapshot.figure.as_ref())),
        }
    }

    pub(super) fn plot(
        self,
        handle: FigureHandle,
        index: usize,
    ) -> Option<Cow<'a, runmat_plot::plots::figure::PlotElement>> {
        match self {
            Self::CompatibilityAlias => super::super::state::clone_figure(handle)?
                .plots()
                .nth(index)
                .cloned()
                .map(Cow::Owned),
            Self::Selected(snapshot) => (snapshot.selection.figure_handle() == handle)
                .then(|| snapshot.figure.plots().nth(index))
                .flatten()
                .map(Cow::Borrowed),
        }
    }
}
