//! Immutable figure snapshots with copy-on-write registry mutation.

use runmat_plot::plots::Figure;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

#[derive(Clone, Default)]
pub(super) struct SharedFigure(Arc<Figure>);

impl SharedFigure {
    pub(super) fn snapshot(&self) -> Arc<Figure> {
        self.0.clone()
    }

    pub(super) fn owned_clone(&self) -> Figure {
        self.0.as_ref().clone()
    }
}

impl From<Figure> for SharedFigure {
    fn from(figure: Figure) -> Self {
        Self(Arc::new(figure))
    }
}

impl Deref for SharedFigure {
    type Target = Figure;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl DerefMut for SharedFigure {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.0)
    }
}
