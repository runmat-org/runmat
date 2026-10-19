//! Immutable figure snapshots with copy-on-write registry mutation.

use runmat_plot::plots::Figure;
use std::ops::{Deref, DerefMut};
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
pub type FigureOwner = Arc<Figure>;
#[cfg(target_arch = "wasm32")]
pub type FigureOwner = Rc<Figure>;

#[derive(Clone, Default)]
pub(super) struct SharedFigure(FigureOwner);

impl SharedFigure {
    pub(super) fn snapshot(&self) -> FigureOwner {
        self.0.clone()
    }

    pub(super) fn owned_clone(&self) -> Figure {
        self.0.as_ref().clone()
    }
}

impl From<Figure> for SharedFigure {
    fn from(figure: Figure) -> Self {
        Self(FigureOwner::new(figure))
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
        FigureOwner::make_mut(&mut self.0)
    }
}
