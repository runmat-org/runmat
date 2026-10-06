//! Ownership of one current-source overlay within an invocation projection.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use super::{SourceCurrentProjection, SourceInfo};

pub(super) struct CurrentSourceOverlay {
    current: Rc<RefCell<SourceCurrentProjection>>,
    live: Rc<Cell<bool>>,
}

impl CurrentSourceOverlay {
    pub(super) fn push(
        current: Rc<RefCell<SourceCurrentProjection>>,
        source: Option<SourceInfo>,
    ) -> Self {
        let live = current.borrow_mut().push(source);
        Self { current, live }
    }

    pub(super) fn retire(&self) {
        self.current.borrow_mut().retire(&self.live);
    }
}
