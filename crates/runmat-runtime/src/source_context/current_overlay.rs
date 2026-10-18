//! Ownership of one current-source overlay within an invocation projection.

use std::cell::Cell;
use std::rc::Rc;

use super::{SourceCurrentProjection, SourceInfo};
use crate::context::poll_projection::PollProjection;

pub(super) struct CurrentSourceOverlay {
    current: PollProjection<SourceCurrentProjection>,
    live: Rc<Cell<bool>>,
}

impl CurrentSourceOverlay {
    pub(super) fn push(
        current: PollProjection<SourceCurrentProjection>,
        source: Option<SourceInfo>,
    ) -> Self {
        let live = current.borrow_mut().push(source);
        Self { current, live }
    }

    pub(super) fn retire(&self) {
        self.current.borrow_mut().retire(&self.live);
    }
}
