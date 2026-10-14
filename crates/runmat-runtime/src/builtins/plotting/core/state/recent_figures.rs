//! Figure-change receipts belong to a selected command, not its session lane.

use super::FigureHandle;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone, Debug, Default)]
pub(crate) struct FigureReceiptOwner(Rc<RefCell<HashSet<FigureHandle>>>);

impl FigureReceiptOwner {
    pub(crate) fn record(&self, handle: FigureHandle) {
        self.0.borrow_mut().insert(handle);
    }

    pub(crate) fn reset(&self) {
        self.0.borrow_mut().clear();
    }

    pub(crate) fn take(&self) -> Vec<FigureHandle> {
        self.0.borrow_mut().drain().collect()
    }
}

#[cfg(test)]
#[path = "recent_figures/tests.rs"]
mod tests;
