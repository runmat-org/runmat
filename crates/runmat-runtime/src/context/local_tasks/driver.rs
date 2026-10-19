//! Poll-scoped proof that a foreground call belongs to this local executor.

use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub(super) struct SupervisorId(Rc<()>);

impl SupervisorId {
    pub(super) fn new() -> Self {
        Self(Rc::new(()))
    }

    pub(super) fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

thread_local! {
    static ACTIVE_DRIVER: RefCell<Option<SupervisorId>> = const { RefCell::new(None) };
}

pub(super) fn active_driver() -> Option<SupervisorId> {
    ACTIVE_DRIVER.with(|active| active.borrow().clone())
}

pub(super) struct ActiveDriverPoll(Option<SupervisorId>);

impl ActiveDriverPoll {
    pub(super) fn enter(owner: &SupervisorId) -> Self {
        let previous = ACTIVE_DRIVER.with(|active| active.replace(Some(owner.clone())));
        Self(previous)
    }
}

impl Drop for ActiveDriverPoll {
    fn drop(&mut self) {
        ACTIVE_DRIVER.with(|active| {
            active.replace(self.0.take());
        });
    }
}
