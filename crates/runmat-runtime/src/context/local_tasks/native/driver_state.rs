use super::super::driver::SupervisorId;
use super::*;
use tokio::task::LocalSet;

/// One host poll authority shared by distinct session task owners. Child
/// sessions never take ownership of this LocalSet or close it on retirement.
pub(super) struct DriverState {
    pub(super) id: SupervisorId,
    pub(super) local: RefCell<Option<Rc<LocalSet>>>,
    pub(super) closed: Cell<bool>,
    pub(super) driving: Cell<bool>,
    pub(super) children: RefCell<Vec<Weak<SupervisorInner>>>,
}

impl DriverState {
    pub(super) fn new() -> Self {
        Self {
            id: SupervisorId::new(),
            local: RefCell::new(Some(Rc::new(LocalSet::new()))),
            closed: Cell::new(false),
            driving: Cell::new(false),
            children: RefCell::new(Vec::new()),
        }
    }
}
