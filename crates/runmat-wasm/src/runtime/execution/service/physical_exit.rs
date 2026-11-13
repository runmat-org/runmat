//! Existing retained-work lease table. Owned builtins require literal physical
//! exit; selected programs require typed Runtime invocation retirement.
//! Callers enforce their distinct proof gates. Rejection proves neither.
use super::BrowserExecutionService;
use runmat_execution::TaskId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local! {
    // A failed launch/cancel Promise cannot release an owned numerical claim.
    // Keep its service, request, and reservation alive independently of caller
    // references until its route's actual typed retirement proof is accepted.
    static LIVE_PHYSICAL_WORK: RefCell<HashMap<TaskId, Rc<BrowserExecutionService>>> =
        RefCell::new(HashMap::new());
}

pub(super) fn retain_until_physical_exit(task_id: TaskId, service: Rc<BrowserExecutionService>) {
    let previous = LIVE_PHYSICAL_WORK.with(|live| live.borrow_mut().insert(task_id, service));
    drop(previous);
}

pub(super) fn physical_exit_certified(task_id: TaskId) {
    let retired = LIVE_PHYSICAL_WORK.with(|live| live.borrow_mut().remove(&task_id));
    drop(retired);
}
