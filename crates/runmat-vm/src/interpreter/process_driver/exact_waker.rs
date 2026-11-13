//! Raw-waker adapter for the known-layout exact shared task wake owner.

use super::{wake_managed, TaskWake};
use runmat_execution::resource::ExactShared;
use std::mem::ManuallyDrop;
use std::task::{RawWaker, RawWakerVTable, Waker};

pub(super) fn waker(owner: ExactShared<TaskWake>) -> Waker {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TaskWake>();
    let raw = RawWaker::new(owner.into_raw(), &VTABLE);
    // SAFETY: VTABLE accounts for exactly one transferred ExactShared strong
    // count in every clone, wake and drop operation.
    unsafe { Waker::from_raw(raw) }
}

unsafe fn clone_raw(pointer: *const ()) -> RawWaker {
    // SAFETY: RawWaker data was produced by ExactShared<TaskWake>::into_raw.
    let borrowed = ManuallyDrop::new(unsafe { ExactShared::<TaskWake>::from_raw(pointer) });
    let cloned = ExactShared::clone(&borrowed);
    RawWaker::new(cloned.into_raw(), &VTABLE)
}

unsafe fn wake_raw(pointer: *const ()) {
    // SAFETY: wake consumes the RawWaker's one transferred strong count.
    let owner = unsafe { ExactShared::<TaskWake>::from_raw(pointer) };
    wake_managed(&owner);
}

unsafe fn wake_by_ref_raw(pointer: *const ()) {
    // SAFETY: this callback borrows but does not consume the transferred
    // count; ManuallyDrop prevents decrementing it.
    let owner = ManuallyDrop::new(unsafe { ExactShared::<TaskWake>::from_raw(pointer) });
    wake_managed(&owner);
}

unsafe fn drop_raw(pointer: *const ()) {
    // SAFETY: drop consumes the RawWaker's one transferred strong count.
    drop(unsafe { ExactShared::<TaskWake>::from_raw(pointer) });
}

static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_raw, wake_raw, wake_by_ref_raw, drop_raw);
