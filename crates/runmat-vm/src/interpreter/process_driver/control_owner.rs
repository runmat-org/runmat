//! Revocable exact owners for Managed continuation control state.

use super::*;
use allocator_api2::boxed::Box as FallibleBox;
use runmat_runtime::context::host_allocation::AdmittedCollectionAllocator;
use runmat_value::{ExactHostAllocationPermit, ExactHostNumericAllocationError};
use std::alloc::Layout;

pub(super) enum CompletionOwner<T> {
    Legacy(Rc<ReplyCell<T>>),
    Managed(ExactShared<ReplyCell<T>>),
}

impl<T> Clone for CompletionOwner<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Legacy(owner) => Self::Legacy(Rc::clone(owner)),
            Self::Managed(owner) => Self::Managed(owner.clone()),
        }
    }
}

impl<T> CompletionOwner<T> {
    pub(super) fn get(&self) -> &ReplyCell<T> {
        match self {
            Self::Legacy(owner) => owner,
            Self::Managed(owner) => owner.get(),
        }
    }
}

pub(super) struct ManagedReplyControl<T> {
    pub(super) completion: ExactShared<ReplyCell<T>>,
}

impl<T> ReplyControl for ManagedReplyControl<T> {
    fn is_canceled(&self) -> bool {
        self.completion.get().is_canceled()
    }

    fn publish(&self) {
        self.completion.get().publish();
    }
}

pub(super) enum ReplyControlOwner {
    Legacy(Rc<dyn ReplyControl>),
    Managed(ManagedReplyControlOwner),
}

pub(super) struct ManagedReplyControlOwner {
    pub(super) control: Box<dyn ReplyControl>,
    pub(super) _claim: ExactHostAllocationClaim,
}

impl ReplyControl for ReplyControlOwner {
    fn is_canceled(&self) -> bool {
        match self {
            Self::Legacy(owner) => owner.is_canceled(),
            Self::Managed(owner) => owner.control.is_canceled(),
        }
    }

    fn publish(&self) {
        match self {
            Self::Legacy(owner) => owner.publish(),
            Self::Managed(owner) => owner.control.publish(),
        }
    }
}

pub(super) struct ManagedDriverPort {
    pub(super) state: RefCell<Option<ExactShared<RefCell<DriverState>>>>,
    pub(super) _claim: ExactHostAllocationClaim,
}

impl ManagedDriverPort {
    pub(super) fn owner(&self) -> Option<ExactShared<RefCell<DriverState>>> {
        self.state.borrow().as_ref().cloned()
    }

    pub(super) fn revoke(&self, expected: &ExactShared<RefCell<DriverState>>) {
        let retired = {
            let mut state = self.state.borrow_mut();
            if state.as_ref().is_some_and(|owner| owner.ptr_eq(expected)) {
                state.take()
            } else {
                None
            }
        };
        drop(retired);
    }
}

#[derive(Clone)]
pub(super) enum DriverStateOwner {
    Legacy(Rc<RefCell<DriverState>>),
    Managed(ExactShared<RefCell<DriverState>>),
}

impl DriverStateOwner {
    pub(super) fn borrow(&self) -> std::cell::Ref<'_, DriverState> {
        match self {
            Self::Legacy(owner) => owner.borrow(),
            Self::Managed(owner) => owner.get().borrow(),
        }
    }

    pub(super) fn borrow_mut(&self) -> std::cell::RefMut<'_, DriverState> {
        match self {
            Self::Legacy(owner) => owner.borrow_mut(),
            Self::Managed(owner) => owner.get().borrow_mut(),
        }
    }
}

pub(super) enum DriverOwner {
    Legacy(Rc<RefCell<DriverState>>),
    Managed {
        state: ExactShared<RefCell<DriverState>>,
        port: ExactShared<ManagedDriverPort>,
    },
}

impl DriverOwner {
    pub(super) fn state(&self) -> DriverStateOwner {
        match self {
            Self::Legacy(owner) => DriverStateOwner::Legacy(Rc::clone(owner)),
            Self::Managed { state, .. } => DriverStateOwner::Managed(state.clone()),
        }
    }

    pub(super) fn handle(&self) -> DriverHandleOwner {
        match self {
            Self::Legacy(owner) => DriverHandleOwner::Legacy(Rc::downgrade(owner)),
            Self::Managed { port, .. } => DriverHandleOwner::Managed(port.clone()),
        }
    }

    pub(super) fn revoke(&self) {
        if let Self::Managed { state, port } = self {
            port.get().revoke(state);
        }
    }
}

pub(super) fn managed_request_control_bytes<T>() -> Result<u64, RuntimeError> {
    [
        ExactShared::<ReplyCell<T>>::requested_bytes(),
        ExactShared::<TaskWake>::requested_bytes(),
        Layout::new::<ManagedReplyControl<T>>().size(),
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        let bytes = u64::try_from(bytes).map_err(|_| driver_overflow())?;
        total.checked_add(bytes).ok_or_else(driver_overflow)
    })
}

pub(super) fn try_box<T>(value: T) -> Result<Box<T>, ExactHostNumericAllocationError> {
    let owner = FallibleBox::try_new(value)
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
    let pointer = FallibleBox::into_raw(owner);
    // SAFETY: allocator-api2's Global allocation uses the global allocator
    // and ownership of this exact pointer transfers once into std::Box.
    Ok(unsafe { Box::from_raw(pointer) })
}

pub(super) fn prepare_managed_request<T, F>(
    permit: &dyn ExactHostAllocationPermit,
    wake_hub: WakeHubOwner,
    id: CallId,
    depth: u32,
    future: F,
) -> Result<
    (
        CompletionOwner<T>,
        ReplyControlOwner,
        WakeOwner,
        QueuedFuture,
    ),
    ExactHostNumericAllocationError,
>
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
{
    let completion_claim =
        permit.reserve_exact_claim(ExactShared::<ReplyCell<T>>::requested_bytes())?;
    let completion = ExactShared::try_new(ReplyCell {
        result: RefCell::new(None),
        waker: RefCell::new(None),
        canceled: Cell::new(false),
        published: Cell::new(false),
        _claim: Some(completion_claim),
    })
    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
    let control_claim =
        permit.reserve_exact_claim(Layout::new::<ManagedReplyControl<T>>().size())?;
    let control = try_box(ManagedReplyControl {
        completion: completion.clone(),
    })?;
    let control: Box<dyn ReplyControl> = control;
    let control = ReplyControlOwner::Managed(ManagedReplyControlOwner {
        control,
        _claim: control_claim,
    });
    let wake_claim = permit.reserve_exact_claim(ExactShared::<TaskWake>::requested_bytes())?;
    let hub = match wake_hub {
        WakeHubOwner::Managed(hub) => hub,
        WakeHubOwner::Legacy(_) => return Err(ExactHostNumericAllocationError::AllocationFailed),
    };
    let wake = ExactShared::try_new(TaskWake {
        id,
        depth,
        ready: AtomicBool::new(true),
        active: AtomicBool::new(true),
        hub: WakeHubReference::Managed(hub),
        _claim: Some(wake_claim),
    })
    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
    let completion = CompletionOwner::Managed(completion);
    let queued = try_box(TypedQueuedRequest::new(future, completion.clone()))?;
    Ok((
        completion,
        control,
        WakeOwner::Managed(wake),
        QueuedFuture(Box::into_pin(queued)),
    ))
}

pub(super) fn managed_driver_control_bytes() -> Result<u64, HostAllocationError> {
    [
        ExactShared::<WakeHub>::requested_bytes(),
        ExactShared::<RefCell<DriverState>>::requested_bytes(),
        ExactShared::<ManagedDriverPort>::requested_bytes(),
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        total
            .checked_add(u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow)?)
            .ok_or(HostAllocationError::Overflow)
    })
}

pub(super) fn prepare_managed_driver(
    permit: &dyn ExactHostAllocationPermit,
    allocator: AdmittedCollectionAllocator,
    authority: RuntimeContext,
    dynamic_admission: Option<DynamicEvalAdmissionPort>,
    scratch: Option<Arc<runmat_runtime::context::host_effect::ManagedHostEffectStageReservation>>,
) -> Result<DriverOwner, ExactHostNumericAllocationError> {
    let hub_claim = permit.reserve_exact_claim(ExactShared::<WakeHub>::requested_bytes())?;
    let hub = ExactShared::try_new(WakeHub {
        state: Mutex::new(WakeHubState {
            ready: ReadyStorage::managed(allocator.clone()),
            host: None,
        }),
        _claim: Some(hub_claim),
    })
    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
    let state_claim =
        permit.reserve_exact_claim(ExactShared::<RefCell<DriverState>>::requested_bytes())?;
    let state = ExactShared::try_new(RefCell::new(DriverState {
        dynamic_admission,
        tasks: TaskStorage::managed(allocator),
        current: None,
        next_id: 1,
        accepting: true,
        retiring: false,
        wake_hub: WakeHubOwner::Managed(hub),
        authority: Some(authority),
        _scratch: scratch,
        _claim: Some(state_claim),
    }))
    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
    let port_claim =
        permit.reserve_exact_claim(ExactShared::<ManagedDriverPort>::requested_bytes())?;
    let port = ExactShared::try_new(ManagedDriverPort {
        state: RefCell::new(Some(state.clone())),
        _claim: port_claim,
    })
    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
    Ok(DriverOwner::Managed { state, port })
}
