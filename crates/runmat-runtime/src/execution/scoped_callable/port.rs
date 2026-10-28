//! Exact capability lookup; no source-name resolver or borrowed solver state.
use super::*;
use futures::channel::{mpsc, oneshot};
use runmat_thread_local::runmat_thread_local;
use runmat_value::{CallableInstanceId, ScopedFunctionRevoker};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::{Rc, Weak},
};

runmat_thread_local! {
    static PORTS: RefCell<HashMap<CallableInstanceId, Weak<Port>>> = RefCell::new(HashMap::new());
}

pub(super) struct Pending {
    pub call: Request,
    pub reply: oneshot::Sender<BuiltinResult<RootedValueSequence>>,
}
pub(super) struct Port {
    pub handle: ScopedFunctionHandle,
    pub origin: super::origin::Origin,
    pub busy: Cell<bool>,
    pub driving: Cell<bool>,
    pub sender: mpsc::UnboundedSender<Pending>,
    revoker: ScopedFunctionRevoker,
}

pub(super) struct Registration {
    pub port: Rc<Port>,
    pub receiver: mpsc::UnboundedReceiver<Pending>,
}
impl Registration {
    pub fn new(
        runtime: Option<&RuntimeContext>,
        token: Option<&CancellationToken>,
    ) -> BuiltinResult<Self> {
        let origin = super::origin::Origin::new(runtime, token)?;
        let (handle, revoker) = ScopedFunctionHandle::issue();
        let (sender, receiver) = mpsc::unbounded();
        let port = Rc::new(Port {
            handle,
            origin,
            busy: Cell::new(false),
            driving: Cell::new(false),
            sender,
            revoker,
        });
        PORTS.with(|ports| {
            ports
                .borrow_mut()
                .insert(port.handle.instance_id(), Rc::downgrade(&port));
        });
        Ok(Self { port, receiver })
    }
}
impl Drop for Registration {
    fn drop(&mut self) {
        self.port.revoker.revoke();
        self.receiver.close();
        PORTS.with(|ports| {
            ports.borrow_mut().remove(&self.port.handle.instance_id());
        });
    }
}

#[derive(Clone, Copy)]
pub(super) enum AfterChild {
    Retire,
    Park,
}
/// Close serving before the child is destroyed. A longer-lived auxiliary
/// registration parks between real user events and retires with its owner.
pub(super) struct ChildExit {
    port: Rc<Port>,
    after: AfterChild,
}
impl ChildExit {
    pub fn new(port: &Rc<Port>, after: AfterChild) -> Self {
        Self {
            port: Rc::clone(port),
            after,
        }
    }
    pub fn close(&self) {
        self.port.driving.set(false);
        if matches!(self.after, AfterChild::Retire) {
            self.port.revoker.revoke();
        }
    }
}
impl Drop for ChildExit {
    fn drop(&mut self) {
        self.close();
    }
}
impl Port {
    fn validate_capability(
        &self,
        runtime: Option<&RuntimeContext>,
        handle: &ScopedFunctionHandle,
    ) -> BuiltinResult<()> {
        if !handle.is_live() || !handle.same_capability(&self.handle) {
            return Err(failure(Failure::Expired));
        }
        if !self.origin.matches(runtime) {
            return Err(failure(Failure::Foreign));
        }
        check_original_cancellation(self.origin.runtime(), self.origin.token())?;
        Ok(())
    }

    fn validate_retention(
        &self,
        runtime: Option<&RuntimeContext>,
        handle: &ScopedFunctionHandle,
    ) -> BuiltinResult<()> {
        self.validate_capability(runtime, handle)?;
        require_ordinary_queue(runtime)
    }

    pub fn validate(
        &self,
        runtime: Option<&RuntimeContext>,
        handle: &ScopedFunctionHandle,
    ) -> BuiltinResult<()> {
        self.validate_capability(runtime, handle)?;
        if !self.driving.get() {
            return Err(failure(Failure::Unavailable));
        }
        require_ordinary_queue(runtime)?;
        if self.busy.get() {
            return Err(failure(Failure::Busy));
        }
        Ok(())
    }
}

fn require_ordinary_queue(runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
    crate::context::host_allocation::require_unbounded_workspace(
        runtime,
        "scoped callable",
        "finite host-memory scoped-callable invocation is unsupported",
    )
}

pub(super) fn lookup(
    runtime: Option<&RuntimeContext>,
    handle: &ScopedFunctionHandle,
) -> BuiltinResult<Rc<Port>> {
    let port = registered(handle)?;
    port.validate(runtime, handle)?;
    Ok(port)
}

pub(super) fn validate_retention(
    runtime: Option<&RuntimeContext>,
    handle: &ScopedFunctionHandle,
) -> BuiltinResult<()> {
    registered(handle)?.validate_retention(runtime, handle)
}

fn registered(handle: &ScopedFunctionHandle) -> BuiltinResult<Rc<Port>> {
    if !handle.is_live() {
        return Err(failure(Failure::Expired));
    }
    let port = PORTS
        .with(|ports| {
            ports
                .borrow()
                .get(&handle.instance_id())
                .and_then(Weak::upgrade)
        })
        .ok_or_else(|| failure(Failure::Unavailable))?;
    Ok(port)
}

pub(super) struct Serving(Rc<Port>);
impl Serving {
    pub fn enter(port: &Rc<Port>) -> BuiltinResult<Self> {
        if port.busy.replace(true) {
            return Err(failure(Failure::Busy));
        }
        Ok(Self(Rc::clone(port)))
    }
}
impl Drop for Serving {
    fn drop(&mut self) {
        self.0.busy.set(false);
    }
}
