//! Native addresses are borrowed from exports retained by the entered context.
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, NativeDeviceAccess, OwnedNativeDeviceBuffer,
};

thread_local! {
    static ENTERED: RefCell<Vec<Rc<ExportScope>>> = const { RefCell::new(Vec::new()) };
}

pub(super) struct ExportScope {
    provider: &'static dyn AccelProvider,
    accepting: Cell<bool>,
    exports: RefCell<Vec<OwnedNativeDeviceBuffer>>,
}

impl ExportScope {
    pub(super) fn enter(provider: &'static dyn AccelProvider) -> Rc<Self> {
        let scope = Rc::new(Self {
            provider,
            accepting: Cell::new(true),
            exports: RefCell::new(Vec::new()),
        });
        ENTERED.with(|entered| entered.borrow_mut().push(scope.clone()));
        scope
    }

    pub(super) fn stop_exports(&self) {
        self.accepting.set(false);
    }

    /// Called only after synchronization and context exit have both been attempted.
    /// No allocation destructor executes while the scope stack is borrowed.
    pub(super) fn finish(self: &Rc<Self>) {
        self.stop_exports();
        ENTERED.with(|entered| {
            entered
                .borrow_mut()
                .retain(|scope| !Rc::ptr_eq(scope, self));
        });
        let exports = std::mem::take(&mut *self.exports.borrow_mut());
        drop(exports);
    }
}

pub(crate) fn native_device_address(
    handle: &GpuTensorHandle,
    access: NativeDeviceAccess,
) -> anyhow::Result<u64> {
    let scope = ENTERED.with(|entered| entered.borrow().last().cloned());
    let scope = scope.ok_or_else(|| {
        anyhow::anyhow!("native GPU address requires an entered invocation context")
    })?;
    if !scope.accepting.get() {
        anyhow::bail!("native GPU invocation context is leaving");
    }
    // Use the exact entered provider, never ambient provider reselection.
    if scope.provider.device_id() != handle.device_id() {
        anyhow::bail!("GPU array does not belong to the entered native context");
    }
    let export =
        runmat_accelerate_api::export_native_device_buffer(scope.provider, handle, access)?;
    if !scope.accepting.get() {
        anyhow::bail!("native GPU invocation context left during export");
    }
    let address = export.address_view().device_address;
    scope.exports.borrow_mut().push(export);
    Ok(address)
}
