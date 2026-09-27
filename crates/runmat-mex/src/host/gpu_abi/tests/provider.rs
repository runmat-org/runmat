use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorView, NativeDeviceApi,
    NativeDeviceContextGuard,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

#[derive(Clone, Copy)]
pub(super) enum Fault {
    None,
    Error,
    Panic,
}

impl Fault {
    fn apply(self, phase: &str) -> anyhow::Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Error => Err(anyhow::anyhow!("{phase} failure")),
            Self::Panic => panic!("{phase} panic"),
        }
    }
}

#[derive(Default)]
struct Events {
    log: Mutex<Vec<&'static str>>,
    active: AtomicBool,
}

pub(super) struct Provider {
    id: u32,
    events: Arc<Events>,
    enter: Fault,
    synchronize: Fault,
    leave: Fault,
}

impl Provider {
    pub(super) fn new(enter: Fault, synchronize: Fault, leave: Fault) -> &'static Self {
        Box::leak(Box::new(Self {
            id: runmat_accelerate_api::next_device_id(),
            events: Arc::default(),
            enter,
            synchronize,
            leave,
        }))
    }

    pub(super) fn events(&self) -> Vec<&'static str> {
        self.events.log.lock().unwrap().clone()
    }
}

impl AccelProvider for Provider {
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("context fixture has no numerical allocations")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("context fixture has no numerical allocations") })
    }
    fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
        anyhow::bail!("context fixture has no numerical allocations")
    }
    fn device_info(&self) -> String {
        "native context lifecycle fixture".into()
    }
    fn device_id(&self) -> u32 {
        self.id
    }
    fn native_device_api(&self) -> Option<NativeDeviceApi> {
        Some(NativeDeviceApi::Cuda)
    }

    fn enter_native_device_context(&self) -> anyhow::Result<NativeDeviceContextGuard> {
        self.events.log.lock().unwrap().push("enter");
        self.enter.apply("entry")?;
        assert!(!self.events.active.swap(true, Ordering::SeqCst));
        let events = self.events.clone();
        let fault = self.leave;
        Ok(NativeDeviceContextGuard::new(move || {
            assert!(
                events.active.swap(false, Ordering::SeqCst),
                "exit must occur exactly once"
            );
            events.log.lock().unwrap().push("leave");
            fault.apply("exit")
        }))
    }

    fn synchronize_native_device(&self) -> anyhow::Result<()> {
        assert!(
            self.events.active.load(Ordering::SeqCst),
            "context must remain current during synchronization"
        );
        self.events.log.lock().unwrap().push("synchronize");
        self.synchronize.apply("sync")
    }
}
