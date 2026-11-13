use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocation, TestAllocationOwner};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView,
    NativeDeviceAccess, NativeDeviceApi, NativeDeviceBuffer, NativeDeviceContext,
    NativeDeviceContextGuard, NumericElementType,
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
    retained: Mutex<Vec<RetirementProbe>>,
}

pub(super) struct Provider {
    allocations: TestAllocationOwner,
    events: Arc<Events>,
    enter: Fault,
    synchronize: Fault,
    leave: Fault,
}

impl Provider {
    pub(super) fn new(enter: Fault, synchronize: Fault, leave: Fault) -> &'static Self {
        Box::leak(Box::new(Self {
            allocations: TestAllocationOwner::new(),
            events: Arc::default(),
            enter,
            synchronize,
            leave,
        }))
    }

    pub(super) fn events(&self) -> Vec<&'static str> {
        self.events.log.lock().unwrap().clone()
    }

    pub(super) fn empty_allocation(&self) -> TestAllocation {
        self.allocations
            .allocate_zeroed(vec![0, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap()
    }

    pub(super) fn expect_retained(&self, probe: &RetirementProbe) {
        self.events.retained.lock().unwrap().push(probe.clone());
    }
}

impl AccelProvider for Provider {
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("context fixture has no numerical allocations")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("context fixture has no numerical allocations") })
    }
    fn device_info(&self) -> String {
        "native context lifecycle fixture".into()
    }
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
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
            assert!(events
                .retained
                .lock()
                .unwrap()
                .iter()
                .all(|probe| probe.retirements() == 0));
            assert!(
                events.active.swap(false, Ordering::SeqCst),
                "exit must occur exactly once"
            );
            events.log.lock().unwrap().push("leave");
            fault.apply("exit")
        }))
    }

    fn synchronize_native_device(&self) -> anyhow::Result<()> {
        assert!(self
            .events
            .retained
            .lock()
            .unwrap()
            .iter()
            .all(|probe| probe.retirements() == 0));
        assert!(
            self.events.active.load(Ordering::SeqCst),
            "context must remain current during synchronization"
        );
        self.events.log.lock().unwrap().push("synchronize");
        self.synchronize.apply("sync")
    }

    fn export_native_device_buffer_raw(
        &self,
        handle: &GpuTensorHandle,
        _: NativeDeviceAccess,
    ) -> anyhow::Result<NativeDeviceBuffer> {
        assert!(self.events.active.load(Ordering::SeqCst));
        assert_eq!(handle.device_id(), self.device_id());
        assert_eq!(handle.physical().payload_bytes(), 0);
        self.events.log.lock().unwrap().push("export");
        Ok(NativeDeviceBuffer {
            context: NativeDeviceContext {
                api: NativeDeviceApi::Cuda,
                provider_device_id: self.device_id(),
                device_ordinal: 0,
                context_identity: 1,
                stream_identity: 1,
            },
            device_address: 0,
            byte_length: 0,
            descriptor: handle.descriptor(),
        })
    }
}
