use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};

use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostNumericDataOwned,
    HostNumericTensorOwned, HostNumericTensorView, HostTensorOwned, HostTensorView,
    NativeDeviceBuffer, NativeDeviceContext, NativeDeviceContextGuard,
};

use super::super::*;

#[derive(Debug, Clone)]
struct Allocation {
    lanes: Box<[f64]>,
    shape: Vec<usize>,
    storage: GpuTensorStorage,
}

pub(super) struct FixtureCudaProvider {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    allocations: Mutex<BTreeMap<u64, runmat_accelerate_api::test_support::RetirementProbe>>,
    pub(super) context_enters: AtomicUsize,
    pub(super) context_leaves: Arc<AtomicUsize>,
}

impl FixtureCudaProvider {
    pub(super) fn shared() -> &'static Self {
        static PROVIDER: OnceLock<&'static FixtureCudaProvider> = OnceLock::new();
        let provider = *PROVIDER.get_or_init(|| {
            Box::leak(Box::new(Self {
                owner: runmat_accelerate_api::test_support::TestAllocationOwner::new(),
                allocations: Mutex::new(BTreeMap::new()),
                context_enters: AtomicUsize::new(0),
                context_leaves: Arc::new(AtomicUsize::new(0)),
            }))
        });
        static REGISTER: Once = Once::new();
        REGISTER.call_once(|| {
            // SAFETY: the fixture provider is leaked for the test process
            // lifetime and owns a unique device id.
            unsafe { runmat_accelerate_api::register_device_provider(provider) };
        });
        provider
    }

    pub(super) fn test_gate() -> &'static Mutex<()> {
        static GATE: OnceLock<Mutex<()>> = OnceLock::new();
        GATE.get_or_init(|| Mutex::new(()))
    }

    pub(super) fn reset(&self) {
        assert_eq!(
            self.live_count(),
            0,
            "previous witness retained fixture allocations"
        );
        self.allocations.lock().unwrap().clear();
        self.context_enters.store(0, Ordering::Release);
        self.context_leaves.store(0, Ordering::Release);
    }

    fn insert(
        &self,
        shape: Vec<usize>,
        storage: GpuTensorStorage,
        lanes: Box<[f64]>,
    ) -> GpuTensorHandle {
        let allocation = self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(lanes.into_vec()),
                shape,
                storage,
            })
            .unwrap();
        self.allocations
            .lock()
            .unwrap()
            .insert(allocation.handle.buffer_id(), allocation.probe);
        allocation.handle
    }

    fn probe(
        &self,
        handle: &GpuTensorHandle,
    ) -> anyhow::Result<runmat_accelerate_api::test_support::RetirementProbe> {
        if handle.device_id() != self.device_id() {
            anyhow::bail!("fixture handle belongs to another provider");
        }
        self.allocations
            .lock()
            .unwrap()
            .get(&handle.buffer_id())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("fixture allocation is unavailable"))
    }

    fn insert_logical(&self, data: Vec<u8>, shape: Vec<usize>) -> GpuTensorHandle {
        let allocation = self.owner.allocate_logical(data, shape).unwrap();
        self.allocations
            .lock()
            .unwrap()
            .insert(allocation.handle.buffer_id(), allocation.probe);
        allocation.handle
    }

    fn download_payload(&self, handle: &GpuTensorHandle) -> anyhow::Result<HostNumericTensorOwned> {
        self.probe(handle)?.with_data(handle, |data| {
            let data = match data {
                HostNumericDataView::F64(values) => HostNumericDataOwned::F64(values.to_vec()),
                HostNumericDataView::U8(values) => HostNumericDataOwned::U8(values.to_vec()),
                _ => anyhow::bail!("fixture only supports double and logical payloads"),
            };
            Ok(HostNumericTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: handle.physical().storage(),
            })
        })?
    }

    fn allocation(&self, handle: &GpuTensorHandle) -> anyhow::Result<Allocation> {
        self.probe(handle)?.with_data(handle, |data| {
            let HostNumericDataView::F64(values) = data else {
                panic!("fixture lost its F64 backing")
            };
            Allocation {
                lanes: values.to_vec().into_boxed_slice(),
                shape: handle.shape().to_vec(),
                storage: handle.physical().storage(),
            }
        })
    }

    pub(super) fn frees(&self) -> usize {
        self.allocations
            .lock()
            .unwrap()
            .values()
            .map(|probe| probe.retirements())
            .sum()
    }

    pub(super) fn live_count(&self) -> usize {
        self.allocations
            .lock()
            .unwrap()
            .values()
            .filter(|probe| probe.retirements() == 0)
            .count()
    }
}

impl AccelProvider for FixtureCudaProvider {
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        Ok(self.upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(host.data),
            shape: host.shape,
            storage: GpuTensorStorage::Real,
        })?)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        let result = self.allocation(handle).map(|allocation| HostTensorOwned {
            data: allocation.lanes.into_vec(),
            shape: allocation.shape,
            storage: allocation.storage,
        });
        Box::pin(async move { result })
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        host.validate()?;
        let HostNumericDataView::F64(values) = host.data else {
            return Err(anyhow::anyhow!("fixture provider accepts only double data").into());
        };
        runmat_value::record_host_copy(
            runmat_value::HostCopyReason::ProviderUpload,
            values.len().saturating_mul(std::mem::size_of::<f64>()),
        );
        Ok(self.insert(
            host.shape.to_vec(),
            host.storage,
            values.to_vec().into_boxed_slice(),
        ))
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        let result = self.download_payload(handle).inspect(|allocation| {
            runmat_value::record_host_copy(
                runmat_value::HostCopyReason::ProviderReadback,
                allocation
                    .data
                    .len()
                    .saturating_mul(allocation.data.element_type().element_size()),
            );
        });
        Box::pin(async move { result })
    }

    fn upload_logical_raw(
        &self,
        host: &HostLogicalTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        host.validate()?;
        Ok(self.insert_logical(host.data.to_vec(), host.shape.to_vec()))
    }

    fn device_info(&self) -> String {
        "fixture CUDA provider".into()
    }

    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn native_device_api(&self) -> Option<NativeDeviceApi> {
        Some(NativeDeviceApi::Cuda)
    }

    fn native_device_context(&self) -> anyhow::Result<NativeDeviceContext> {
        Ok(NativeDeviceContext {
            api: NativeDeviceApi::Cuda,
            provider_device_id: self.device_id(),
            device_ordinal: 0,
            context_identity: 0xCAFE,
            stream_identity: 0,
        })
    }

    fn enter_native_device_context(&self) -> anyhow::Result<NativeDeviceContextGuard> {
        self.context_enters.fetch_add(1, Ordering::AcqRel);
        let leaves = Arc::clone(&self.context_leaves);
        Ok(NativeDeviceContextGuard::new(move || {
            leaves.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }))
    }

    fn export_native_device_buffer_raw(
        &self,
        handle: &GpuTensorHandle,
        _access: NativeDeviceAccess,
    ) -> anyhow::Result<NativeDeviceBuffer> {
        let context = self.native_device_context()?;
        self.probe(handle)?
            .with_native_address(handle, |device_address, byte_length| NativeDeviceBuffer {
                context,
                device_address,
                byte_length,
                descriptor: handle.descriptor(),
            })
    }

    fn allocate_native_device_buffer(
        &self,
        shape: &[usize],
        class: NativeDeviceClass,
        storage: GpuTensorStorage,
        initialization: NativeDeviceInitialization,
    ) -> anyhow::Result<GpuTensorHandle> {
        class.validate_storage(storage)?;
        let bytes =
            runmat_accelerate_api::native_buffer_byte_length(shape, class.element_type(), storage)?;
        if class == NativeDeviceClass::Logical {
            return Ok(self.insert_logical(vec![0; bytes], shape.to_vec()));
        }
        if class != NativeDeviceClass::Numeric(NumericElementType::F64) {
            anyhow::bail!("fixture provider accepts only double data");
        }
        let mut length = shape.iter().product::<usize>();
        if storage == GpuTensorStorage::ComplexInterleaved {
            length = length.saturating_mul(2);
        }
        let fill = match initialization {
            NativeDeviceInitialization::Zeroed => 0.0,
            NativeDeviceInitialization::Uninitialized => f64::NAN,
        };
        Ok(self.insert(
            shape.to_vec(),
            storage,
            vec![fill; length].into_boxed_slice(),
        ))
    }

    fn copy_native_device_buffer(
        &self,
        source: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        if runmat_accelerate_api::handle_is_logical(source) {
            let allocation = self.download_payload(source)?;
            let HostNumericDataOwned::U8(data) = allocation.data else {
                anyhow::bail!("native logical fixture storage is not U8")
            };
            return Ok(self.insert_logical(data, allocation.shape));
        }
        let allocation = self.allocation(source)?;
        Ok(self.insert(allocation.shape, allocation.storage, allocation.lanes))
    }

    fn copy_native_device_component(
        &self,
        source: &GpuTensorHandle,
        component: NativeDeviceComponent,
    ) -> anyhow::Result<GpuTensorHandle> {
        let allocation = self.allocation(source)?;
        if allocation.storage != GpuTensorStorage::ComplexInterleaved {
            anyhow::bail!("fixture component source is not complex");
        }
        let offset = usize::from(component == NativeDeviceComponent::Imaginary);
        let lanes = allocation
            .lanes
            .iter()
            .skip(offset)
            .step_by(2)
            .copied()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(self.insert(allocation.shape, GpuTensorStorage::Real, lanes))
    }

    fn combine_native_device_components(
        &self,
        real: &GpuTensorHandle,
        imaginary: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        let real = self.allocation(real)?;
        let imaginary = self.allocation(imaginary)?;
        if real.shape != imaginary.shape
            || real.storage != GpuTensorStorage::Real
            || imaginary.storage != GpuTensorStorage::Real
        {
            anyhow::bail!("fixture complex components do not match");
        }
        let lanes = real
            .lanes
            .iter()
            .zip(imaginary.lanes.iter())
            .flat_map(|(real, imaginary)| [*real, *imaginary])
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(self.insert(real.shape, GpuTensorStorage::ComplexInterleaved, lanes))
    }

    fn synchronize_native_device(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

pub(super) struct FixtureAmbientProvider {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    allocation: runmat_accelerate_api::test_support::TestAllocation,
    pub(super) downloads: AtomicUsize,
}

impl FixtureAmbientProvider {
    pub(super) fn shared() -> &'static Self {
        static PROVIDER: OnceLock<&'static FixtureAmbientProvider> = OnceLock::new();
        let provider = *PROVIDER.get_or_init(|| {
            let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
            let allocation = owner
                .allocate_numeric(HostNumericTensorOwned {
                    data: HostNumericDataOwned::F64(vec![11.0, 13.0]),
                    shape: vec![1, 2],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            Box::leak(Box::new(Self {
                owner,
                allocation,
                downloads: AtomicUsize::new(0),
            }))
        });
        static REGISTER: Once = Once::new();
        REGISTER.call_once(|| {
            // SAFETY: process-retained fixture and privately minted device identity.
            unsafe { runmat_accelerate_api::register_device_provider(provider) };
        });
        provider
    }

    pub(super) fn handle(&self) -> GpuTensorHandle {
        self.allocation.handle.clone()
    }
}

impl AccelProvider for FixtureAmbientProvider {
    fn upload(&self, _host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("ambient fixture does not accept uploads")
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        let result = if handle == &self.handle() {
            Ok(HostTensorOwned {
                data: vec![11.0, 13.0],
                shape: vec![1, 2],
                storage: GpuTensorStorage::Real,
            })
        } else {
            Err(anyhow::anyhow!("ambient fixture handle is unavailable"))
        };
        Box::pin(async move { result })
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        let result = if handle == &self.handle() {
            self.downloads.fetch_add(1, Ordering::AcqRel);
            runmat_value::record_host_copy(
                runmat_value::HostCopyReason::ProviderReadback,
                2 * std::mem::size_of::<f64>(),
            );
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![11.0, 13.0]),
                shape: vec![1, 2],
                storage: GpuTensorStorage::Real,
            })
        } else {
            Err(anyhow::anyhow!("ambient fixture handle is unavailable"))
        };
        Box::pin(async move { result })
    }

    fn device_info(&self) -> String {
        "fixture non-CUDA provider".into()
    }

    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
}
