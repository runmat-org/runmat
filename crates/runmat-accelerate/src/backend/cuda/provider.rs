use std::sync::Arc;

use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use runmat_accelerate_api::{
    native_buffer_byte_length, AccelDownloadFuture, AccelIntegerDownloadFuture,
    AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage,
    HostIntegerTensorOwned, HostIntegerTensorView, HostLogicalTensorView, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostNumericTensorView, HostTensorOwned,
    HostTensorView, NativeDeviceAccess, NativeDeviceApi, NativeDeviceBuffer, NativeDeviceClass,
    NativeDeviceComponent, NativeDeviceContext, NativeDeviceContextGuard,
    NativeDeviceInitialization,
};

use super::allocation_state::{Allocation, AllocationState};
use super::driver::{CudaDriver, DevicePointer};
use super::transfer::{
    numeric_owned_bytes_mut, numeric_to_integer, numeric_view_bytes, zeroed_numeric,
};

static PROVIDER: OnceCell<Option<&'static CudaProvider>> = OnceCell::new();

pub struct CudaProvider {
    driver: Arc<CudaDriver>,
    ordinal: u32,
    name: String,
    memory_bytes: u64,
    device_id: u32,
    allocations: Arc<AllocationState>,
}

impl std::fmt::Debug for CudaProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaProvider")
            .field("ordinal", &self.ordinal)
            .field("name", &self.name)
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

impl CudaProvider {
    fn initialize(ordinal: u32) -> Result<Option<Self>> {
        let Some(driver) = CudaDriver::load()? else {
            return Ok(None);
        };
        let driver = Arc::new(driver);
        let (cuda_device, context, name, memory_bytes) = driver.initialize(ordinal as i32)?;
        let device_id = runmat_accelerate_api::next_device_id();
        let allocations = Arc::new(AllocationState::new(
            driver.clone(),
            cuda_device,
            context,
            device_id,
        ));
        Ok(Some(Self {
            driver,
            ordinal,
            name,
            memory_bytes: u64::try_from(memory_bytes)
                .map_err(|_| anyhow!("CUDA device memory exceeds the supported range"))?,
            device_id,
            allocations,
        }))
    }

    fn with_context<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        self.allocations.with_context(operation)
    }

    fn allocation(&self, handle: &GpuTensorHandle) -> Result<Allocation> {
        self.allocations.allocation(handle)
    }

    fn allocate(
        &self,
        shape: &[usize],
        class: NativeDeviceClass,
        storage: GpuTensorStorage,
        initialization: NativeDeviceInitialization,
    ) -> Result<GpuTensorHandle> {
        self.allocate_initialized(shape, class, storage, |pointer, byte_length| {
            if initialization == NativeDeviceInitialization::Zeroed {
                self.driver.zero(pointer, byte_length)
            } else {
                Ok(())
            }
        })
    }

    /// Keep fresh pointers unpublished until all synchronous initialization
    /// succeeds. The allocation owner also retires them on initializer unwind.
    fn allocate_initialized(
        &self,
        shape: &[usize],
        class: NativeDeviceClass,
        storage: GpuTensorStorage,
        initialize: impl FnOnce(DevicePointer, usize) -> Result<()>,
    ) -> Result<GpuTensorHandle> {
        class.validate_storage(storage)?;
        let byte_length = native_buffer_byte_length(shape, class.element_type(), storage)?;
        self.allocations.allocate(
            shape.to_vec(),
            class,
            storage,
            byte_length,
            || self.driver.allocate(byte_length),
            |pointer| initialize(pointer, byte_length),
        )
    }

    fn upload_numeric_impl(&self, host: &HostNumericTensorView<'_>) -> Result<GpuTensorHandle> {
        host.validate()?;
        let bytes = numeric_view_bytes(host.data);
        let handle = self.allocate_initialized(
            host.shape,
            NativeDeviceClass::Numeric(host.element_type()),
            host.storage,
            |pointer, _| self.driver.upload(pointer, bytes),
        )?;
        runmat_value::record_host_copy(runmat_value::HostCopyReason::ProviderUpload, bytes.len());
        Ok(handle)
    }

    fn download_numeric_impl(&self, handle: &GpuTensorHandle) -> Result<HostNumericTensorOwned> {
        let allocation = self.allocation(handle)?;
        let element_type = handle
            .descriptor()
            .element_type
            .ok_or_else(|| anyhow!("CUDA handle is missing its element type"))?;
        let storage = handle
            .descriptor()
            .storage
            .ok_or_else(|| anyhow!("CUDA handle is missing its storage layout"))?;
        let byte_length = native_buffer_byte_length(handle.shape(), element_type, storage)?;
        let lanes = byte_length
            .checked_div(element_type.element_size())
            .ok_or_else(|| anyhow!("invalid CUDA element size"))?;
        let mut data = zeroed_numeric(element_type, lanes);
        self.with_context(|| {
            self.driver
                .download(allocation.pointer, numeric_owned_bytes_mut(&mut data))
        })?;
        runmat_value::record_host_copy(runmat_value::HostCopyReason::ProviderReadback, byte_length);
        let result = HostNumericTensorOwned {
            data,
            shape: handle.shape().to_vec(),
            storage,
        };
        result.validate()?;
        Ok(result)
    }

    fn copy_component_impl(
        &self,
        source: &GpuTensorHandle,
        component: NativeDeviceComponent,
    ) -> Result<GpuTensorHandle> {
        let source_allocation = self.allocation(source)?;
        if source.descriptor().storage != Some(GpuTensorStorage::ComplexInterleaved) {
            return Err(anyhow!(
                "CUDA component extraction requires complex storage"
            ));
        }
        let element_type = source
            .descriptor()
            .element_type
            .ok_or_else(|| anyhow!("CUDA handle is missing its element type"))?;
        let element_bytes = element_type.element_size();
        let offset = match component {
            NativeDeviceComponent::Real => 0,
            NativeDeviceComponent::Imaginary => element_bytes,
        };
        let elements = native_buffer_byte_length(
            source.shape(),
            element_type,
            GpuTensorStorage::ComplexInterleaved,
        )? / (element_bytes * 2);
        self.allocate_initialized(
            source.shape(),
            NativeDeviceClass::Numeric(element_type),
            GpuTensorStorage::Real,
            |pointer, _| {
                self.driver.copy_strided(
                    pointer,
                    element_bytes,
                    source_allocation.pointer + offset as u64,
                    element_bytes * 2,
                    element_bytes,
                    elements,
                )
            },
        )
    }

    fn combine_components_impl(
        &self,
        real: &GpuTensorHandle,
        imaginary: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        if real.shape() != imaginary.shape()
            || real.descriptor().element_type != imaginary.descriptor().element_type
            || real.descriptor().storage != Some(GpuTensorStorage::Real)
            || imaginary.descriptor().storage != Some(GpuTensorStorage::Real)
        {
            return Err(anyhow!(
                "CUDA complex components must be matching real arrays"
            ));
        }
        let real_allocation = self.allocation(real)?;
        let imaginary_allocation = self.allocation(imaginary)?;
        let element_type = real
            .descriptor()
            .element_type
            .ok_or_else(|| anyhow!("CUDA handle is missing its element type"))?;
        let element_bytes = element_type.element_size();
        let elements =
            native_buffer_byte_length(real.shape(), element_type, GpuTensorStorage::Real)?
                / element_bytes;
        self.allocate_initialized(
            real.shape(),
            NativeDeviceClass::Numeric(element_type),
            GpuTensorStorage::ComplexInterleaved,
            |pointer, _| {
                self.driver.copy_strided(
                    pointer,
                    element_bytes * 2,
                    real_allocation.pointer,
                    element_bytes,
                    element_bytes,
                    elements,
                )?;
                self.driver.copy_strided(
                    pointer + element_bytes as u64,
                    element_bytes * 2,
                    imaginary_allocation.pointer,
                    element_bytes,
                    element_bytes,
                    elements,
                )
            },
        )
    }
}

impl AccelProvider for CudaProvider {
    fn upload(&self, host: &HostTensorView<'_>) -> Result<GpuTensorHandle> {
        self.upload_numeric_impl(&HostNumericTensorView {
            data: HostNumericDataView::F64(host.data),
            shape: host.shape,
            storage: GpuTensorStorage::Real,
        })
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let downloaded = self.download_numeric_impl(handle)?;
            let HostNumericDataOwned::F64(data) = downloaded.data else {
                return Err(anyhow!("legacy CUDA download requires double storage"));
            };
            Ok(HostTensorOwned {
                data,
                shape: downloaded.shape,
                storage: downloaded.storage,
            })
        })
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        Ok(self.upload_numeric_impl(host)?)
    }

    fn upload_logical_raw(
        &self,
        host: &HostLogicalTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        host.validate()?;
        let result = self
            .allocate_initialized(
                host.shape,
                NativeDeviceClass::Logical,
                GpuTensorStorage::Real,
                |pointer, _| self.driver.upload(pointer, host.data),
            )
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?;
        runmat_value::record_host_copy(
            runmat_value::HostCopyReason::ProviderUpload,
            host.data.len(),
        );
        Ok(result)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move { self.download_numeric_impl(handle) })
    }

    fn download_numeric_host_workspace_bound(
        &self,
        handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        // `zeroed_numeric` is downloaded into place and returned without a
        // second host copy; the CUDA allocation itself is provider memory.
        // The result also owns one copied shape Vec.
        runmat_accelerate_api::checked_numeric_download_host_payload_bytes(handle)
            .and_then(|payload| {
                u64::try_from(
                    handle
                        .shape()
                        .len()
                        .checked_mul(std::mem::size_of::<usize>())?,
                )
                .ok()
                .and_then(|shape| payload.checked_add(shape))
            })
            .map(runmat_execution::ResourceByteBound::Known)
            .unwrap_or(runmat_execution::ResourceByteBound::Unavailable(
                runmat_execution::ResourceBoundUnavailable::NoScratchBound,
            ))
    }

    fn upload_integer(&self, host: &HostIntegerTensorView<'_>) -> Result<GpuTensorHandle> {
        self.upload_numeric_impl(&(*host).into())
    }

    fn download_integer<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelIntegerDownloadFuture<'a> {
        Box::pin(async move {
            let downloaded = self.download_numeric_impl(handle)?;
            let data = numeric_to_integer(downloaded.data)?;
            Ok(HostIntegerTensorOwned {
                data,
                shape: downloaded.shape,
            })
        })
    }

    fn acquire_live(
        &self,
        wire: &runmat_accelerate_api::allocation::GpuTensorWireDescriptor,
    ) -> Result<GpuTensorHandle, runmat_accelerate_api::allocation::AllocationAcquisitionError>
    {
        self.allocations.acquire(wire)
    }

    fn device_info(&self) -> String {
        format!("{} (CUDA device {})", self.name, self.ordinal)
    }

    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn device_info_struct(&self) -> runmat_accelerate_api::ApiDeviceInfo {
        runmat_accelerate_api::ApiDeviceInfo {
            device_id: self.device_id,
            name: self.name.clone(),
            vendor: "NVIDIA".into(),
            memory_bytes: Some(self.memory_bytes),
            backend: Some("cuda".into()),
        }
    }

    fn execution_accelerator_device(
        &self,
        inventory_epoch: u64,
    ) -> Result<Option<runmat_execution::resource::AcceleratorDevice>> {
        runmat_accelerate_api::execution_accelerator_device(
            self,
            runmat_accelerate_api::RUNMAT_CUDA_PROVIDER_ID,
            runmat_accelerate_api::RUNMAT_BUILTIN_PROVIDER_VERSION,
            runmat_accelerate_api::ExecutionProviderContract::CudaNativeV1,
            "primary-gpu",
            &format!("cuda-ordinal-{}", self.ordinal),
            inventory_epoch,
        )
    }

    fn native_device_api(&self) -> Option<NativeDeviceApi> {
        Some(NativeDeviceApi::Cuda)
    }

    fn native_device_context(&self) -> Result<NativeDeviceContext> {
        Ok(NativeDeviceContext {
            api: NativeDeviceApi::Cuda,
            provider_device_id: self.device_id,
            device_ordinal: self.ordinal,
            context_identity: self.allocations.context_identity()? as u64,
            stream_identity: 0,
        })
    }

    fn enter_native_device_context(&self) -> Result<NativeDeviceContextGuard> {
        self.allocations.enter_native()
    }

    fn export_native_device_buffer_raw(
        &self,
        handle: &GpuTensorHandle,
        access: NativeDeviceAccess,
    ) -> Result<NativeDeviceBuffer> {
        let allocation = self.allocation(handle)?;
        if access == NativeDeviceAccess::ReadWrite {
            self.allocations.validate_native_write(handle)?;
        }
        Ok(NativeDeviceBuffer {
            context: self.native_device_context()?,
            device_address: allocation.pointer,
            byte_length: native_buffer_byte_length(
                handle.shape(),
                handle.physical().element_type(),
                handle.physical().storage(),
            )?,
            descriptor: allocation.descriptor,
        })
    }

    fn allocate_native_device_buffer(
        &self,
        shape: &[usize],
        class: NativeDeviceClass,
        storage: GpuTensorStorage,
        initialization: NativeDeviceInitialization,
    ) -> Result<GpuTensorHandle> {
        self.allocate(shape, class, storage, initialization)
    }

    fn copy_native_device_buffer(&self, source: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let source_allocation = self.allocation(source)?;
        self.allocate_initialized(
            source.shape(),
            if source.view().is_logical() {
                NativeDeviceClass::Logical
            } else {
                NativeDeviceClass::Numeric(source.physical().element_type())
            },
            source
                .descriptor()
                .storage
                .ok_or_else(|| anyhow!("CUDA handle is missing its storage layout"))?,
            |pointer, byte_length| {
                self.driver
                    .copy(pointer, source_allocation.pointer, byte_length)
            },
        )
    }

    fn copy_native_device_component(
        &self,
        source: &GpuTensorHandle,
        component: NativeDeviceComponent,
    ) -> Result<GpuTensorHandle> {
        self.copy_component_impl(source, component)
    }

    fn combine_native_device_components(
        &self,
        real: &GpuTensorHandle,
        imaginary: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        self.combine_components_impl(real, imaginary)
    }

    fn synchronize_native_device(&self) -> Result<()> {
        self.with_context(|| self.driver.synchronize())
    }
}

pub fn register_cuda_provider() -> Result<Option<&'static CudaProvider>> {
    PROVIDER
        .get_or_try_init(|| {
            let Some(provider) = CudaProvider::initialize(0)? else {
                return Ok(None);
            };
            let provider = Box::leak(Box::new(provider));
            // SAFETY: the provider is intentionally leaked for process lifetime
            // and owns a unique device id allocated by the central registry.
            unsafe { runmat_accelerate_api::register_device_provider(provider) };
            Ok(Some(provider))
        })
        .copied()
}
