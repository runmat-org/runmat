use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use runmat_accelerate_api::{
    native_buffer_byte_length, AccelDownloadFuture, AccelIntegerDownloadFuture,
    AccelNumericDownloadFuture, AccelProvider, GpuTensorDescriptor, GpuTensorHandle,
    GpuTensorStorage, HostIntegerTensorOwned, HostIntegerTensorView, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorOwned, HostNumericTensorView, HostTensorOwned,
    HostTensorView, NativeDeviceAccess, NativeDeviceApi, NativeDeviceBuffer, NativeDeviceComponent,
    NativeDeviceContext, NativeDeviceContextGuard, NativeDeviceInitialization, NumericElementType,
};

use super::driver::{CudaDriver, DevicePointer};
use super::transfer::{
    numeric_owned_bytes_mut, numeric_to_integer, numeric_view_bytes, zeroed_numeric,
};

static PROVIDER: OnceCell<Option<&'static CudaProvider>> = OnceCell::new();

#[derive(Debug, Clone, Copy)]
struct Allocation {
    pointer: DevicePointer,
    byte_length: usize,
    descriptor: GpuTensorDescriptor,
}

pub struct CudaProvider {
    driver: Arc<CudaDriver>,
    cuda_device: i32,
    ordinal: u32,
    context: usize,
    name: String,
    memory_bytes: u64,
    device_id: u32,
    next_buffer: AtomicU64,
    allocations: Mutex<HashMap<u64, Allocation>>,
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

impl Drop for CudaProvider {
    fn drop(&mut self) {
        if let Ok(allocations) = self.allocations.get_mut() {
            let pointers = allocations
                .values()
                .map(|allocation| allocation.pointer)
                .collect::<Vec<_>>();
            let _ = self.with_context(|| {
                for pointer in pointers {
                    let _ = self.driver.free(pointer);
                }
                Ok(())
            });
        }
        self.driver.release_primary_context(self.cuda_device);
    }
}

impl CudaProvider {
    fn initialize(ordinal: u32) -> Result<Option<Self>> {
        let Some(driver) = CudaDriver::load()? else {
            return Ok(None);
        };
        let driver = Arc::new(driver);
        let (cuda_device, context, name, memory_bytes) = driver.initialize(ordinal as i32)?;
        Ok(Some(Self {
            driver,
            cuda_device,
            ordinal,
            context,
            name,
            memory_bytes: u64::try_from(memory_bytes)
                .map_err(|_| anyhow!("CUDA device memory exceeds the supported range"))?,
            device_id: runmat_accelerate_api::next_device_id(),
            next_buffer: AtomicU64::new(1),
            allocations: Mutex::new(HashMap::new()),
        }))
    }

    fn with_context<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        self.driver.push_context(self.context)?;
        let result = operation();
        let leave = self.driver.pop_context();
        match (result, leave) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error.context("could not leave CUDA context")),
        }
    }

    fn allocation(&self, handle: &GpuTensorHandle) -> Result<Allocation> {
        if handle.device_id != self.device_id {
            return Err(anyhow!("CUDA handle belongs to a different provider"));
        }
        let allocation = self
            .allocations
            .lock()
            .map_err(|_| anyhow!("CUDA allocation registry is poisoned"))?
            .get(&handle.buffer_id)
            .copied()
            .ok_or_else(|| anyhow!("CUDA handle is stale or unknown"))?;
        if allocation.descriptor.element_type != handle.descriptor.element_type
            || allocation.descriptor.storage != handle.descriptor.storage
        {
            return Err(anyhow!(
                "CUDA handle descriptor does not match its allocation"
            ));
        }
        Ok(allocation)
    }

    fn insert(
        &self,
        shape: Vec<usize>,
        descriptor: GpuTensorDescriptor,
        pointer: DevicePointer,
        byte_length: usize,
    ) -> Result<GpuTensorHandle> {
        let buffer_id = self
            .next_buffer
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| anyhow!("CUDA buffer identity space is exhausted"))?;
        self.allocations
            .lock()
            .map_err(|_| anyhow!("CUDA allocation registry is poisoned"))?
            .insert(
                buffer_id,
                Allocation {
                    pointer,
                    byte_length,
                    descriptor,
                },
            );
        Ok(GpuTensorHandle {
            shape,
            device_id: self.device_id,
            buffer_id,
            descriptor,
        })
    }

    fn allocate(
        &self,
        shape: &[usize],
        element_type: NumericElementType,
        storage: GpuTensorStorage,
        initialization: NativeDeviceInitialization,
    ) -> Result<GpuTensorHandle> {
        let byte_length = native_buffer_byte_length(shape, element_type, storage)?;
        let pointer = self.with_context(|| {
            let pointer = self.driver.allocate(byte_length)?;
            if initialization == NativeDeviceInitialization::Zeroed {
                if let Err(error) = self.driver.zero(pointer, byte_length) {
                    let _ = self.driver.free(pointer);
                    return Err(error);
                }
            }
            Ok(pointer)
        })?;
        let descriptor = GpuTensorDescriptor::numeric(element_type, storage);
        self.insert(shape.to_vec(), descriptor, pointer, byte_length)
            .inspect_err(|_| {
                let _ = self.with_context(|| self.driver.free(pointer));
            })
    }

    fn upload_numeric_impl(&self, host: &HostNumericTensorView<'_>) -> Result<GpuTensorHandle> {
        host.validate()?;
        let handle = self.allocate(
            host.shape,
            host.element_type(),
            host.storage,
            NativeDeviceInitialization::Uninitialized,
        )?;
        let allocation = self.allocation(&handle)?;
        let bytes = numeric_view_bytes(host.data);
        let result = self.with_context(|| self.driver.upload(allocation.pointer, bytes));
        if let Err(error) = result {
            let _ = self.free(&handle);
            return Err(error);
        }
        runmat_value::record_host_copy(
            runmat_value::HostCopyReason::ProviderUpload,
            allocation.byte_length,
        );
        Ok(handle)
    }

    fn download_numeric_impl(&self, handle: &GpuTensorHandle) -> Result<HostNumericTensorOwned> {
        let allocation = self.allocation(handle)?;
        let element_type = handle
            .descriptor
            .element_type
            .ok_or_else(|| anyhow!("CUDA handle is missing its element type"))?;
        let storage = handle
            .descriptor
            .storage
            .ok_or_else(|| anyhow!("CUDA handle is missing its storage layout"))?;
        let lanes = allocation
            .byte_length
            .checked_div(element_type.element_size())
            .ok_or_else(|| anyhow!("invalid CUDA element size"))?;
        let mut data = zeroed_numeric(element_type, lanes);
        self.with_context(|| {
            self.driver
                .download(allocation.pointer, numeric_owned_bytes_mut(&mut data))
        })?;
        runmat_value::record_host_copy(
            runmat_value::HostCopyReason::ProviderReadback,
            allocation.byte_length,
        );
        let result = HostNumericTensorOwned {
            data,
            shape: handle.shape.clone(),
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
        if source.descriptor.storage != Some(GpuTensorStorage::ComplexInterleaved) {
            return Err(anyhow!(
                "CUDA component extraction requires complex storage"
            ));
        }
        let element_type = source
            .descriptor
            .element_type
            .ok_or_else(|| anyhow!("CUDA handle is missing its element type"))?;
        let output = self.allocate(
            &source.shape,
            element_type,
            GpuTensorStorage::Real,
            NativeDeviceInitialization::Uninitialized,
        )?;
        let output_allocation = self.allocation(&output)?;
        let element_bytes = element_type.element_size();
        let offset = match component {
            NativeDeviceComponent::Real => 0,
            NativeDeviceComponent::Imaginary => element_bytes,
        };
        let elements = source_allocation.byte_length / (element_bytes * 2);
        let result = self.with_context(|| {
            self.driver.copy_strided(
                output_allocation.pointer,
                element_bytes,
                source_allocation.pointer + offset as u64,
                element_bytes * 2,
                element_bytes,
                elements,
            )
        });
        if let Err(error) = result {
            let _ = self.free(&output);
            return Err(error);
        }
        Ok(output)
    }

    fn combine_components_impl(
        &self,
        real: &GpuTensorHandle,
        imaginary: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        if real.shape != imaginary.shape
            || real.descriptor.element_type != imaginary.descriptor.element_type
            || real.descriptor.storage != Some(GpuTensorStorage::Real)
            || imaginary.descriptor.storage != Some(GpuTensorStorage::Real)
        {
            return Err(anyhow!(
                "CUDA complex components must be matching real arrays"
            ));
        }
        let real_allocation = self.allocation(real)?;
        let imaginary_allocation = self.allocation(imaginary)?;
        let element_type = real
            .descriptor
            .element_type
            .ok_or_else(|| anyhow!("CUDA handle is missing its element type"))?;
        let output = self.allocate(
            &real.shape,
            element_type,
            GpuTensorStorage::ComplexInterleaved,
            NativeDeviceInitialization::Uninitialized,
        )?;
        let output_allocation = self.allocation(&output)?;
        let element_bytes = element_type.element_size();
        let elements = real_allocation.byte_length / element_bytes;
        let result = self.with_context(|| {
            self.driver.copy_strided(
                output_allocation.pointer,
                element_bytes * 2,
                real_allocation.pointer,
                element_bytes,
                element_bytes,
                elements,
            )?;
            self.driver.copy_strided(
                output_allocation.pointer + element_bytes as u64,
                element_bytes * 2,
                imaginary_allocation.pointer,
                element_bytes,
                element_bytes,
                elements,
            )
        });
        if let Err(error) = result {
            let _ = self.free(&output);
            return Err(error);
        }
        Ok(output)
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

    fn upload_numeric(&self, host: &HostNumericTensorView<'_>) -> Result<GpuTensorHandle> {
        self.upload_numeric_impl(host)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move { self.download_numeric_impl(handle) })
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

    fn free(&self, handle: &GpuTensorHandle) -> Result<()> {
        if handle.device_id != self.device_id {
            return Err(anyhow!("CUDA handle belongs to a different provider"));
        }
        let allocation = self
            .allocations
            .lock()
            .map_err(|_| anyhow!("CUDA allocation registry is poisoned"))?
            .remove(&handle.buffer_id)
            .ok_or_else(|| anyhow!("CUDA handle is stale or already freed"))?;
        self.with_context(|| self.driver.free(allocation.pointer))
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
        .map(Some)
    }

    fn native_device_api(&self) -> Option<NativeDeviceApi> {
        Some(NativeDeviceApi::Cuda)
    }

    fn native_device_context(&self) -> Result<NativeDeviceContext> {
        Ok(NativeDeviceContext {
            api: NativeDeviceApi::Cuda,
            provider_device_id: self.device_id,
            device_ordinal: self.ordinal,
            context_identity: self.context as u64,
            stream_identity: 0,
        })
    }

    fn enter_native_device_context(&self) -> Result<NativeDeviceContextGuard> {
        self.driver.push_context(self.context)?;
        let driver = Arc::clone(&self.driver);
        Ok(NativeDeviceContextGuard::new(move || driver.pop_context()))
    }

    fn export_native_device_buffer(
        &self,
        handle: &GpuTensorHandle,
        _access: NativeDeviceAccess,
    ) -> Result<NativeDeviceBuffer> {
        let allocation = self.allocation(handle)?;
        Ok(NativeDeviceBuffer {
            context: self.native_device_context()?,
            device_address: allocation.pointer,
            byte_length: allocation.byte_length,
            descriptor: allocation.descriptor,
        })
    }

    fn allocate_native_device_buffer(
        &self,
        shape: &[usize],
        element_type: NumericElementType,
        storage: GpuTensorStorage,
        initialization: NativeDeviceInitialization,
    ) -> Result<GpuTensorHandle> {
        self.allocate(shape, element_type, storage, initialization)
    }

    fn copy_native_device_buffer(&self, source: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let source_allocation = self.allocation(source)?;
        let output = self.allocate(
            &source.shape,
            source
                .descriptor
                .element_type
                .ok_or_else(|| anyhow!("CUDA handle is missing its element type"))?,
            source
                .descriptor
                .storage
                .ok_or_else(|| anyhow!("CUDA handle is missing its storage layout"))?,
            NativeDeviceInitialization::Uninitialized,
        )?;
        let output_allocation = self.allocation(&output)?;
        let result = self.with_context(|| {
            self.driver.copy(
                output_allocation.pointer,
                source_allocation.pointer,
                source_allocation.byte_length,
            )
        });
        if let Err(error) = result {
            let _ = self.free(&output);
            return Err(error);
        }
        Ok(output)
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
