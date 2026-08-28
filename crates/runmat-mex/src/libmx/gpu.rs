use runmat_accelerate_api::{
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorOwned,
    HostNumericTensorView, NativeDeviceAccess, NativeDeviceApi, NativeDeviceComponent,
    NativeDeviceInitialization, NumericElementType,
};
use runmat_value::{ComplexElement, HostNumericBuffer, IntegerStorage, NumericStorage};

use crate::mxarray::{MxArrayData, MxComplex, MxGpuLease, MxInterleavedStorage, MxNumeric};
use crate::{MxApi, MxArray, MxClassId};

impl MxApi {
    pub fn gpu_create_from_array(
        &mut self,
        source: *const MxArray,
        independent_copy: bool,
    ) -> Result<*mut MxArray, String> {
        let source = self.arena.get(source).map_err(|error| error.to_string())?;
        if let MxArrayData::Gpu(gpu) = source.data() {
            let target = cuda_provider()?;
            let owner = runmat_accelerate_api::provider_for_handle(gpu.lease.handle())
                .ok_or_else(|| "GPU array provider is unavailable".to_string())?;
            if owner.device_id() == target.device_id() {
                let lease = if independent_copy {
                    gpu.lease.duplicate().map_err(|error| error.to_string())?
                } else {
                    gpu.lease.clone()
                };
                let value = MxArray::gpu_owned(gpu.class_id, lease)?;
                return Ok(self.arena.allocate(value));
            }
            let downloaded = pollster::block_on(owner.download_numeric(gpu.lease.handle()))
                .map_err(|error| format!("could not transfer GPU array to CUDA: {error}"))?;
            let handle = target
                .upload_numeric(&downloaded.as_view())
                .map_err(|error| format!("could not upload GPU array to CUDA: {error}"))?;
            let value = MxArray::gpu_owned(gpu.class_id, MxGpuLease::owned(handle))?;
            return Ok(self.arena.allocate(value));
        }

        let (class_id, upload) = host_upload(source)?;
        let provider = cuda_provider()?;
        let handle = match &upload {
            UploadSource::Borrowed(view) => provider.upload_numeric(view),
            UploadSource::Owned(owned) => provider.upload_numeric(&owned.as_view()),
        }
        .map_err(|error| format!("could not upload array to CUDA: {error}"))?;
        let value = MxArray::gpu_owned(class_id, MxGpuLease::owned(handle))?;
        Ok(self.arena.allocate(value))
    }

    pub fn gpu_create(
        &mut self,
        class_id: MxClassId,
        shape: Vec<usize>,
        complex: bool,
        initialize: bool,
    ) -> Result<*mut MxArray, String> {
        let element_type = class_element_type(class_id)?;
        let storage = if complex {
            GpuTensorStorage::ComplexInterleaved
        } else {
            GpuTensorStorage::Real
        };
        let initialization = if initialize {
            NativeDeviceInitialization::Zeroed
        } else {
            NativeDeviceInitialization::Uninitialized
        };
        let handle = cuda_provider()?
            .allocate_native_device_buffer(&shape, element_type, storage, initialization)
            .map_err(|error| format!("could not allocate CUDA array: {error}"))?;
        let value = MxArray::gpu_owned(class_id, MxGpuLease::owned(handle))?;
        Ok(self.arena.allocate(value))
    }

    pub fn gpu_to_host(&mut self, source: *const MxArray) -> Result<*mut MxArray, String> {
        let source = self.arena.get(source).map_err(|error| error.to_string())?;
        let MxArrayData::Gpu(gpu) = source.data() else {
            return Err("mxGPUArray does not refer to GPU storage".into());
        };
        let provider = runmat_accelerate_api::provider_for_handle(gpu.lease.handle())
            .ok_or_else(|| "GPU array provider is unavailable".to_string())?;
        let downloaded = pollster::block_on(provider.download_numeric(gpu.lease.handle()))
            .map_err(|error| format!("could not gather GPU array: {error}"))?;
        let value = host_array_from_download(gpu.class_id, downloaded)?;
        Ok(self.arena.allocate(value))
    }

    pub fn gpu_data(&mut self, source: *mut MxArray, writable: bool) -> Result<u64, String> {
        let source = self
            .arena
            .get_mut(source)
            .map_err(|error| error.to_string())?;
        let MxArrayData::Gpu(gpu) = source.data_mut() else {
            return Err("mxGPUArray does not refer to GPU storage".into());
        };
        gpu.lease
            .device_address(if writable {
                NativeDeviceAccess::ReadWrite
            } else {
                NativeDeviceAccess::ReadOnly
            })
            .map_err(|error| error.to_string())
    }

    pub fn gpu_class_id(&self, source: *const MxArray) -> Result<MxClassId, String> {
        let source = self.arena.get(source).map_err(|error| error.to_string())?;
        let MxArrayData::Gpu(gpu) = source.data() else {
            return Err("mxGPUArray does not refer to GPU storage".into());
        };
        Ok(gpu.class_id)
    }

    pub fn is_gpu_array(&self, source: *const MxArray) -> bool {
        self.arena
            .get(source)
            .is_ok_and(|value| matches!(value.data(), MxArrayData::Gpu(_)))
    }

    pub fn gpu_is_same(&self, left: *const MxArray, right: *const MxArray) -> Result<bool, String> {
        let gpu = |pointer| {
            self.arena
                .get(pointer)
                .map_err(|error| error.to_string())
                .and_then(|value| match value.data() {
                    MxArrayData::Gpu(gpu) => Ok(gpu),
                    _ => Err("mxGPUArray does not refer to GPU storage".into()),
                })
        };
        let left = gpu(left)?;
        let right = gpu(right)?;
        Ok(left.lease.handle() == right.lease.handle())
    }

    pub fn gpu_copy_component(
        &mut self,
        source: *const MxArray,
        component: NativeDeviceComponent,
    ) -> Result<*mut MxArray, String> {
        let source = self.arena.get(source).map_err(|error| error.to_string())?;
        let MxArrayData::Gpu(gpu) = source.data() else {
            return Err("mxGPUArray does not refer to GPU storage".into());
        };
        let storage = gpu
            .lease
            .handle()
            .descriptor
            .storage
            .ok_or_else(|| "GPU array has no storage descriptor".to_string())?;
        let lease = match (storage, component) {
            (GpuTensorStorage::Real, NativeDeviceComponent::Real) => {
                gpu.lease.duplicate().map_err(|error| error.to_string())?
            }
            (GpuTensorStorage::Real, NativeDeviceComponent::Imaginary) => {
                let handle = cuda_provider()?
                    .allocate_native_device_buffer(
                        source.shape(),
                        class_element_type(gpu.class_id)?,
                        GpuTensorStorage::Real,
                        NativeDeviceInitialization::Zeroed,
                    )
                    .map_err(|error| error.to_string())?;
                MxGpuLease::owned(handle)
            }
            (GpuTensorStorage::ComplexInterleaved, component) => gpu
                .lease
                .copy_component(component)
                .map_err(|error| error.to_string())?,
        };
        let value = MxArray::gpu_owned(gpu.class_id, lease)?;
        Ok(self.arena.allocate(value))
    }

    pub fn gpu_create_complex(
        &mut self,
        real: *const MxArray,
        imaginary: *const MxArray,
    ) -> Result<*mut MxArray, String> {
        let component = |pointer| {
            let value = self.arena.get(pointer).map_err(|error| error.to_string())?;
            let MxArrayData::Gpu(gpu) = value.data() else {
                return Err("mxGPUArray does not refer to GPU storage".to_string());
            };
            if gpu.lease.handle().descriptor.storage != Some(GpuTensorStorage::Real) {
                return Err("complex GPU components must be real arrays".to_string());
            }
            Ok((value.shape().to_vec(), gpu.class_id, gpu.lease.clone()))
        };
        let (real_shape, real_class, real_lease) = component(real)?;
        let (imag_shape, imag_class, imag_lease) = component(imaginary)?;
        if real_shape != imag_shape || real_class != imag_class {
            return Err("complex GPU components must have matching shape and class".into());
        }
        let lease =
            MxGpuLease::combine(&real_lease, &imag_lease).map_err(|error| error.to_string())?;
        let value = MxArray::gpu_owned(real_class, lease)?;
        Ok(self.arena.allocate(value))
    }
}

fn cuda_provider() -> Result<&'static dyn runmat_accelerate_api::AccelProvider, String> {
    runmat_accelerate_api::provider_for_native_device(NativeDeviceApi::Cuda).ok_or_else(|| {
        "GPU MEX requires a registered CUDA provider; WGPU buffers cannot be exposed as CUDA pointers"
            .into()
    })
}

enum UploadSource<'a> {
    Borrowed(HostNumericTensorView<'a>),
    Owned(HostNumericTensorOwned),
}

fn host_upload(source: &MxArray) -> Result<(MxClassId, UploadSource<'_>), String> {
    match source.data() {
        MxArrayData::Numeric(MxNumeric { real, imag: None }) => Ok((
            source.class_id(),
            UploadSource::Borrowed(HostNumericTensorView {
                data: host_numeric_view(real)?,
                shape: source.shape(),
                storage: GpuTensorStorage::Real,
            }),
        )),
        MxArrayData::Numeric(MxNumeric {
            real,
            imag: Some(imag),
        }) => {
            let data = interleave_components(real, imag)?;
            runmat_value::record_host_copy(
                runmat_value::HostCopyReason::MemoryLayoutConversion,
                data.len()
                    .saturating_mul(data.element_type().element_size()),
            );
            Ok((
                source.class_id(),
                UploadSource::Owned(HostNumericTensorOwned {
                    data,
                    shape: source.shape().to_vec(),
                    storage: GpuTensorStorage::ComplexInterleaved,
                }),
            ))
        }
        MxArrayData::Interleaved(interleaved) => Ok((
            source.class_id(),
            UploadSource::Borrowed(HostNumericTensorView {
                data: interleaved_view(&interleaved.values),
                shape: source.shape(),
                storage: GpuTensorStorage::ComplexInterleaved,
            }),
        )),
        MxArrayData::Logical(values) => Ok((
            MxClassId::Logical,
            UploadSource::Borrowed(HostNumericTensorView {
                data: HostNumericDataView::U8(values.as_slice()),
                shape: source.shape(),
                storage: GpuTensorStorage::Real,
            }),
        )),
        MxArrayData::Sparse(_) => {
            Err("sparse GPU arrays are not supported by this provider".into())
        }
        _ => Err("mxGPUArray creation requires a numeric or logical array".into()),
    }
}

fn host_numeric_view(values: &HostNumericBuffer) -> Result<HostNumericDataView<'_>, String> {
    if let Some(values) = values.as_f64_slice() {
        return Ok(HostNumericDataView::F64(values));
    }
    if let Some(values) = values.as_f32_slice() {
        return Ok(HostNumericDataView::F32(values));
    }
    match values.integer_storage() {
        Some(IntegerStorage::I8(values)) => Ok(HostNumericDataView::I8(values)),
        Some(IntegerStorage::I16(values)) => Ok(HostNumericDataView::I16(values)),
        Some(IntegerStorage::I32(values)) => Ok(HostNumericDataView::I32(values)),
        Some(IntegerStorage::I64(values)) => Ok(HostNumericDataView::I64(values)),
        Some(IntegerStorage::U8(values)) => Ok(HostNumericDataView::U8(values)),
        Some(IntegerStorage::U16(values)) => Ok(HostNumericDataView::U16(values)),
        Some(IntegerStorage::U32(values)) => Ok(HostNumericDataView::U32(values)),
        Some(IntegerStorage::U64(values)) => Ok(HostNumericDataView::U64(values)),
        None => Err("numeric host buffer has no typed storage".into()),
    }
}

fn interleaved_view(values: &MxInterleavedStorage) -> HostNumericDataView<'_> {
    macro_rules! lanes {
        ($values:expr, $variant:ident, $type:ty) => {{
            let length = $values.len().saturating_mul(2);
            // SAFETY: both complex element types have an explicit C layout of
            // exactly two adjacent values of the same primitive type.
            let data =
                unsafe { std::slice::from_raw_parts($values.as_ptr().cast::<$type>(), length) };
            HostNumericDataView::$variant(data)
        }};
    }
    match values {
        MxInterleavedStorage::F64(values) => lanes!(&values[..], F64, f64),
        MxInterleavedStorage::F32(values) => lanes!(&values[..], F32, f32),
        MxInterleavedStorage::I8(values) => lanes!(values, I8, i8),
        MxInterleavedStorage::I16(values) => lanes!(values, I16, i16),
        MxInterleavedStorage::I32(values) => lanes!(values, I32, i32),
        MxInterleavedStorage::I64(values) => lanes!(values, I64, i64),
        MxInterleavedStorage::U8(values) => lanes!(values, U8, u8),
        MxInterleavedStorage::U16(values) => lanes!(values, U16, u16),
        MxInterleavedStorage::U32(values) => lanes!(values, U32, u32),
        MxInterleavedStorage::U64(values) => lanes!(values, U64, u64),
    }
}

fn interleave_components(
    real: &HostNumericBuffer,
    imaginary: &HostNumericBuffer,
) -> Result<HostNumericDataOwned, String> {
    if real.numeric_dtype() != imaginary.numeric_dtype() || real.len() != imaginary.len() {
        return Err("complex GPU components must have matching class and length".into());
    }
    macro_rules! interleave {
        ($variant:ident, $real:expr, $imaginary:expr) => {{
            let mut output = Vec::with_capacity($real.len().saturating_mul(2));
            for (real, imaginary) in $real.iter().zip($imaginary) {
                output.push(*real);
                output.push(*imaginary);
            }
            HostNumericDataOwned::$variant(output)
        }};
    }
    match (host_numeric_view(real)?, host_numeric_view(imaginary)?) {
        (HostNumericDataView::F64(real), HostNumericDataView::F64(imag)) => {
            Ok(interleave!(F64, real, imag))
        }
        (HostNumericDataView::F32(real), HostNumericDataView::F32(imag)) => {
            Ok(interleave!(F32, real, imag))
        }
        (HostNumericDataView::I8(real), HostNumericDataView::I8(imag)) => {
            Ok(interleave!(I8, real, imag))
        }
        (HostNumericDataView::I16(real), HostNumericDataView::I16(imag)) => {
            Ok(interleave!(I16, real, imag))
        }
        (HostNumericDataView::I32(real), HostNumericDataView::I32(imag)) => {
            Ok(interleave!(I32, real, imag))
        }
        (HostNumericDataView::I64(real), HostNumericDataView::I64(imag)) => {
            Ok(interleave!(I64, real, imag))
        }
        (HostNumericDataView::U8(real), HostNumericDataView::U8(imag)) => {
            Ok(interleave!(U8, real, imag))
        }
        (HostNumericDataView::U16(real), HostNumericDataView::U16(imag)) => {
            Ok(interleave!(U16, real, imag))
        }
        (HostNumericDataView::U32(real), HostNumericDataView::U32(imag)) => {
            Ok(interleave!(U32, real, imag))
        }
        (HostNumericDataView::U64(real), HostNumericDataView::U64(imag)) => {
            Ok(interleave!(U64, real, imag))
        }
        _ => Err("complex GPU component classes do not match".into()),
    }
}

fn class_element_type(class_id: MxClassId) -> Result<NumericElementType, String> {
    Ok(match class_id {
        MxClassId::Double => NumericElementType::F64,
        MxClassId::Single => NumericElementType::F32,
        MxClassId::Int8 => NumericElementType::I8,
        MxClassId::Int16 => NumericElementType::I16,
        MxClassId::Int32 => NumericElementType::I32,
        MxClassId::Int64 => NumericElementType::I64,
        MxClassId::Logical | MxClassId::Uint8 => NumericElementType::U8,
        MxClassId::Uint16 => NumericElementType::U16,
        MxClassId::Uint32 => NumericElementType::U32,
        MxClassId::Uint64 => NumericElementType::U64,
        _ => return Err("GPU arrays require a numeric or logical class".into()),
    })
}

fn host_array_from_download(
    class_id: MxClassId,
    downloaded: HostNumericTensorOwned,
) -> Result<MxArray, String> {
    if class_id == MxClassId::Logical {
        let HostNumericDataOwned::U8(values) = downloaded.data else {
            return Err("logical GPU storage is not byte-addressable".into());
        };
        return MxArray::logical(values, downloaded.shape);
    }
    let storage = owned_numeric_storage(downloaded.data);
    match downloaded.storage {
        GpuTensorStorage::Real => MxArray::numeric(storage, downloaded.shape, None),
        GpuTensorStorage::ComplexInterleaved => {
            let values = interleaved_storage(storage)?;
            MxArray::interleaved(values, downloaded.shape)
        }
    }
}

fn owned_numeric_storage(data: HostNumericDataOwned) -> NumericStorage {
    match data {
        HostNumericDataOwned::F64(values) => NumericStorage::F64(values),
        HostNumericDataOwned::F32(values) => NumericStorage::F32(values),
        HostNumericDataOwned::I8(values) => NumericStorage::I8(values),
        HostNumericDataOwned::I16(values) => NumericStorage::I16(values),
        HostNumericDataOwned::I32(values) => NumericStorage::I32(values),
        HostNumericDataOwned::I64(values) => NumericStorage::I64(values),
        HostNumericDataOwned::U8(values) => NumericStorage::U8(values),
        HostNumericDataOwned::U16(values) => NumericStorage::U16(values),
        HostNumericDataOwned::U32(values) => NumericStorage::U32(values),
        HostNumericDataOwned::U64(values) => NumericStorage::U64(values),
    }
}

fn interleaved_storage(storage: NumericStorage) -> Result<MxInterleavedStorage, String> {
    macro_rules! pairs {
        ($values:expr, $variant:ident, $element:expr) => {{
            if $values.len() % 2 != 0 {
                return Err("complex GPU download has an incomplete element".into());
            }
            MxInterleavedStorage::$variant(
                $values
                    .chunks_exact(2)
                    .map(|pair| $element(pair[0], pair[1]))
                    .collect::<Vec<_>>()
                    .into(),
            )
        }};
    }
    Ok(match storage {
        NumericStorage::F64(values) => pairs!(values, F64, ComplexElement),
        NumericStorage::F32(values) => pairs!(values, F32, ComplexElement),
        NumericStorage::I8(values) => integer_pairs(values, MxInterleavedStorage::I8),
        NumericStorage::I16(values) => integer_pairs(values, MxInterleavedStorage::I16),
        NumericStorage::I32(values) => integer_pairs(values, MxInterleavedStorage::I32),
        NumericStorage::I64(values) => integer_pairs(values, MxInterleavedStorage::I64),
        NumericStorage::U8(values) => integer_pairs(values, MxInterleavedStorage::U8),
        NumericStorage::U16(values) => integer_pairs(values, MxInterleavedStorage::U16),
        NumericStorage::U32(values) => integer_pairs(values, MxInterleavedStorage::U32),
        NumericStorage::U64(values) => integer_pairs(values, MxInterleavedStorage::U64),
    })
}

fn integer_pairs<T: Copy>(
    values: Vec<T>,
    constructor: impl FnOnce(Vec<MxComplex<T>>) -> MxInterleavedStorage,
) -> MxInterleavedStorage {
    constructor(
        values
            .chunks_exact(2)
            .map(|pair| MxComplex {
                real: pair[0],
                imag: pair[1],
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, Once, OnceLock};

    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, GpuTensorDescriptor, GpuTensorHandle,
        HostNumericDataOwned, HostNumericTensorOwned, HostNumericTensorView, HostTensorOwned,
        HostTensorView, NativeDeviceBuffer, NativeDeviceContext, NativeDeviceContextGuard,
    };
    use runmat_value::{NumericStorage, Value};

    use super::*;
    use crate::{value_from_mx, MxApiMode};

    #[derive(Debug, Clone)]
    struct Allocation {
        lanes: Box<[f64]>,
        shape: Vec<usize>,
        storage: GpuTensorStorage,
    }

    #[derive(Debug)]
    struct FixtureCudaProvider {
        next: AtomicU64,
        allocations: Mutex<BTreeMap<u64, Allocation>>,
        frees: AtomicUsize,
        context_enters: AtomicUsize,
        context_leaves: Arc<AtomicUsize>,
    }

    impl FixtureCudaProvider {
        const DEVICE_ID: u32 = u32::MAX - 73;

        fn shared() -> &'static Self {
            static PROVIDER: OnceLock<&'static FixtureCudaProvider> = OnceLock::new();
            let provider = *PROVIDER.get_or_init(|| {
                Box::leak(Box::new(Self {
                    next: AtomicU64::new(1),
                    allocations: Mutex::new(BTreeMap::new()),
                    frees: AtomicUsize::new(0),
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

        fn test_gate() -> &'static Mutex<()> {
            static GATE: OnceLock<Mutex<()>> = OnceLock::new();
            GATE.get_or_init(|| Mutex::new(()))
        }

        fn reset(&self) {
            self.allocations.lock().unwrap().clear();
            self.frees.store(0, Ordering::Release);
            self.context_enters.store(0, Ordering::Release);
            self.context_leaves.store(0, Ordering::Release);
        }

        fn insert(
            &self,
            shape: Vec<usize>,
            storage: GpuTensorStorage,
            lanes: Box<[f64]>,
        ) -> GpuTensorHandle {
            let id = self.next.fetch_add(1, Ordering::Relaxed);
            self.allocations.lock().unwrap().insert(
                id,
                Allocation {
                    lanes,
                    shape: shape.clone(),
                    storage,
                },
            );
            GpuTensorHandle {
                shape,
                device_id: Self::DEVICE_ID,
                buffer_id: id,
                descriptor: GpuTensorDescriptor::numeric(NumericElementType::F64, storage),
            }
        }

        fn allocation(&self, handle: &GpuTensorHandle) -> anyhow::Result<Allocation> {
            if handle.device_id != Self::DEVICE_ID {
                anyhow::bail!("fixture handle belongs to another provider");
            }
            self.allocations
                .lock()
                .unwrap()
                .get(&handle.buffer_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("fixture allocation is unavailable"))
        }
    }

    impl AccelProvider for FixtureCudaProvider {
        fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
            self.upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(host.data),
                shape: host.shape,
                storage: GpuTensorStorage::Real,
            })
        }

        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            let result = self.allocation(handle).map(|allocation| HostTensorOwned {
                data: allocation.lanes.into_vec(),
                shape: allocation.shape,
                storage: allocation.storage,
            });
            Box::pin(async move { result })
        }

        fn upload_numeric(
            &self,
            host: &HostNumericTensorView<'_>,
        ) -> anyhow::Result<GpuTensorHandle> {
            host.validate()?;
            let HostNumericDataView::F64(values) = host.data else {
                anyhow::bail!("fixture provider accepts only double data");
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
            let result = self.allocation(handle).map(|allocation| {
                runmat_value::record_host_copy(
                    runmat_value::HostCopyReason::ProviderReadback,
                    allocation
                        .lanes
                        .len()
                        .saturating_mul(std::mem::size_of::<f64>()),
                );
                HostNumericTensorOwned {
                    data: HostNumericDataOwned::F64(allocation.lanes.into_vec()),
                    shape: allocation.shape,
                    storage: allocation.storage,
                }
            });
            Box::pin(async move { result })
        }

        fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
            if self
                .allocations
                .lock()
                .unwrap()
                .remove(&handle.buffer_id)
                .is_none()
            {
                anyhow::bail!("fixture allocation was already released");
            }
            self.frees.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }

        fn device_info(&self) -> String {
            "fixture CUDA provider".into()
        }

        fn device_id(&self) -> u32 {
            Self::DEVICE_ID
        }

        fn native_device_api(&self) -> Option<NativeDeviceApi> {
            Some(NativeDeviceApi::Cuda)
        }

        fn native_device_context(&self) -> anyhow::Result<NativeDeviceContext> {
            Ok(NativeDeviceContext {
                api: NativeDeviceApi::Cuda,
                provider_device_id: Self::DEVICE_ID,
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

        fn export_native_device_buffer(
            &self,
            handle: &GpuTensorHandle,
            _access: NativeDeviceAccess,
        ) -> anyhow::Result<NativeDeviceBuffer> {
            let allocations = self.allocations.lock().unwrap();
            let allocation = allocations
                .get(&handle.buffer_id)
                .ok_or_else(|| anyhow::anyhow!("fixture allocation is unavailable"))?;
            Ok(NativeDeviceBuffer {
                context: self.native_device_context()?,
                device_address: allocation.lanes.as_ptr() as u64,
                byte_length: allocation.lanes.len() * std::mem::size_of::<f64>(),
                descriptor: handle.descriptor,
            })
        }

        fn allocate_native_device_buffer(
            &self,
            shape: &[usize],
            element_type: NumericElementType,
            storage: GpuTensorStorage,
            initialization: NativeDeviceInitialization,
        ) -> anyhow::Result<GpuTensorHandle> {
            if element_type != NumericElementType::F64 {
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

    #[derive(Debug)]
    struct FixtureAmbientProvider {
        downloads: AtomicUsize,
    }

    impl FixtureAmbientProvider {
        const DEVICE_ID: u32 = u32::MAX - 74;

        fn shared() -> &'static Self {
            static PROVIDER: FixtureAmbientProvider = FixtureAmbientProvider {
                downloads: AtomicUsize::new(0),
            };
            static REGISTER: Once = Once::new();
            REGISTER.call_once(|| {
                // SAFETY: the static fixture owns a process-unique device id.
                unsafe { runmat_accelerate_api::register_device_provider(&PROVIDER) };
            });
            &PROVIDER
        }

        fn handle(&self) -> GpuTensorHandle {
            GpuTensorHandle {
                shape: vec![1, 2],
                device_id: Self::DEVICE_ID,
                buffer_id: 1,
                descriptor: GpuTensorDescriptor::numeric(
                    NumericElementType::F64,
                    GpuTensorStorage::Real,
                ),
            }
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

        fn free(&self, _handle: &GpuTensorHandle) -> anyhow::Result<()> {
            anyhow::bail!("borrowed ambient fixture handles cannot be freed")
        }

        fn device_info(&self) -> String {
            "fixture non-CUDA provider".into()
        }

        fn device_id(&self) -> u32 {
            Self::DEVICE_ID
        }
    }

    #[test]
    fn gpu_arrays_preserve_alias_identity_and_transfer_owned_lifetimes_once() {
        let _gate = FixtureCudaProvider::test_gate().lock().unwrap();
        let provider = FixtureCudaProvider::shared();
        provider.reset();
        let _selection = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

        {
            let guard = provider.enter_native_device_context().unwrap();
            assert_eq!(provider.context_enters.load(Ordering::Acquire), 1);
            drop(guard);
            assert_eq!(provider.context_leaves.load(Ordering::Acquire), 1);
        }

        let mut api = MxApi::new(MxApiMode::InterleavedComplex);
        let host = api.allocate(
            MxArray::numeric(NumericStorage::F64(vec![1.0, 2.0]), vec![1, 2], None).unwrap(),
        );
        let gpu = api.gpu_create_from_array(host, false).unwrap();
        let alias = api.gpu_create_from_array(gpu, false).unwrap();
        let duplicate = api.gpu_create_from_array(gpu, true).unwrap();
        assert!(api.gpu_is_same(gpu, alias).unwrap());
        assert!(!api.gpu_is_same(gpu, duplicate).unwrap());

        let address = api.gpu_data(gpu, true).unwrap();
        // SAFETY: the fixture provider deliberately exposes host-addressable
        // storage to exercise the native pointer and ownership contract.
        unsafe { *(address as *mut f64) = 9.0 };
        let gathered = api.gpu_to_host(alias).unwrap();
        let Value::Tensor(gathered) = value_from_mx(api.arena().get(gathered).unwrap()).unwrap()
        else {
            panic!("gathered GPU array must be a dense tensor");
        };
        assert_eq!(gathered.materialize_f64(), vec![9.0, 2.0]);

        api.arena_mut().destroy(duplicate).unwrap();
        assert_eq!(provider.frees.load(Ordering::Acquire), 1);
        api.arena_mut().destroy(gpu).unwrap();
        assert_eq!(provider.frees.load(Ordering::Acquire), 1);

        let Value::GpuTensor(published) = value_from_mx(api.arena().get(alias).unwrap()).unwrap()
        else {
            panic!("GPU output must retain its provider handle");
        };
        api.arena_mut().destroy(alias).unwrap();
        assert_eq!(provider.frees.load(Ordering::Acquire), 1);
        assert_eq!(published.shape, vec![1, 2]);
        provider.free(&published).unwrap();
        assert_eq!(provider.frees.load(Ordering::Acquire), 2);
    }

    #[test]
    fn non_cuda_gpu_input_transfers_explicitly_to_the_cuda_provider() {
        let _gate = FixtureCudaProvider::test_gate().lock().unwrap();
        let cuda = FixtureCudaProvider::shared();
        let ambient = FixtureAmbientProvider::shared();
        cuda.reset();
        ambient.downloads.store(0, Ordering::Release);
        let _selection = runmat_accelerate_api::ThreadProviderGuard::set(Some(ambient));
        let readback_before =
            runmat_value::host_copy_metrics(runmat_value::HostCopyReason::ProviderReadback);
        let upload_before =
            runmat_value::host_copy_metrics(runmat_value::HostCopyReason::ProviderUpload);

        let mut api = MxApi::new(MxApiMode::InterleavedComplex);
        let source =
            api.allocate(MxArray::gpu_borrowed(MxClassId::Double, ambient.handle()).unwrap());
        let transferred = api.gpu_create_from_array(source, false).unwrap();
        let MxArrayData::Gpu(transferred_gpu) = api.arena().get(transferred).unwrap().data() else {
            panic!("transferred array must remain provider-resident");
        };
        assert_eq!(transferred_gpu.lease.handle().device_id, cuda.device_id());
        assert_ne!(
            transferred_gpu.lease.handle().device_id,
            ambient.device_id()
        );
        assert_eq!(ambient.downloads.load(Ordering::Acquire), 1);

        let gathered = api.gpu_to_host(transferred).unwrap();
        let Value::Tensor(gathered) = value_from_mx(api.arena().get(gathered).unwrap()).unwrap()
        else {
            panic!("transferred array must gather as a dense tensor");
        };
        assert_eq!(gathered.materialize_f64(), vec![11.0, 13.0]);
        let readback_after =
            runmat_value::host_copy_metrics(runmat_value::HostCopyReason::ProviderReadback);
        let upload_after =
            runmat_value::host_copy_metrics(runmat_value::HostCopyReason::ProviderUpload);
        assert_eq!(readback_after.operations - readback_before.operations, 2);
        assert_eq!(readback_after.bytes - readback_before.bytes, 32);
        assert_eq!(upload_after.operations - upload_before.operations, 1);
        assert_eq!(upload_after.bytes - upload_before.bytes, 16);

        api.arena_mut().destroy(source).unwrap();
        api.arena_mut().destroy(transferred).unwrap();
        assert_eq!(cuda.frees.load(Ordering::Acquire), 1);
    }

    #[test]
    fn complex_gpu_components_stay_on_the_provider_and_round_trip_exactly() {
        let _gate = FixtureCudaProvider::test_gate().lock().unwrap();
        let provider = FixtureCudaProvider::shared();
        provider.reset();
        let _selection = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

        let mut api = MxApi::new(MxApiMode::InterleavedComplex);
        let real = api
            .gpu_create(MxClassId::Double, vec![2, 1], false, true)
            .unwrap();
        let imaginary = api
            .gpu_create(MxClassId::Double, vec![2, 1], false, false)
            .unwrap();
        let real_address = api.gpu_data(real, true).unwrap() as *mut f64;
        let imaginary_address = api.gpu_data(imaginary, true).unwrap() as *mut f64;
        // SAFETY: fixture allocations are aligned, host-addressable f64 lanes.
        unsafe {
            real_address.write(1.5);
            real_address.add(1).write(-2.0);
            imaginary_address.write(3.0);
            imaginary_address.add(1).write(4.5);
        }

        let complex = api.gpu_create_complex(real, imaginary).unwrap();
        assert!(api.arena().get(complex).unwrap().is_complex());
        let copied_real = api
            .gpu_copy_component(complex, NativeDeviceComponent::Real)
            .unwrap();
        let copied_imaginary = api
            .gpu_copy_component(complex, NativeDeviceComponent::Imaginary)
            .unwrap();
        let gathered_real = api.gpu_to_host(copied_real).unwrap();
        let gathered_imaginary = api.gpu_to_host(copied_imaginary).unwrap();
        let Value::Tensor(gathered_real) =
            value_from_mx(api.arena().get(gathered_real).unwrap()).unwrap()
        else {
            panic!("real component must gather as a tensor");
        };
        let Value::Tensor(gathered_imaginary) =
            value_from_mx(api.arena().get(gathered_imaginary).unwrap()).unwrap()
        else {
            panic!("imaginary component must gather as a tensor");
        };
        assert_eq!(gathered_real.materialize_f64(), vec![1.5, -2.0]);
        assert_eq!(gathered_imaginary.materialize_f64(), vec![3.0, 4.5]);

        for pointer in [real, imaginary, complex, copied_real, copied_imaginary] {
            api.arena_mut().destroy(pointer).unwrap();
        }
        assert_eq!(provider.frees.load(Ordering::Acquire), 5);
        assert!(provider.allocations.lock().unwrap().is_empty());
    }

    #[test]
    fn compiled_gpu_matrix_fixture_uses_provider_memory_and_publishes_without_copy() {
        let _gate = FixtureCudaProvider::test_gate().lock().unwrap();
        let provider = FixtureCudaProvider::shared();
        provider.reset();
        let _selection = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("gpu_gateway.c");
        std::fs::write(
            &source,
            r#"
#include "mex.h"
#include "gpu/mxGPUArray.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                 const mxArray *prhs[]) {
    if (nrhs != 1 || nlhs != 2) {
        mexErrMsgIdAndTxt("RunMat:test:arity",
                          "expected one input and two outputs");
    }
    if (mxInitGPU() != MX_GPU_SUCCESS) {
        mexErrMsgIdAndTxt("RunMat:test:gpu", "could not enter GPU context");
    }
    const mxGPUArray *input = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetDataReadOnly(input) == NULL) {
        mexErrMsgIdAndTxt("RunMat:test:gpu", "input has no device data");
    }
    mxGPUArray *copy = mxGPUCopyGPUArray(input);
    if (mxGPUIsSame(input, copy)) {
        mexErrMsgIdAndTxt("RunMat:test:gpu", "copy retained input identity");
    }
    double *data = (double *)mxGPUGetData(copy);
    data[0] += 5.0;
    plhs[0] = mxGPUCreateMxArrayOnGPU(copy);
    plhs[1] = mxGPUCreateMxArrayOnCPU(copy);
    mxGPUDestroyGPUArray(input);
    mxGPUDestroyGPUArray(copy);
}
"#,
        )
        .unwrap();

        let artifact = crate::MexBuild::new(&source, directory.path())
            .api(crate::MexApi::R2018a)
            .compile()
            .unwrap();
        let module = crate::MexModule::load(&artifact.module).unwrap();
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&[2.0, 4.0]),
                shape: &[1, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let invocation = module
            .invoke(&[Value::GpuTensor(input.clone())], 2, module.api_mode())
            .unwrap();
        let Value::GpuTensor(output) = &invocation.outputs[0] else {
            panic!("first fixture output must stay provider-resident");
        };
        assert_ne!(output.buffer_id, input.buffer_id);
        let Value::Tensor(gathered) = &invocation.outputs[1] else {
            panic!("second fixture output must be gathered host data");
        };
        assert_eq!(gathered.materialize_f64(), vec![7.0, 4.0]);
        assert_eq!(provider.context_enters.load(Ordering::Acquire), 1);
        assert_eq!(provider.context_leaves.load(Ordering::Acquire), 1);
        assert_eq!(provider.allocations.lock().unwrap().len(), 2);

        provider.free(&input).unwrap();
        provider.free(output).unwrap();
        assert_eq!(provider.frees.load(Ordering::Acquire), 2);
    }
}
