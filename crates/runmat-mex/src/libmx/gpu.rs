use runmat_accelerate_api::{
    GpuTensorStorage, HostLogicalTensorView, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostNumericTensorView, NativeDeviceAccess, NativeDeviceApi,
    NativeDeviceClass, NativeDeviceComponent, NativeDeviceInitialization, NumericElementType,
};
use runmat_value::{ComplexElement, HostNumericBuffer, IntegerStorage, NumericStorage};

use crate::mxarray::{MxArrayData, MxComplex, MxGpuLease, MxInterleavedStorage, MxNumeric};
use crate::{MxApi, MxArray, MxClassId};

impl MxApi {
    pub fn gpu_create_from_array(
        &mut self,
        source: *const MxArray,
        independent_copy: bool,
    ) -> Result<*mut MxArray, crate::MexGpuError> {
        let source = self.arena.get(source).map_err(|error| error.to_string())?;
        if let MxArrayData::Gpu(gpu) = source.data() {
            let target = cuda_provider()?;
            let owner = runmat_accelerate_api::provider_for_handle(gpu.lease.handle())
                .ok_or_else(|| "GPU array provider is unavailable".to_string())?;
            if owner.device_id() == target.device_id() {
                let lease = if independent_copy {
                    gpu.lease
                        .duplicate()
                        .map_err(crate::MexGpuError::Provider)?
                } else {
                    gpu.lease.clone()
                };
                let value = MxArray::gpu_owned(gpu.class_id, lease)?;
                return Ok(self.arena.allocate(value));
            }
            let downloaded = pollster::block_on(owner.download_numeric(gpu.lease.handle()))
                .map_err(|error| format!("could not transfer GPU array to CUDA: {error}"))?;
            let handle = if gpu.class_id == MxClassId::Logical {
                let logical = host_array_from_download(MxClassId::Logical, downloaded)?;
                let (_, UploadSource::Logical(view)) = host_upload(&logical)? else {
                    unreachable!("logical host array has a logical upload view")
                };
                runmat_accelerate_api::upload_logical_protecting(
                    target,
                    &view,
                    &[gpu.lease.handle()],
                )
            } else {
                runmat_accelerate_api::upload_numeric_protecting(
                    target,
                    &downloaded.as_view(),
                    &[gpu.lease.handle()],
                )
            }
            .map_err(|source| crate::MexGpuError::Upload {
                context: "could not upload GPU array to CUDA",
                source,
            })?;
            let value = MxArray::gpu_owned(gpu.class_id, MxGpuLease::owned(handle))?;
            return Ok(self.arena.allocate(value));
        }

        let (class_id, upload) = host_upload(source)?;
        let provider = cuda_provider()?;
        let handle = match &upload {
            UploadSource::Borrowed(view) => {
                runmat_accelerate_api::upload_numeric_protecting(provider, view, &[])
            }
            UploadSource::Owned(owned) => {
                runmat_accelerate_api::upload_numeric_protecting(provider, &owned.as_view(), &[])
            }
            UploadSource::Logical(view) => {
                runmat_accelerate_api::upload_logical_protecting(provider, view, &[])
            }
        }
        .map_err(|source| crate::MexGpuError::Upload {
            context: "could not upload array to CUDA",
            source,
        })?;
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
        let class = native_class(class_id)?;
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
        class
            .validate_storage(storage)
            .map_err(|error| error.to_string())?;
        runmat_accelerate_api::native_buffer_byte_length(&shape, class.element_type(), storage)
            .map_err(|error| error.to_string())?;
        let provider = cuda_provider()?;
        let handle = provider
            .allocate_native_device_buffer(&shape, class, storage, initialization)
            .map_err(|error| format!("could not allocate CUDA array: {error}"))?;
        runmat_accelerate_api::validate_native_device_result(
            &handle,
            provider.device_id(),
            &shape,
            class,
            storage,
        )
        .map_err(|error| error.to_string())?;
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
        Ok(left
            .lease
            .handle()
            .shares_allocation_with(right.lease.handle()))
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
            .descriptor()
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
                        native_class(gpu.class_id)?,
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
            if gpu.lease.handle().descriptor().storage != Some(GpuTensorStorage::Real) {
                return Err("complex GPU components must be real arrays".to_string());
            }
            Ok((value.shape().to_vec(), gpu.class_id, gpu.lease.clone()))
        };
        let (real_shape, real_class, real_lease) = component(real)?;
        let (imag_shape, imag_class, imag_lease) = component(imaginary)?;
        if real_shape != imag_shape || real_class != imag_class {
            return Err("complex GPU components must have matching shape and class".into());
        }
        native_class(real_class)?
            .validate_storage(GpuTensorStorage::ComplexInterleaved)
            .map_err(|error| error.to_string())?;
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
    Logical(HostLogicalTensorView<'a>),
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
            UploadSource::Logical(HostLogicalTensorView {
                data: values.as_slice(),
                shape: source.shape(),
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

fn native_class(class_id: MxClassId) -> Result<NativeDeviceClass, String> {
    if class_id == MxClassId::Logical {
        Ok(NativeDeviceClass::Logical)
    } else {
        class_element_type(class_id).map(NativeDeviceClass::Numeric)
    }
}

fn host_array_from_download(
    class_id: MxClassId,
    downloaded: HostNumericTensorOwned,
) -> Result<MxArray, String> {
    if class_id == MxClassId::Logical {
        if downloaded.storage != GpuTensorStorage::Real {
            return Err("logical GPU storage must be real".into());
        }
        let values = match downloaded.data {
            HostNumericDataOwned::U8(values) => values,
            HostNumericDataOwned::F32(values) => values
                .into_iter()
                .map(|value| u8::from(value != 0.0))
                .collect(),
            HostNumericDataOwned::F64(values) => values
                .into_iter()
                .map(|value| u8::from(value != 0.0))
                .collect(),
            _ => return Err("logical GPU storage must use U8, F32, or F64 encoding".into()),
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
#[path = "gpu/typed_inputs_tests.rs"]
mod typed_inputs_tests;

#[cfg(test)]
#[path = "gpu/tests.rs"]
mod tests;
