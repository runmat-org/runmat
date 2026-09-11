use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::thread::{self, ThreadId};

use runmat_value::{
    record_host_copy, CellArray, CharArray, ComplexElement, ComplexStorage, ComplexTensor,
    HandleRef, HostComplexBuffer, HostCopyReason, HostIndexBuffer, HostNumericBuffer,
    IntegerComplexStorage, IntegerStorage, LogicalArray, NumericScalar, NumericStorage,
    ObjectArray, ObjectInstance, SparseTensor, StructArray, StructValue, Tensor, Value,
};

use crate::mxarray::{
    MxApiMode, MxArray, MxArrayData, MxBoundaryInterface, MxClassId, MxHandleToken,
    MxInterleavedStorage, MxNumeric, MxSparse, MxSparseValues,
};

#[derive(Debug)]
struct MxHandleEntry {
    generation: u64,
    value: HandleRef,
}

/// Invocation-scoped origin-thread authority for garbage-collected handles.
///
/// Native lanes carry only [`MxHandleToken`] values. They cannot resolve or
/// dereference a token; conversion back to a RunMat handle is permitted only
/// through this context on the thread that created it.
#[derive(Debug)]
pub struct MxValueContext {
    origin_thread: ThreadId,
    next_resource: RefCell<u64>,
    handles: RefCell<BTreeMap<u64, MxHandleEntry>>,
}

impl MxValueContext {
    pub fn new() -> Self {
        Self {
            origin_thread: thread::current().id(),
            next_resource: RefCell::new(1),
            handles: RefCell::new(BTreeMap::new()),
        }
    }

    pub(crate) fn register_handle(
        &self,
        value: &HandleRef,
    ) -> Result<MxHandleToken, MxConversionError> {
        self.require_origin_thread()?;
        let mut next_resource = self.next_resource.borrow_mut();
        let resource = *next_resource;
        *next_resource = next_resource
            .checked_add(1)
            .ok_or_else(|| MxConversionError::new("MEX handle identity exhausted"))?;
        let generation = 1;
        self.handles.borrow_mut().insert(
            resource,
            MxHandleEntry {
                generation,
                value: value.clone(),
            },
        );
        Ok(MxHandleToken {
            resource,
            generation,
            class_name: value.class_name.clone(),
        })
    }

    pub(crate) fn resolve_handle(
        &self,
        token: &MxHandleToken,
    ) -> Result<HandleRef, MxConversionError> {
        self.require_origin_thread()?;
        self.handles
            .borrow()
            .get(&token.resource)
            .filter(|entry| entry.generation == token.generation)
            .map(|entry| entry.value.clone())
            .ok_or_else(|| MxConversionError::new("stale MEX handle token"))
    }

    /// Convert a runtime value at its originating task boundary.
    #[doc(hidden)]
    pub fn encode(
        &self,
        value: &Value,
        mode: MxApiMode,
        interface: MxBoundaryInterface,
    ) -> Result<MxArray, MxConversionError> {
        validate_portable_value(value)?;
        value_to_mx_for_interface_in_context(value, mode, interface, Some(self))
    }

    /// Restore a runtime value at its originating task boundary.
    #[doc(hidden)]
    pub fn decode(&self, value: &MxArray) -> Result<Value, MxConversionError> {
        value_from_mx_in_context(value, Some(self))
    }

    fn require_origin_thread(&self) -> Result<(), MxConversionError> {
        if thread::current().id() == self.origin_thread {
            Ok(())
        } else {
            Err(MxConversionError::new(
                "MEX handle tokens can only be resolved on their originating runtime thread",
            ))
        }
    }
}

impl Default for MxValueContext {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MxConversionError {
    pub message: String,
}

impl MxConversionError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for MxConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for MxConversionError {}

pub fn value_to_mx(value: &Value, mode: MxApiMode) -> Result<MxArray, MxConversionError> {
    validate_portable_value(value)?;
    value_to_mx_for_interface_in_context(value, mode, MxBoundaryInterface::CMatrix, None)
}

fn validate_portable_value(value: &Value) -> Result<(), MxConversionError> {
    runmat_value::validate_no_transient_sequence(value).map_err(|_| {
        MxConversionError::new(
            "TransientSequenceNotPortable: transient output sequences cannot cross the MEX value boundary",
        )
    })
}

pub(crate) fn value_to_mx_for_interface_in_context(
    value: &Value,
    mode: MxApiMode,
    interface: MxBoundaryInterface,
    context: Option<&MxValueContext>,
) -> Result<MxArray, MxConversionError> {
    match value {
        Value::Num(value) => numeric_to_mx(NumericStorage::F64(vec![*value]), vec![1, 1]),
        Value::Int(value) => numeric_to_mx(
            NumericStorage::from_integer_storage(IntegerStorage::from_scalar(value.clone())),
            vec![1, 1],
        ),
        Value::Complex(real, imag) => complex_to_mx(
            NumericStorage::F64(vec![*real]),
            NumericStorage::F64(vec![*imag]),
            vec![1, 1],
            mode,
        ),
        Value::Bool(value) => {
            MxArray::logical(vec![u8::from(*value)], vec![1, 1]).map_err(MxConversionError::new)
        }
        Value::LogicalArray(value) => {
            MxArray::logical_buffer(value.data.clone(), value.shape.clone())
                .map_err(MxConversionError::new)
        }
        Value::String(value) if interface == MxBoundaryInterface::CxxData => {
            record_host_copy(HostCopyReason::CharacterEncoding, value.len());
            MxArray::string(vec![value.clone()], vec![1, 1]).map_err(MxConversionError::new)
        }
        Value::String(value) => string_to_mx(value),
        Value::StringArray(value) if interface == MxBoundaryInterface::CxxData => {
            record_host_copy(
                HostCopyReason::CharacterEncoding,
                value.data.iter().map(String::len).sum(),
            );
            MxArray::string(value.data.clone(), value.shape.clone()).map_err(MxConversionError::new)
        }
        Value::CharArray(value) => char_to_mx(value),
        Value::Tensor(value) => {
            MxArray::numeric_buffer(value.host_buffer().clone(), value.shape.clone(), None)
                .map_err(MxConversionError::new)
        }
        Value::ComplexTensor(value) => complex_tensor_to_mx(value, mode),
        Value::SparseTensor(value) => sparse_to_mx(value, mode),
        Value::Cell(value) => cell_to_mx(value, mode, interface, context),
        Value::Struct(value) => struct_to_mx(value, mode, interface, context),
        Value::StructArray(value) => struct_array_to_mx(value, mode, interface, context),
        Value::Object(value) => object_to_mx(
            &value.class_name,
            &[value],
            vec![1, 1],
            mode,
            interface,
            context,
        ),
        Value::ObjectArray(value) => {
            let objects = value
                .data()
                .iter()
                .map(|element| match element {
                    Value::Object(object) => Ok(object),
                    Value::HandleObject(_) => Err(MxConversionError::new(
                        "handle objects cannot cross the C Matrix API by value",
                    )),
                    _ => unreachable!("ObjectArray validates its elements"),
                })
                .collect::<Result<Vec<_>, _>>()?;
            object_to_mx(
                value.class_name(),
                &objects,
                value.shape().to_vec(),
                mode,
                interface,
                context,
            )
        }
        Value::HandleObject(value) => context
            .ok_or_else(|| {
                MxConversionError::new(
                    "handle-object conversion requires an invocation value context",
                )
            })?
            .register_handle(value)
            .map(MxArray::handle),
        Value::GpuTensor(handle) => {
            let class_id = gpu_class_id(handle)?;
            MxArray::gpu_borrowed(class_id, handle.clone()).map_err(MxConversionError::new)
        }
        other => Err(MxConversionError::new(format!(
            "{} values do not have a C Matrix API representation",
            value_kind(other)
        ))),
    }
}

pub fn value_from_mx(value: &MxArray) -> Result<Value, MxConversionError> {
    value_from_mx_in_context(value, None)
}

pub(crate) fn value_from_mx_in_context(
    value: &MxArray,
    context: Option<&MxValueContext>,
) -> Result<Value, MxConversionError> {
    match value.data() {
        MxArrayData::Numeric(numeric) => numeric_from_mx(numeric, value.shape()),
        MxArrayData::Interleaved(interleaved) => match &interleaved.values {
            MxInterleavedStorage::F64(values) if values.len() == 1 => {
                Ok(Value::Complex(values[0].0, values[0].1))
            }
            MxInterleavedStorage::F64(values) => ComplexTensor::from_complex_storage(
                ComplexStorage::F64(values.clone()),
                value.shape().to_vec(),
            )
            .map(Value::ComplexTensor)
            .map_err(MxConversionError::new),
            MxInterleavedStorage::F32(values) => ComplexTensor::from_complex_storage(
                ComplexStorage::F32(values.clone()),
                value.shape().to_vec(),
            )
            .map(Value::ComplexTensor)
            .map_err(MxConversionError::new),
            _ => {
                let (real, imag) = interleaved.values.components();
                complex_from_components(real, imag, value.shape())
            }
        },
        MxArrayData::Logical(values) => {
            if values.len() == 1 {
                Ok(Value::Bool(values[0] != 0))
            } else {
                LogicalArray::from_host_buffer(values.clone(), value.shape().to_vec())
                    .map(Value::LogicalArray)
                    .map_err(MxConversionError::new)
            }
        }
        MxArrayData::Char(values) => char_from_mx(values, value.shape()),
        MxArrayData::String(values) => {
            record_host_copy(
                HostCopyReason::CharacterEncoding,
                values.iter().map(String::len).sum(),
            );
            if values.len() == 1 {
                Ok(Value::String(values[0].clone()))
            } else {
                runmat_value::StringArray::new(values.clone(), value.shape().to_vec())
                    .map(Value::StringArray)
                    .map_err(MxConversionError::new)
            }
        }
        MxArrayData::Cell(values) => cell_from_mx(values, value.shape(), context),
        MxArrayData::Struct { fields, values } => {
            struct_from_mx(fields, values, value.shape(), context)
        }
        MxArrayData::Object {
            class_name,
            properties,
            values,
        } => object_from_mx(class_name, properties, values, value.shape(), context),
        MxArrayData::Handle(value) => context
            .ok_or_else(|| {
                MxConversionError::new(
                    "handle-object conversion requires an invocation value context",
                )
            })?
            .resolve_handle(value)
            .map(Value::HandleObject),
        MxArrayData::Sparse(value) => sparse_from_mx(value),
        MxArrayData::Gpu(gpu) => {
            let handle = gpu.lease.publish(value.shape());
            runmat_accelerate_api::set_handle_class_identity(&handle, mx_class_name(gpu.class_id));
            Ok(Value::GpuTensor(handle))
        }
    }
}

fn gpu_class_id(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> Result<MxClassId, MxConversionError> {
    if runmat_accelerate_api::handle_is_logical(handle) {
        return Ok(MxClassId::Logical);
    }
    let element = handle.descriptor.element_type.ok_or_else(|| {
        MxConversionError::new("GPU-resident value is missing its physical element type")
    })?;
    Ok(match element {
        runmat_accelerate_api::NumericElementType::F64 => MxClassId::Double,
        runmat_accelerate_api::NumericElementType::F32 => MxClassId::Single,
        runmat_accelerate_api::NumericElementType::I8 => MxClassId::Int8,
        runmat_accelerate_api::NumericElementType::I16 => MxClassId::Int16,
        runmat_accelerate_api::NumericElementType::I32 => MxClassId::Int32,
        runmat_accelerate_api::NumericElementType::I64 => MxClassId::Int64,
        runmat_accelerate_api::NumericElementType::U8 => MxClassId::Uint8,
        runmat_accelerate_api::NumericElementType::U16 => MxClassId::Uint16,
        runmat_accelerate_api::NumericElementType::U32 => MxClassId::Uint32,
        runmat_accelerate_api::NumericElementType::U64 => MxClassId::Uint64,
    })
}

fn mx_class_name(class_id: MxClassId) -> &'static str {
    match class_id {
        MxClassId::Logical => "logical",
        MxClassId::Double => "double",
        MxClassId::Single => "single",
        MxClassId::Int8 => "int8",
        MxClassId::Uint8 => "uint8",
        MxClassId::Int16 => "int16",
        MxClassId::Uint16 => "uint16",
        MxClassId::Int32 => "int32",
        MxClassId::Uint32 => "uint32",
        MxClassId::Int64 => "int64",
        MxClassId::Uint64 => "uint64",
        _ => "gpuArray",
    }
}

fn numeric_to_mx(storage: NumericStorage, shape: Vec<usize>) -> Result<MxArray, MxConversionError> {
    MxArray::numeric(storage, shape, None).map_err(MxConversionError::new)
}

fn complex_to_mx(
    real: NumericStorage,
    imag: NumericStorage,
    shape: Vec<usize>,
    mode: MxApiMode,
) -> Result<MxArray, MxConversionError> {
    match mode {
        MxApiMode::SeparateComplex => {
            MxArray::numeric(real, shape, Some(imag)).map_err(MxConversionError::new)
        }
        MxApiMode::InterleavedComplex => MxInterleavedStorage::from_components(&real, &imag)
            .and_then(|values| MxArray::interleaved(values, shape))
            .map_err(MxConversionError::new),
    }
}

fn complex_tensor_to_mx(
    value: &ComplexTensor,
    mode: MxApiMode,
) -> Result<MxArray, MxConversionError> {
    if mode == MxApiMode::InterleavedComplex {
        let values = match value.complex_storage() {
            ComplexStorage::F64(values) => MxInterleavedStorage::F64(values.clone()),
            ComplexStorage::F32(values) => MxInterleavedStorage::F32(values.clone()),
            ComplexStorage::Integer(_) => {
                let (real, imag) = complex_components(value)?;
                return complex_to_mx(real, imag, value.shape.clone(), mode);
            }
        };
        return MxArray::interleaved(values, value.shape.clone()).map_err(MxConversionError::new);
    }

    let (real, imag) = complex_components(value)?;
    complex_to_mx(real, imag, value.shape.clone(), mode)
}

fn complex_components(
    value: &ComplexTensor,
) -> Result<(NumericStorage, NumericStorage), MxConversionError> {
    match value.complex_storage() {
        ComplexStorage::F64(values) => Ok((
            NumericStorage::F64(values.iter().map(|value| value.0).collect()),
            NumericStorage::F64(values.iter().map(|value| value.1).collect()),
        )),
        ComplexStorage::F32(values) => Ok((
            NumericStorage::F32(values.iter().map(|value| value.0).collect()),
            NumericStorage::F32(values.iter().map(|value| value.1).collect()),
        )),
        ComplexStorage::Integer(values) => Ok((
            NumericStorage::from_integer_storage(values.real.clone()),
            NumericStorage::from_integer_storage(values.imag.clone()),
        )),
    }
}

fn string_to_mx(value: &str) -> Result<MxArray, MxConversionError> {
    let encoded = value.encode_utf16().collect::<Vec<_>>();
    let length = encoded.len();
    record_host_copy(
        HostCopyReason::CharacterEncoding,
        length.saturating_mul(std::mem::size_of::<u16>()),
    );
    MxArray::character(encoded, vec![1, length]).map_err(MxConversionError::new)
}

fn char_to_mx(value: &CharArray) -> Result<MxArray, MxConversionError> {
    let mut encoded = Vec::with_capacity(value.data.len());
    for character in value.to_column_major() {
        let codepoint = u32::from(character);
        let unit = u16::try_from(codepoint).map_err(|_| {
            MxConversionError::new(
                "RunMat character arrays containing non-BMP scalars cannot preserve their shape in the UTF-16 C Matrix API",
            )
        })?;
        encoded.push(unit);
    }
    record_host_copy(
        HostCopyReason::CharacterEncoding,
        encoded.len().saturating_mul(std::mem::size_of::<u16>()),
    );
    MxArray::character(encoded, value.shape().to_vec()).map_err(MxConversionError::new)
}

fn sparse_to_mx(value: &SparseTensor, mode: MxApiMode) -> Result<MxArray, MxConversionError> {
    let values = if value.is_logical() {
        record_host_copy(HostCopyReason::SparseLayoutConversion, value.nnz());
        MxSparseValues::Logical(vec![1; value.nnz()].into())
    } else if let Some(values) = value.complex_host_buffer() {
        match mode {
            MxApiMode::InterleavedComplex => MxSparseValues::InterleavedComplex(values.clone()),
            MxApiMode::SeparateComplex => {
                let mut real = Vec::with_capacity(values.len());
                let mut imaginary = Vec::with_capacity(values.len());
                for ComplexElement(real_value, imaginary_value) in values.iter().copied() {
                    real.push(real_value);
                    imaginary.push(imaginary_value);
                }
                record_host_copy(
                    HostCopyReason::SparseLayoutConversion,
                    values
                        .len()
                        .saturating_mul(std::mem::size_of::<ComplexElement<f64>>()),
                );
                MxSparseValues::SeparateComplex {
                    real: HostNumericBuffer::from_numeric_storage(NumericStorage::F64(real)),
                    imaginary: HostNumericBuffer::from_numeric_storage(NumericStorage::F64(
                        imaginary,
                    )),
                }
            }
        }
    } else if value.complex_f32_host_buffer().is_some() {
        return Err(MxConversionError::new(
            "the current C Matrix boundary does not represent complex-single sparse storage",
        ));
    } else {
        MxSparseValues::Numeric(
            value
                .numeric_host_buffer()
                .ok_or_else(|| MxConversionError::new("sparse value has no numeric class"))?
                .clone(),
        )
    };
    MxArray::sparse(MxSparse {
        rows: value.rows,
        cols: value.cols,
        col_ptrs: value.col_ptrs.clone(),
        row_indices: value.row_indices.clone(),
        values,
        nzmax: value.nnz(),
    })
    .map_err(MxConversionError::new)
}

fn cell_to_mx(
    value: &CellArray,
    mode: MxApiMode,
    interface: MxBoundaryInterface,
    context: Option<&MxValueContext>,
) -> Result<MxArray, MxConversionError> {
    let column_major = value.to_column_major();
    let values = column_major
        .iter()
        .map(|value| {
            value_to_mx_for_interface_in_context(value, mode, interface, context)
                .map(Box::new)
                .map(Some)
        })
        .collect::<Result<Vec<_>, _>>()?;
    MxArray::cell(values, value.shape.clone()).map_err(MxConversionError::new)
}

fn struct_array_to_mx(
    value: &StructArray,
    mode: MxApiMode,
    interface: MxBoundaryInterface,
    context: Option<&MxValueContext>,
) -> Result<MxArray, MxConversionError> {
    let fields = value.field_names().cloned().collect::<Vec<_>>();
    let capacity = fields.len().checked_mul(value.len()).ok_or_else(|| {
        MxConversionError::new("structure array field storage exceeds platform limits")
    })?;
    let mut values = Vec::with_capacity(capacity);
    for field in &fields {
        let column = value
            .field_values(field)
            .ok_or_else(|| MxConversionError::new("structure array field is missing"))?;
        for field_value in column {
            values.push(
                value_to_mx_for_interface_in_context(field_value, mode, interface, context)
                    .map(Box::new)
                    .map(Some)?,
            );
        }
    }
    MxArray::structure(fields, values, value.shape().to_vec()).map_err(MxConversionError::new)
}

fn struct_to_mx(
    value: &StructValue,
    mode: MxApiMode,
    interface: MxBoundaryInterface,
    context: Option<&MxValueContext>,
) -> Result<MxArray, MxConversionError> {
    let fields = value.field_names().cloned().collect::<Vec<_>>();
    let values = fields
        .iter()
        .map(|field| {
            value_to_mx_for_interface_in_context(
                value
                    .fields
                    .get(field)
                    .expect("field name came from the same struct"),
                mode,
                interface,
                context,
            )
            .map(Box::new)
            .map(Some)
        })
        .collect::<Result<Vec<_>, _>>()?;
    MxArray::structure(fields, values, vec![1, 1]).map_err(MxConversionError::new)
}

fn object_to_mx(
    class_name: &runmat_types::ClassIdentity,
    objects: &[&ObjectInstance],
    shape: Vec<usize>,
    mode: MxApiMode,
    interface: MxBoundaryInterface,
    context: Option<&MxValueContext>,
) -> Result<MxArray, MxConversionError> {
    if objects
        .iter()
        .any(|object| &object.class_name != class_name)
    {
        return Err(MxConversionError::new(
            "C Matrix object arrays must contain one concrete class",
        ));
    }
    let mut properties = objects
        .iter()
        .flat_map(|object| object.properties.keys().cloned())
        .collect::<Vec<_>>();
    properties.sort();
    properties.dedup();
    let mut values = Vec::with_capacity(properties.len() * objects.len());
    for property in &properties {
        for object in objects {
            values.push(
                object
                    .properties
                    .get(property)
                    .map(|value| {
                        value_to_mx_for_interface_in_context(value, mode, interface, context)
                            .map(Box::new)
                    })
                    .transpose()?,
            );
        }
    }
    MxArray::object(class_name.clone(), properties, values, shape).map_err(MxConversionError::new)
}

fn numeric_from_mx(value: &MxNumeric, shape: &[usize]) -> Result<Value, MxConversionError> {
    if let Some(imag) = &value.imag {
        return complex_from_host_components(value.real.clone(), imag.clone(), shape);
    }
    if value.real.len() == 1 {
        return scalar_from_numeric(value.real.value_at(0).unwrap(), shape);
    }
    Tensor::from_host_buffer(value.real.clone(), shape.to_vec())
        .map(Value::Tensor)
        .map_err(MxConversionError::new)
}

fn complex_from_host_components(
    real: HostNumericBuffer,
    imag: HostNumericBuffer,
    shape: &[usize],
) -> Result<Value, MxConversionError> {
    complex_from_components(
        real.into_numeric_storage(),
        imag.into_numeric_storage(),
        shape,
    )
}

fn scalar_from_numeric(value: NumericScalar, shape: &[usize]) -> Result<Value, MxConversionError> {
    match value {
        NumericScalar::F64(value) => Ok(Value::Num(value)),
        NumericScalar::F32(value) => Tensor::from_f32(vec![value], shape.to_vec())
            .map(Value::Tensor)
            .map_err(MxConversionError::new),
        value => Ok(Value::Int(
            value
                .into_int_value()
                .expect("non-floating numeric scalar is integer"),
        )),
    }
}

fn complex_from_components(
    real: NumericStorage,
    imag: NumericStorage,
    shape: &[usize],
) -> Result<Value, MxConversionError> {
    if real.numeric_dtype() != imag.numeric_dtype() || real.len() != imag.len() {
        return Err(MxConversionError::new(
            "complex components have different classes or lengths",
        ));
    }
    if real.len() == 1 && matches!(real, NumericStorage::F64(_)) {
        let NumericScalar::F64(real) = real.value_at(0).unwrap() else {
            unreachable!()
        };
        let NumericScalar::F64(imag) = imag.value_at(0).unwrap() else {
            unreachable!()
        };
        return Ok(Value::Complex(real, imag));
    }
    let storage = match (real, imag) {
        (NumericStorage::F64(real), NumericStorage::F64(imag)) => ComplexStorage::F64(
            real.into_iter()
                .zip(imag)
                .map(ComplexElement::from)
                .collect(),
        ),
        (NumericStorage::F32(real), NumericStorage::F32(imag)) => ComplexStorage::F32(
            real.into_iter()
                .zip(imag)
                .map(ComplexElement::from)
                .collect(),
        ),
        (real, imag) => ComplexStorage::Integer(
            IntegerComplexStorage::new(
                real.into_integer_storage()
                    .map_err(|_| MxConversionError::new("real complex storage is not integer"))?,
                imag.into_integer_storage().map_err(|_| {
                    MxConversionError::new("imaginary complex storage is not integer")
                })?,
            )
            .map_err(MxConversionError::new)?,
        ),
    };
    ComplexTensor::from_complex_storage(storage, shape.to_vec())
        .map(Value::ComplexTensor)
        .map_err(MxConversionError::new)
}

fn char_from_mx(values: &[u16], shape: &[usize]) -> Result<Value, MxConversionError> {
    record_host_copy(
        HostCopyReason::CharacterEncoding,
        values.len().saturating_mul(std::mem::size_of::<u16>()),
    );
    let mut characters = Vec::with_capacity(values.len());
    for value in values {
        let character = char::from_u32(u32::from(*value)).ok_or_else(|| {
            MxConversionError::new(
                "UTF-16 surrogate code units cannot be represented by the current RunMat character scalar",
            )
        })?;
        characters.push(character);
    }
    CharArray::from_column_major(characters, shape.to_vec())
        .map(Value::CharArray)
        .map_err(MxConversionError::new)
}

fn cell_from_mx(
    values: &[Option<Box<MxArray>>],
    shape: &[usize],
    context: Option<&MxValueContext>,
) -> Result<Value, MxConversionError> {
    let values = values
        .iter()
        .map(|value| {
            value
                .as_deref()
                .map(|value| value_from_mx_in_context(value, context))
                .transpose()
                .map(|value| value.unwrap_or_else(|| Value::Tensor(Tensor::zeros(vec![0, 0]))))
        })
        .collect::<Result<Vec<_>, _>>()?;
    CellArray::from_column_major(values, shape.to_vec())
        .map(Value::Cell)
        .map_err(MxConversionError::new)
}

fn struct_from_mx(
    fields: &[String],
    values: &[Option<Box<MxArray>>],
    shape: &[usize],
    context: Option<&MxValueContext>,
) -> Result<Value, MxConversionError> {
    let numel = shape
        .iter()
        .try_fold(1usize, |product, extent| product.checked_mul(*extent))
        .ok_or_else(|| MxConversionError::new("structure array shape exceeds platform limits"))?;
    let expected = fields.len().checked_mul(numel).ok_or_else(|| {
        MxConversionError::new("structure array field storage exceeds platform limits")
    })?;
    if values.len() != expected {
        return Err(MxConversionError::new(
            "structure array field storage is inconsistent",
        ));
    }
    let field_values = values
        .iter()
        .map(|value| {
            value
                .as_deref()
                .map(|value| value_from_mx_in_context(value, context))
                .transpose()
                .map(|value| value.unwrap_or_else(|| Value::Tensor(Tensor::zeros(vec![0, 0]))))
        })
        .collect::<Result<Vec<_>, _>>()?;
    StructArray::normalize_field_major(fields.to_vec(), field_values, shape.to_vec())
        .map_err(MxConversionError::new)
}

fn object_from_mx(
    class_name: &runmat_types::ClassIdentity,
    properties: &[String],
    values: &[Option<Box<MxArray>>],
    shape: &[usize],
    context: Option<&MxValueContext>,
) -> Result<Value, MxConversionError> {
    let numel = shape.iter().try_fold(1usize, |count, extent| {
        count
            .checked_mul(*extent)
            .ok_or_else(|| MxConversionError::new("object array shape exceeds platform limits"))
    })?;
    let mut objects = Vec::with_capacity(numel);
    for element in 0..numel {
        let mut object = ObjectInstance::new(class_name.clone());
        for (property_index, property) in properties.iter().enumerate() {
            if let Some(value) = values[property_index * numel + element].as_deref() {
                object
                    .properties
                    .insert(property.clone(), value_from_mx_in_context(value, context)?);
            }
        }
        objects.push(object);
    }
    if numel == 1 {
        return Ok(Value::Object(objects.pop().expect("one object element")));
    }
    ObjectArray::from_objects(class_name.clone(), objects, shape.to_vec())
        .map(Value::ObjectArray)
        .map_err(MxConversionError::new)
}

fn sparse_from_mx(value: &MxSparse) -> Result<Value, MxConversionError> {
    let nnz = value.col_ptrs.last().copied().unwrap_or(0);
    if nnz > value.nzmax {
        return Err(MxConversionError::new(
            "sparse column pointers exceed allocated nzmax",
        ));
    }
    let exact_capacity = value.row_indices.len() == nnz;
    let row_indices = if exact_capacity {
        value.row_indices.clone()
    } else {
        record_host_copy(
            HostCopyReason::SparseLayoutConversion,
            nnz.saturating_mul(std::mem::size_of::<usize>()),
        );
        HostIndexBuffer::new(value.row_indices[..nnz].to_vec())
    };
    let result = match &value.values {
        MxSparseValues::Logical(_) => SparseTensor::from_host_logical_pattern(
            value.rows,
            value.cols,
            value.col_ptrs.clone(),
            row_indices,
        ),
        MxSparseValues::Numeric(values) if exact_capacity && values.len() == nnz => {
            SparseTensor::from_host_numeric_buffers(
                value.rows,
                value.cols,
                value.col_ptrs.clone(),
                row_indices,
                values.clone(),
            )
        }
        MxSparseValues::Numeric(values) => {
            let mut compact = NumericStorage::zeros(values.numeric_dtype(), nnz);
            for index in 0..nnz {
                compact
                    .set_value(
                        index,
                        values.value_at(index).ok_or_else(|| {
                            MxConversionError::new("sparse numeric storage is incomplete")
                        })?,
                    )
                    .map_err(MxConversionError::new)?;
            }
            record_host_copy(
                HostCopyReason::SparseLayoutConversion,
                compact.checked_byte_len().unwrap_or(usize::MAX),
            );
            SparseTensor::from_host_numeric_buffers(
                value.rows,
                value.cols,
                value.col_ptrs.clone(),
                row_indices,
                HostNumericBuffer::from_numeric_storage(compact),
            )
        }
        MxSparseValues::InterleavedComplex(values) if exact_capacity && values.len() == nnz => {
            SparseTensor::from_host_complex_buffers(
                value.rows,
                value.cols,
                value.col_ptrs.clone(),
                row_indices,
                values.clone(),
            )
        }
        MxSparseValues::InterleavedComplex(values) => {
            record_host_copy(
                HostCopyReason::SparseLayoutConversion,
                nnz.saturating_mul(std::mem::size_of::<ComplexElement<f64>>()),
            );
            SparseTensor::from_host_complex_buffers(
                value.rows,
                value.cols,
                value.col_ptrs.clone(),
                row_indices,
                HostComplexBuffer::from_elements(values[..nnz].to_vec()),
            )
        }
        MxSparseValues::SeparateComplex { real, imaginary } => {
            let mut interleaved = Vec::with_capacity(nnz);
            for index in 0..nnz {
                let real = real
                    .as_f64_slice()
                    .and_then(|values| values.get(index))
                    .copied()
                    .ok_or_else(|| MxConversionError::new("sparse real storage is incomplete"))?;
                let imaginary = imaginary
                    .as_f64_slice()
                    .and_then(|values| values.get(index))
                    .copied()
                    .ok_or_else(|| {
                        MxConversionError::new("sparse imaginary storage is incomplete")
                    })?;
                interleaved.push(ComplexElement(real, imaginary));
            }
            record_host_copy(
                HostCopyReason::SparseLayoutConversion,
                interleaved
                    .len()
                    .saturating_mul(std::mem::size_of::<ComplexElement<f64>>()),
            );
            SparseTensor::from_host_complex_buffers(
                value.rows,
                value.cols,
                value.col_ptrs.clone(),
                row_indices,
                HostComplexBuffer::from_elements(interleaved),
            )
        }
    };
    result
        .map(Value::SparseTensor)
        .map_err(MxConversionError::new)
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::StringArray(_) => "string array",
        Value::Symbolic(_) | Value::SymbolicArray(_) => "symbolic",
        Value::GpuTensor(_) => "GPU-resident",
        Value::HandleObject(_) => "handle object",
        Value::Listener(_) => "listener",
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_) => "callable",
        Value::MException(_) => "exception",
        Value::Future(_) | Value::Task(_) | Value::Pool(_) | Value::Job(_) => "execution-handle",
        Value::Foreign(_) => "foreign",
        _ => "unsupported",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntValue, IntegerStorage};

    #[test]
    fn mex_conversion_rejects_nested_transient_sequences_before_recursion() {
        let nested = Value::Cell(
            runmat_value::CellArray::new(vec![Value::OutputList(Vec::new())], 1, 1)
                .expect("nested cell"),
        );
        let error = value_to_mx(&nested, MxApiMode::SeparateComplex)
            .expect_err("transient sequence must not cross MEX");
        assert!(error.to_string().contains("TransientSequenceNotPortable"));
    }

    #[test]
    fn all_numeric_classes_round_trip_exactly_including_wide_uint64() {
        let cases = vec![
            NumericStorage::F64(vec![1.25, -2.5]),
            NumericStorage::F32(vec![1.25, -2.5]),
            NumericStorage::I8(vec![i8::MIN, i8::MAX]),
            NumericStorage::I16(vec![i16::MIN, i16::MAX]),
            NumericStorage::I32(vec![i32::MIN, i32::MAX]),
            NumericStorage::I64(vec![i64::MIN, i64::MAX]),
            NumericStorage::U8(vec![0, u8::MAX]),
            NumericStorage::U16(vec![0, u16::MAX]),
            NumericStorage::U32(vec![0, u32::MAX]),
            NumericStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];
        for storage in cases {
            let original = Value::Tensor(
                Tensor::from_numeric_storage(storage, vec![2, 1]).expect("typed tensor"),
            );
            for mode in [MxApiMode::SeparateComplex, MxApiMode::InterleavedComplex] {
                let boundary = value_to_mx(&original, mode).expect("convert to mxArray");
                assert_eq!(value_from_mx(&boundary).expect("convert back"), original);
            }
        }
    }

    #[test]
    fn dense_numeric_boundary_retains_one_host_allocation() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let original = Value::Tensor(tensor.clone());
        let mut boundary = value_to_mx(&original, MxApiMode::SeparateComplex).unwrap();

        // SAFETY: both addresses are observed without dereferencing and both
        // owners remain alive for the duration of the comparison.
        let source_pointer = unsafe { tensor.host_buffer().foreign_data_pointer() };
        assert_eq!(boundary.data_pointer(), source_pointer);

        let output = value_from_mx(&boundary).unwrap();
        let Value::Tensor(output) = output else {
            panic!("dense array should remain a tensor");
        };
        assert!(output
            .host_buffer()
            .shares_allocation_with(tensor.host_buffer()));
    }

    #[test]
    fn interleaved_complex_boundary_retains_one_host_allocation() {
        let tensor = ComplexTensor::new(vec![(1.0, -2.0), (3.0, -4.0)], vec![2, 1]).unwrap();
        let ComplexStorage::F64(source) = tensor.complex_storage() else {
            unreachable!("constructor creates double complex storage")
        };
        let original = Value::ComplexTensor(tensor.clone());
        let mut boundary = value_to_mx(&original, MxApiMode::InterleavedComplex).unwrap();

        let MxArrayData::Interleaved(interleaved) = boundary.data() else {
            panic!("interleaved mode must use interleaved storage");
        };
        let MxInterleavedStorage::F64(boundary_values) = &interleaved.values else {
            panic!("double complex input must retain its class");
        };
        assert!(source.shares_allocation_with(boundary_values));

        // SAFETY: both addresses are observed without dereferencing and the
        // owners remain alive for the duration of the comparison.
        let source_pointer = unsafe { source.foreign_data_pointer() };
        assert_eq!(boundary.data_pointer(), source_pointer);

        let output = value_from_mx(&boundary).unwrap();
        let Value::ComplexTensor(output) = output else {
            panic!("non-scalar complex array must remain a tensor");
        };
        let ComplexStorage::F64(output_values) = output.complex_storage() else {
            panic!("double complex output must retain its class");
        };
        assert!(source.shares_allocation_with(output_values));
    }

    #[test]
    fn sparse_complex_boundary_is_zero_copy_when_interleaved_and_explicit_when_separate() {
        let sparse = SparseTensor::new_complex(
            3,
            2,
            vec![0, 2, 3],
            vec![0, 2, 1],
            vec![(1.0, -2.0), (3.0, 4.0), (-5.0, 6.0)],
        )
        .unwrap();
        let source = sparse.complex_host_buffer().unwrap();
        let original = Value::SparseTensor(sparse.clone());

        let mut interleaved = value_to_mx(&original, MxApiMode::InterleavedComplex).unwrap();
        let MxArrayData::Sparse(MxSparse {
            values: MxSparseValues::InterleavedComplex(boundary_values),
            ..
        }) = interleaved.data()
        else {
            panic!("interleaved mode must retain sparse interleaved storage");
        };
        assert!(source.shares_allocation_with(boundary_values));
        // SAFETY: both owners remain alive and the addresses are not dereferenced.
        assert_eq!(interleaved.data_pointer(), unsafe {
            source.foreign_data_pointer()
        });
        let Value::SparseTensor(round_trip) = value_from_mx(&interleaved).unwrap() else {
            panic!("sparse complex output expected");
        };
        assert!(source.shares_allocation_with(round_trip.complex_host_buffer().unwrap()));

        let mut separate = value_to_mx(&original, MxApiMode::SeparateComplex).unwrap();
        let MxArrayData::Sparse(MxSparse {
            values: MxSparseValues::SeparateComplex { real, imaginary },
            ..
        }) = separate.data()
        else {
            panic!("separate mode must expose sparse component storage");
        };
        assert_eq!(real.as_f64_slice(), Some(&[1.0, 3.0, -5.0][..]));
        assert_eq!(imaginary.as_f64_slice(), Some(&[-2.0, 4.0, 6.0][..]));
        assert!(!separate.data_pointer().is_null());
        assert!(!separate.imaginary_pointer().is_null());
        assert_eq!(value_from_mx(&separate).unwrap(), original);
    }

    #[test]
    fn sparse_complex_single_reports_the_matrix_boundary_limit_explicitly() {
        let sparse = SparseTensor::new_complex_f32(1, 1, vec![0, 1], vec![0], vec![(1.0, -2.0)])
            .expect("complex single sparse value");
        let error = value_to_mx(&Value::SparseTensor(sparse), MxApiMode::InterleavedComplex)
            .expect_err("the current matrix boundary has no complex-single sparse representation");
        assert!(error
            .to_string()
            .contains("does not represent complex-single sparse storage"));
    }

    #[test]
    fn complex_integer_storage_round_trips_in_both_api_modes() {
        let original = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                    IntegerStorage::U64(vec![1, 2]),
                )
                .unwrap(),
                vec![1, 2],
            )
            .unwrap(),
        );
        for mode in [MxApiMode::SeparateComplex, MxApiMode::InterleavedComplex] {
            let boundary = value_to_mx(&original, mode).unwrap();
            assert_eq!(value_from_mx(&boundary).unwrap(), original);
        }
    }

    #[test]
    fn structure_arrays_use_field_major_mx_storage() {
        let elements = (0..4)
            .map(|index| {
                let mut value = StructValue::new();
                value.insert(
                    "id",
                    Value::Int(IntValue::U64(9_007_199_254_740_993 + index as u64)),
                );
                value.insert("weight", Value::Num(10.0 + index as f64));
                value
            })
            .collect();
        let original = Value::StructArray(StructArray::new(elements, vec![2, 2]).unwrap());
        let boundary = value_to_mx(&original, MxApiMode::InterleavedComplex).unwrap();
        let MxArrayData::Struct { fields, values } = boundary.data() else {
            panic!("expected structure boundary")
        };
        assert_eq!(fields, &["id", "weight"]);
        assert_eq!(values.len(), 8);
        for index in 0..4 {
            assert_eq!(
                value_from_mx(values[index].as_deref().unwrap()).unwrap(),
                Value::Int(IntValue::U64(9_007_199_254_740_993 + index as u64))
            );
            assert_eq!(
                value_from_mx(values[4 + index].as_deref().unwrap()).unwrap(),
                Value::Num(10.0 + index as f64)
            );
        }
        assert_eq!(value_from_mx(&boundary).unwrap(), original);
    }

    #[test]
    fn structure_array_boundary_preserves_nd_and_empty_shapes() {
        let nd_elements = (0..4)
            .map(|index| {
                let mut value = StructValue::new();
                value.insert("id", Value::Num(index as f64));
                value
            })
            .collect();
        let nd = Value::StructArray(StructArray::new(nd_elements, vec![2, 1, 2]).unwrap());
        let empty = Value::StructArray(
            StructArray::empty(vec!["id".into(), "weight".into()], vec![0, 3, 2]).unwrap(),
        );
        for original in [nd, empty] {
            let boundary = value_to_mx(&original, MxApiMode::InterleavedComplex).unwrap();
            assert_eq!(value_from_mx(&boundary).unwrap(), original);
        }
    }

    #[test]
    fn cells_of_structs_remain_cells_at_the_mx_boundary() {
        let original =
            Value::Cell(CellArray::new(vec![Value::Struct(StructValue::new())], 1, 1).unwrap());
        let boundary = value_to_mx(&original, MxApiMode::InterleavedComplex).unwrap();
        assert!(matches!(boundary.data(), MxArrayData::Cell(_)));
        assert_eq!(value_from_mx(&boundary).unwrap(), original);
    }
}
