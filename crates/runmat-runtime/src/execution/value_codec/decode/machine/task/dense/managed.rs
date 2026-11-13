//! The canonical dense task's managed backing. Shape conversion scratch is
//! separately claimed; decoded elements go directly into their final exact
//! allocation. No Vec payload is constructed and subsequently copied/adopted.

use super::*;
use crate::context::host_allocation::{
    collection_block_retained_bytes, AdmittedCollectionAllocator, HostAllocationError,
    HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use allocator_api2::boxed::Box as AllocatedBox;
use runmat_value::{
    ExactHostComplexTensorAllocation, ExactHostIntegerComplexTensorAllocation,
    ExactHostLogicalTensorAllocation, ExactHostTensorAllocation, NumericDType, NumericScalar,
};
use std::alloc::Layout;

pub(super) struct ManagedDense<'a> {
    // Physical owners precede their stage receipt in destruction order.
    output: Option<Output>,
    shape: HostTypedWorkspace<usize>,
    shape_offset: usize,
    element_offset: usize,
    receipt: HostAllocationReceipt,
    runtime: &'a RuntimeContext,
}

pub(super) type ManagedDenseOwner<'a> = AllocatedBox<ManagedDense<'a>, AdmittedCollectionAllocator>;

enum Output {
    Numeric(ExactHostTensorAllocation),
    Logical(ExactHostLogicalTensorAllocation),
    Complex(ExactHostComplexTensorAllocation),
    IntegerComplex(ExactHostIntegerComplexTensorAllocation),
}

impl<'a> ManagedDense<'a> {
    pub(super) fn new(
        source: &DenseValue,
        count: usize,
        path: &str,
        runtime: Option<&'a RuntimeContext>,
    ) -> Result<Option<ManagedDenseOwner<'a>>, ValueCodecError> {
        if HostAllocationReceipt::policy(runtime)
            .map_err(|error| ValueCodecError::allocation(path, error))?
            == HostAllocationPolicy::Unbounded
        {
            return Ok(None);
        }
        let dtype = numeric_dtype(source.element_type);
        let rank = source.shape.len();
        let (raw, claims) = if source.element_type == ElementType::Logical {
            (
                ExactHostLogicalTensorAllocation::requested_bytes_for_dimensions(count, rank),
                ExactHostLogicalTensorAllocation::retained_claim_count(),
            )
        } else if is_complex_integer(source.element_type) {
            (
                ExactHostIntegerComplexTensorAllocation::requested_bytes_for_dimensions(
                    dtype, count, rank,
                ),
                ExactHostIntegerComplexTensorAllocation::retained_claim_count(),
            )
        } else if matches!(
            source.element_type,
            ElementType::ComplexF32 | ElementType::ComplexF64
        ) {
            (
                ExactHostComplexTensorAllocation::requested_bytes_for_dimensions(
                    dtype, count, rank,
                ),
                ExactHostComplexTensorAllocation::retained_claim_count(),
            )
        } else {
            (
                ExactHostTensorAllocation::requested_bytes_for_dimensions(dtype, count, rank),
                ExactHostTensorAllocation::retained_claim_count(),
            )
        };
        let plan = (|| {
            let raw = raw.map_err(HostAllocationError::Numeric)?;
            let raw = u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?;
            let output = HostAllocationPlan::retained_exact_output_bytes(raw, claims)?
                .checked_add(collection_block_retained_bytes(Layout::new::<Self>())?)
                .ok_or(HostAllocationError::Overflow)?;
            HostAllocationPlan::new(HostTypedWorkspace::<usize>::quoted_bytes(rank)?, output)
        })()
        .map_err(|error| ValueCodecError::allocation(path, error))?;
        let receipt = HostAllocationReceipt::admit(runtime, plan)
            .map_err(|error| ValueCodecError::allocation(path, error))?
            .ok_or_else(|| {
                ValueCodecError::allocation(path, HostAllocationError::StaleAuthority)
            })?;
        let runtime = runtime.expect("managed receipt has a selected runtime");
        let shape = receipt
            .temporary_typed(runtime, rank)
            .map_err(|error| ValueCodecError::allocation(path, error))?;
        // This control block is not tensor payload. Use the same exact-layout
        // allocator as other managed collections, retaining its own claim;
        // do not add an unaccounted Box merely to compact the decoder stack.
        let allocator = receipt
            .collection_allocator(runtime)
            .map_err(|error| ValueCodecError::allocation(path, error))?;
        let owner = AllocatedBox::<Self, _>::try_new_uninit_in(allocator);
        receipt
            .checkpoint(runtime)
            .map_err(|error| ValueCodecError::allocation(path, error))?;
        let owner = owner.map_err(|_| {
            ValueCodecError::allocation(
                path,
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
            )
        })?;
        Ok(Some(AllocatedBox::write(
            owner,
            Self {
                output: None,
                shape,
                shape_offset: 0,
                element_offset: 0,
                receipt,
                runtime,
            },
        )))
    }

    pub(super) fn advance(
        &mut self,
        source: &DenseValue,
        maximum_work: usize,
        path: &str,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        self.checkpoint(path)?;
        if self.shape_offset < source.shape.len() {
            let end = source
                .shape
                .len()
                .min(self.shape_offset.saturating_add(maximum_work));
            for index in self.shape_offset..end {
                self.shape.as_mut_slice()[index] = usize::try_from(source.shape[index])
                    .map_err(|_| ValueCodecError::invalid(path, "value exceeds platform limits"))?;
            }
            let completed = end - self.shape_offset;
            self.shape_offset = end;
            self.checkpoint(path)?;
            return Ok((completed, None));
        }
        if self.output.is_none() {
            let permit = self
                .receipt
                .output_permit(self.runtime)
                .map_err(|error| ValueCodecError::allocation(path, error))?;
            let shape = self.shape.as_slice();
            let dtype = numeric_dtype(source.element_type);
            let result = if source.element_type == ElementType::Logical {
                ExactHostLogicalTensorAllocation::new(shape, &permit).map(Output::Logical)
            } else if is_complex_integer(source.element_type) {
                ExactHostIntegerComplexTensorAllocation::new(dtype, shape, &permit)
                    .map(Output::IntegerComplex)
            } else if matches!(
                source.element_type,
                ElementType::ComplexF32 | ElementType::ComplexF64
            ) {
                ExactHostComplexTensorAllocation::new(dtype, shape, &permit).map(Output::Complex)
            } else {
                ExactHostTensorAllocation::new(dtype, shape, &permit).map(Output::Numeric)
            };
            self.output = Some(result.map_err(|error| self.allocation_error(path, error))?);
            self.checkpoint(path)?;
        }
        let width = element_width(source.element_type);
        let count = source.little_endian_data.len() / width;
        let end = count.min(self.element_offset.saturating_add(maximum_work));
        let output = self.output.as_mut().expect("admitted dense output");
        for index in self.element_offset..end {
            let bytes = &source.little_endian_data[index * width..(index + 1) * width];
            match output {
                Output::Numeric(output) => output
                    .set_value(index, scalar(numeric_dtype(source.element_type), bytes))
                    .map_err(|error| ValueCodecError::invalid(path, error))?,
                Output::Logical(output) => output.as_mut_slice()[index] = bytes[0],
                Output::Complex(output) => {
                    let lane = width / 2;
                    let dtype = numeric_dtype(source.element_type);
                    let real = scalar(dtype, &bytes[..lane]);
                    let imag = scalar(dtype, &bytes[lane..]);
                    match (real, imag) {
                        (NumericScalar::F32(real), NumericScalar::F32(imag)) => {
                            output.set_f32(index, ComplexElement(real, imag))
                        }
                        (NumericScalar::F64(real), NumericScalar::F64(imag)) => {
                            output.set(index, real, imag)
                        }
                        _ => unreachable!("floating complex class"),
                    }
                    .map_err(|error| ValueCodecError::allocation(path, error.into()))?;
                }
                Output::IntegerComplex(output) => {
                    let lane = width / 2;
                    let dtype = numeric_dtype(source.element_type);
                    output
                        .set(
                            index,
                            scalar(dtype, &bytes[..lane]),
                            scalar(dtype, &bytes[lane..]),
                        )
                        .map_err(|error| ValueCodecError::invalid(path, error))?;
                }
            }
        }
        let completed = (end - self.element_offset).max(1);
        self.element_offset = end;
        self.checkpoint(path)?;
        if end != count {
            return Ok((completed, None));
        }
        let result = match self.output.take().expect("completed dense output") {
            Output::Numeric(output) => output.into_tensor().map(Value::Tensor),
            Output::Logical(output) => output.into_array().map(Value::LogicalArray),
            Output::Complex(output) => output.into_tensor().map(Value::ComplexTensor),
            Output::IntegerComplex(output) => output.into_tensor().map(Value::ComplexTensor),
        };
        let value = result.map_err(|error| self.allocation_error(path, error))?;
        self.checkpoint(path)?;
        Ok((completed, Some(value)))
    }

    fn checkpoint(&self, path: &str) -> Result<(), ValueCodecError> {
        self.receipt
            .checkpoint(self.runtime)
            .map_err(|error| ValueCodecError::allocation(path, error))
    }

    fn allocation_error(
        &self,
        path: &str,
        source: runmat_value::ExactHostNumericAllocationError,
    ) -> ValueCodecError {
        ValueCodecError::allocation(
            path,
            self.receipt
                .checkpoint(self.runtime)
                .err()
                .unwrap_or(HostAllocationError::Numeric(source)),
        )
    }
}

pub(in crate::execution::value_codec::decode::machine) fn numeric_dtype(
    element: ElementType,
) -> NumericDType {
    match element {
        ElementType::Logical | ElementType::U8 | ElementType::ComplexU8 => NumericDType::U8,
        ElementType::F32 | ElementType::ComplexF32 => NumericDType::F32,
        ElementType::F64 | ElementType::ComplexF64 => NumericDType::F64,
        ElementType::I8 | ElementType::ComplexI8 => NumericDType::I8,
        ElementType::I16 | ElementType::ComplexI16 => NumericDType::I16,
        ElementType::I32 | ElementType::ComplexI32 => NumericDType::I32,
        ElementType::I64 | ElementType::ComplexI64 => NumericDType::I64,
        ElementType::U16 | ElementType::ComplexU16 => NumericDType::U16,
        ElementType::U32 | ElementType::ComplexU32 => NumericDType::U32,
        ElementType::U64 | ElementType::ComplexU64 => NumericDType::U64,
    }
}

pub(in crate::execution::value_codec::decode::machine) fn scalar(
    dtype: NumericDType,
    bytes: &[u8],
) -> NumericScalar {
    macro_rules! decode {
        ($ty:ty, $variant:ident) => {
            NumericScalar::$variant(<$ty>::from_le_bytes(
                bytes.try_into().expect("validated element width"),
            ))
        };
    }
    match dtype {
        NumericDType::F32 => decode!(f32, F32),
        NumericDType::F64 => decode!(f64, F64),
        NumericDType::I8 => decode!(i8, I8),
        NumericDType::I16 => decode!(i16, I16),
        NumericDType::I32 => decode!(i32, I32),
        NumericDType::I64 => decode!(i64, I64),
        NumericDType::U8 => decode!(u8, U8),
        NumericDType::U16 => decode!(u16, U16),
        NumericDType::U32 => decode!(u32, U32),
        NumericDType::U64 => decode!(u64, U64),
    }
}

#[cfg(test)]
#[path = "managed/tests.rs"]
mod tests;
