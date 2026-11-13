//! Exact host solve publication, independent of the factor/triangular kernel.
//! The private payload and immutable shape have independent retained claims.
use crate::builtins::common::{random_args::complex_tensor_into_value, tensor::tensor_into_value};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::{
    ComplexElement, ExactHostAllocationPermit, ExactHostComplexTensorAllocation,
    ExactHostNumericAllocationError, ExactHostTensorAllocation, NumericDType, Value,
};
use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for runmat_value::ComplexElement<f32> {}
    impl Sealed for runmat_value::ComplexElement<f64> {}
}

pub(super) trait SolveOutputElement: sealed::Sealed + Copy {
    type Allocation;
    fn quote(shape: &[usize]) -> Result<u64, HostAllocationError>;
    fn allocate(
        shape: &[usize],
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self::Allocation, ExactHostNumericAllocationError>;
    fn write(
        allocation: &mut Self::Allocation,
        index: usize,
        value: Self,
    ) -> Result<(), ExactHostNumericAllocationError>;
    fn finish(allocation: Self::Allocation) -> Result<Value, ExactHostNumericAllocationError>;
    fn borrow_mut(allocation: &mut Self::Allocation) -> Option<&mut [Self]>;
}

/// This owner is private through construction, every write and final fencing.
/// It publishes no Value on allocation, kernel, cancellation or authority error.
pub(super) struct ManagedSolveOutput<T: SolveOutputElement> {
    allocation: T::Allocation,
    receipt: HostAllocationReceipt,
    scalar: PhantomData<T>,
}
impl<T: SolveOutputElement> ManagedSolveOutput<T> {
    /// Transfer the private allocation and its issuer together to an enclosing
    /// solve publication owner. Neither part authorizes publication alone.
    pub(super) fn into_parts(self) -> (T::Allocation, HostAllocationReceipt) {
        (self.allocation, self.receipt)
    }

    pub(super) fn new(
        shape: &[usize],
        context: &RuntimeContext,
    ) -> Result<Self, HostAllocationError> {
        Self::new_guarded(shape, context, || Ok(()))
    }
    pub(super) fn new_guarded(
        shape: &[usize],
        context: &RuntimeContext,
        original_owner: impl Fn() -> Result<(), HostAllocationError>,
    ) -> Result<Self, HostAllocationError> {
        original_owner()?;
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, T::quote(shape)?)?,
        )?
        .ok_or(HostAllocationError::MissingManagedOwner)?;
        original_owner()?;
        let allocation =
            receipt.with_output_allocation(context, |permit| T::allocate(shape, permit))?;
        original_owner()?;
        Ok(Self {
            allocation,
            receipt,
            scalar: PhantomData,
        })
    }
    /// Lend the private native result to an in-place numerical kernel. No
    /// second working payload or published Value exists during this borrow.
    pub(super) fn as_mut_slice(&mut self) -> Result<&mut [T], HostAllocationError> {
        T::borrow_mut(&mut self.allocation).ok_or(HostAllocationError::Numeric(
            ExactHostNumericAllocationError::UnsupportedClass,
        ))
    }
    pub(super) fn finish(self, context: &RuntimeContext) -> Result<Value, HostAllocationError> {
        self.receipt.checkpoint(context)?;
        let value = T::finish(self.allocation).map_err(HostAllocationError::Numeric)?;
        self.receipt.checkpoint(context)?;
        Ok(value)
    }
}

macro_rules! real_output {
    ($scalar:ty,$dtype:expr,$view:ident) => {
        impl SolveOutputElement for $scalar {
            type Allocation = ExactHostTensorAllocation;
            fn borrow_mut(allocation: &mut Self::Allocation) -> Option<&mut [Self]> {
                allocation.$view()
            }
            fn quote(shape: &[usize]) -> Result<u64, HostAllocationError> {
                HostAllocationPlan::retained_tensor_bytes($dtype, shape)
            }
            fn allocate(
                shape: &[usize],
                permit: &dyn ExactHostAllocationPermit,
            ) -> Result<Self::Allocation, ExactHostNumericAllocationError> {
                ExactHostTensorAllocation::new($dtype, shape, permit)
            }
            fn write(
                allocation: &mut Self::Allocation,
                index: usize,
                value: Self,
            ) -> Result<(), ExactHostNumericAllocationError> {
                let target = allocation
                    .$view()
                    .ok_or(ExactHostNumericAllocationError::UnsupportedClass)?
                    .get_mut(index)
                    .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
                *target = value;
                Ok(())
            }
            fn finish(
                allocation: Self::Allocation,
            ) -> Result<Value, ExactHostNumericAllocationError> {
                allocation.into_tensor().map(tensor_into_value)
            }
        }
    };
}
real_output!(f32, NumericDType::F32, as_f32_slice_mut);
real_output!(f64, NumericDType::F64, as_f64_slice_mut);

macro_rules! complex_output {
    ($scalar:ty,$dtype:expr,$view:ident,$write:expr) => {
        impl SolveOutputElement for ComplexElement<$scalar> {
            type Allocation = ExactHostComplexTensorAllocation;
            fn borrow_mut(allocation: &mut Self::Allocation) -> Option<&mut [Self]> {
                allocation.$view()
            }
            fn quote(shape: &[usize]) -> Result<u64, HostAllocationError> {
                HostAllocationPlan::retained_complex_tensor_bytes($dtype, shape)
            }
            fn allocate(
                shape: &[usize],
                permit: &dyn ExactHostAllocationPermit,
            ) -> Result<Self::Allocation, ExactHostNumericAllocationError> {
                ExactHostComplexTensorAllocation::new($dtype, shape, permit)
            }
            fn write(
                allocation: &mut Self::Allocation,
                index: usize,
                value: Self,
            ) -> Result<(), ExactHostNumericAllocationError> {
                ($write)(allocation, index, value)
            }
            fn finish(
                allocation: Self::Allocation,
            ) -> Result<Value, ExactHostNumericAllocationError> {
                allocation.into_tensor().map(complex_tensor_into_value)
            }
        }
    };
}
complex_output!(
    f32,
    NumericDType::F32,
    as_complex_f32_slice_mut,
    |allocation: &mut ExactHostComplexTensorAllocation, index, value: ComplexElement<f32>| {
        allocation.set_f32(index, value)
    }
);
complex_output!(
    f64,
    NumericDType::F64,
    as_complex_f64_slice_mut,
    |allocation: &mut ExactHostComplexTensorAllocation, index, value: ComplexElement<f64>| {
        allocation.set(index, value.0, value.1)
    }
);
