//! Canonical dense output owners, exact permutation owners, and StackReq scratch.
use super::*;
use crate::builtins::math::linalg::ops::sparse_factor::RetainedFactorScalar;
use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, ExactHostComplexTensorAllocation,
    ExactHostIndexBufferAllocation, HostIndexBuffer, NumericDType, Tensor,
};

pub(in crate::builtins::math::linalg) trait DenseScalar:
    FaerFactorScalar + RetainedFactorScalar
{
    const DTYPE: NumericDType;
    const COMPLEX: bool;
    fn elements(matrix: &NativeDenseFactorMatrix) -> Option<&[Self]>;
    fn elements_mut(matrix: &mut NativeDenseFactorMatrix) -> Option<&mut [Self]>;
    fn faer(values: &[Self]) -> &[Self::Element];
    fn faer_mut(values: &mut [Self]) -> &mut [Self::Element];
    fn from_element(value: Self::Element) -> Self;
    fn from_ieee_parts(real: f64, imag: f64) -> Self;
    fn epsilon() -> f64;
    async fn allocate(
        shape: [usize; 2],
        receipt: Option<&HostAllocationReceipt>,
        runtime: &RuntimeContext,
    ) -> Result<NativeDenseFactorMatrix, DenseFactorError>;
}
macro_rules! real {
    ($ty:ty,$dtype:ident,$read:ident,$write:ident,$construct:ident) => {
        impl DenseScalar for $ty {
            const DTYPE: NumericDType = NumericDType::$dtype;
            const COMPLEX: bool = false;
            fn elements(m: &NativeDenseFactorMatrix) -> Option<&[Self]> {
                if let NativeDenseFactorMatrix::Real(t) = m {
                    t.$read()
                } else {
                    None
                }
            }
            fn elements_mut(m: &mut NativeDenseFactorMatrix) -> Option<&mut [Self]> {
                if let NativeDenseFactorMatrix::Real(t) = m {
                    t.$write()
                } else {
                    None
                }
            }
            fn faer(v: &[Self]) -> &[Self::Element] {
                v
            }
            fn faer_mut(v: &mut [Self]) -> &mut [Self::Element] {
                v
            }
            fn from_element(v: Self::Element) -> Self {
                v
            }
            fn from_ieee_parts(real: f64, _imag: f64) -> Self {
                real as $ty
            }
            fn epsilon() -> f64 {
                f64::from(<$ty>::EPSILON)
            }
            async fn allocate(
                shape: [usize; 2],
                receipt: Option<&HostAllocationReceipt>,
                runtime: &RuntimeContext,
            ) -> Result<NativeDenseFactorMatrix, DenseFactorError> {
                let t = if let Some(r) = receipt {
                    r.exact_output_tensor(runtime, Self::DTYPE, &shape)?
                        .into_tensor()?
                } else {
                    let len = shape[0]
                        .checked_mul(shape[1])
                        .ok_or(HostAllocationError::Overflow)?;
                    Tensor::$construct(
                        unbounded_vec::<Self>(len, Some(runtime)).await?,
                        shape.to_vec(),
                    )
                    .map_err(DenseFactorError::InvalidData)?
                };
                Ok(NativeDenseFactorMatrix::Real(t))
            }
        }
    };
}
real!(
    f32,
    F32,
    as_f32_slice,
    try_as_f32_slice_mut_unique,
    from_f32
);
real!(f64, F64, as_f64_slice, try_as_f64_slice_mut_unique, new);
macro_rules! complex {
    ($ty:ty,$dtype:ident,$read:ident,$write:ident,$shared:ident,$unique:ident,$storage:ident) => {
        impl DenseScalar for ComplexElement<$ty> {
            const DTYPE: NumericDType = NumericDType::$dtype;
            const COMPLEX: bool = true;
            fn elements(m: &NativeDenseFactorMatrix) -> Option<&[Self]> {
                if let NativeDenseFactorMatrix::Complex(t) = m {
                    t.$read()
                } else {
                    None
                }
            }
            fn elements_mut(m: &mut NativeDenseFactorMatrix) -> Option<&mut [Self]> {
                if let NativeDenseFactorMatrix::Complex(t) = m {
                    t.$write()
                } else {
                    None
                }
            }
            fn faer(v: &[Self]) -> &[Self::Element] {
                super::super::native_view::$shared(v)
            }
            fn faer_mut(v: &mut [Self]) -> &mut [Self::Element] {
                super::super::native_view::$unique(v)
            }
            fn from_element(v: Self::Element) -> Self {
                ComplexElement(v.re, v.im)
            }
            fn from_ieee_parts(real: f64, imag: f64) -> Self {
                ComplexElement(real as $ty, imag as $ty)
            }
            fn epsilon() -> f64 {
                f64::from(<$ty>::EPSILON)
            }
            async fn allocate(
                shape: [usize; 2],
                receipt: Option<&HostAllocationReceipt>,
                runtime: &RuntimeContext,
            ) -> Result<NativeDenseFactorMatrix, DenseFactorError> {
                let t = if let Some(r) = receipt {
                    r.with_output_allocation(runtime, |p| {
                        ExactHostComplexTensorAllocation::new(Self::DTYPE, &shape, p)
                    })?
                    .into_tensor()?
                } else {
                    let len = shape[0]
                        .checked_mul(shape[1])
                        .ok_or(HostAllocationError::Overflow)?;
                    let values = unbounded_vec::<Self>(len, Some(runtime)).await?;
                    ComplexTensor::from_complex_storage(
                        ComplexStorage::$storage(runmat_value::HostComplexBuffer::from_elements(
                            values,
                        )),
                        shape.to_vec(),
                    )
                    .map_err(DenseFactorError::InvalidData)?
                };
                Ok(NativeDenseFactorMatrix::Complex(t))
            }
        }
    };
}
complex!(
    f32,
    F32,
    as_f32_slice,
    try_as_f32_slice_mut_unique,
    complex_f32,
    complex_f32_mut,
    F32
);
complex!(
    f64,
    F64,
    as_f64_slice,
    try_as_f64_slice_mut_unique,
    complex_f64,
    complex_f64_mut,
    F64
);

pub(super) fn matrix_bytes<T: DenseScalar>(shape: [usize; 2]) -> Result<u64, HostAllocationError> {
    if T::COMPLEX {
        HostAllocationPlan::retained_complex_tensor_bytes(T::DTYPE, &shape)
    } else {
        HostAllocationPlan::retained_tensor_bytes(T::DTYPE, &shape)
    }
}
pub(super) fn index_bytes(len: usize) -> Result<u64, HostAllocationError> {
    let raw = ExactHostIndexBufferAllocation::requested_bytes(len)?;
    HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
        ExactHostIndexBufferAllocation::retained_claim_count(),
    )
}
pub(super) enum IndexOwner {
    Managed(ExactHostIndexBufferAllocation),
    Unbounded(Vec<usize>),
}
impl IndexOwner {
    pub async fn new(
        len: usize,
        receipt: Option<&HostAllocationReceipt>,
        runtime: &RuntimeContext,
    ) -> Result<Self, DenseFactorError> {
        Ok(if let Some(r) = receipt {
            Self::Managed(
                r.with_output_allocation(runtime, |p| ExactHostIndexBufferAllocation::new(len, p))?,
            )
        } else {
            Self::Unbounded(unbounded_vec(len, Some(runtime)).await?)
        })
    }
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        match self {
            Self::Managed(v) => v.as_mut_slice(),
            Self::Unbounded(v) => v,
        }
    }
    pub fn finish(self) -> Result<HostIndexBuffer, DenseFactorError> {
        match self {
            Self::Managed(v) => v.into_buffer_fallible().map_err(DenseFactorError::Numeric),
            Self::Unbounded(v) => Ok(HostIndexBuffer::new(v)),
        }
    }
}
pub(super) enum Scratch {
    Managed(AlignedHostWorkspace),
    Unbounded(MemBuffer),
}
impl Scratch {
    pub fn new(
        req: StackReq,
        receipt: Option<&HostAllocationReceipt>,
        runtime: &RuntimeContext,
    ) -> Result<Self, DenseFactorError> {
        Ok(if let Some(r) = receipt {
            Self::Managed(AlignedHostWorkspace::new(
                req.layout().map_err(|_| HostAllocationError::Overflow)?,
                r,
                runtime,
            )?)
        } else {
            Self::Unbounded(MemBuffer::new(req))
        })
    }
    pub fn bytes(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        match self {
            Self::Managed(v) => v.as_mut_uninit(),
            Self::Unbounded(v) => v,
        }
    }
}
pub(super) fn scratch_bytes(req: StackReq) -> Result<u64, HostAllocationError> {
    AlignedHostWorkspace::quoted_bytes(req.layout().map_err(|_| HostAllocationError::Overflow)?)
}
pub(super) async fn run_owned<R: crate::execution::OwnedCpuPayload>(
    runtime: &RuntimeContext,
    receipt: Option<&HostAllocationReceipt>,
    body: impl FnOnce() -> R + crate::execution::OwnedCpuPayload,
) -> Result<R, DenseFactorError> {
    if let Some(r) = receipt {
        r.checkpoint(runtime)?;
    }
    let result = if runtime.service_ports().blocking_cpu().is_none()
        && runtime.execution().cpu_cooperation_policy() == CpuCooperationPolicy::NeverYield
    {
        body()
    } else {
        runtime
            .cpu_cooperation()
            .run_owned_blocking_in_context(
                runtime,
                runtime.service_ports().blocking_cpu().map(|s| s.as_ref()),
                receipt.map(HostAllocationReceipt::retain_scratch_for_cpu_work),
                body,
            )
            .await
            .map_err(DenseFactorError::Execution)?
    };
    if let Some(r) = receipt {
        r.checkpoint(runtime)?;
    }
    Ok(result)
}
