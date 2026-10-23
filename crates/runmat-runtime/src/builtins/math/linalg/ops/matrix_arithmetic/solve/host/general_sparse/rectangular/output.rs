//! Publish through the existing exact solve owner, or explicit Unbounded CPU
//! storage. The factor's original issuer gates both branches and all awaits.
use super::super::super::managed_output::{ManagedSolveOutput, SolveOutputElement};
use super::map_qr;
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPolicy, HostAllocationReceipt},
    RuntimeContext,
};
use crate::BuiltinResult;
use runmat_value::{ComplexElement, ComplexStorage, ComplexTensor, Tensor, Value};

pub(in crate::builtins::math::linalg::ops::matrix_arithmetic::solve::host) trait Element:
    SolveOutputElement
{
    fn zero() -> Self;
    fn ordinary(values: Vec<Self>, shape: Vec<usize>) -> Result<Value, String>;
}
impl Element for f32 {
    fn zero() -> Self {
        0.0
    }
    fn ordinary(values: Vec<Self>, shape: Vec<usize>) -> Result<Value, String> {
        Tensor::from_f32(values, shape).map(crate::builtins::common::tensor::tensor_into_value)
    }
}
impl Element for f64 {
    fn zero() -> Self {
        0.0
    }
    fn ordinary(values: Vec<Self>, shape: Vec<usize>) -> Result<Value, String> {
        Tensor::new(values, shape).map(crate::builtins::common::tensor::tensor_into_value)
    }
}
macro_rules! complex {
    ($scalar:ty,$storage:ident) => {
        impl Element for ComplexElement<$scalar> {
            fn zero() -> Self {
                Self(0.0, 0.0)
            }
            fn ordinary(values: Vec<Self>, shape: Vec<usize>) -> Result<Value, String> {
                ComplexTensor::from_complex_storage(ComplexStorage::$storage(values.into()), shape)
                    .map(crate::builtins::common::random_args::complex_tensor_into_value)
            }
        }
    };
}
complex!(f32, F32);
complex!(f64, F64);
enum Storage<S: Element> {
    Managed(S::Allocation),
    Ordinary { values: Vec<S>, shape: [usize; 2] },
}
pub(in crate::builtins::math::linalg::ops::matrix_arithmetic::solve::host) struct Output<S: Element>
{
    storage: Storage<S>,
    receipt: Option<HostAllocationReceipt>,
}
impl<S: Element> Output<S> {
    pub(in crate::builtins::math::linalg::ops::matrix_arithmetic::solve::host) async fn new(
        rows: usize,
        columns: usize,
        context: &RuntimeContext,
        original_owner: impl Fn() -> Result<(), HostAllocationError>,
    ) -> BuiltinResult<Self> {
        let check = || original_owner().map_err(|e| map_qr(e.into(), context));
        check()?;
        let shape = [rows, columns];
        let policy =
            HostAllocationReceipt::policy(Some(context)).map_err(|e| map_qr(e.into(), context))?;
        if policy == HostAllocationPolicy::Managed {
            let (allocation, receipt) =
                ManagedSolveOutput::<S>::new_guarded(&shape, context, || original_owner())
                    .map_err(|e| map_qr(e.into(), context))?
                    .into_parts();
            return Ok(Self {
                storage: Storage::Managed(allocation),
                receipt: Some(receipt),
            });
        }
        let len = rows
            .checked_mul(columns)
            .ok_or_else(|| map_qr(HostAllocationError::Overflow.into(), context))?;
        let mut values = Vec::new();
        values.try_reserve_exact(len).map_err(|_| {
            map_qr(
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                )
                .into(),
                context,
            )
        })?;
        let mut budget = context.cpu_cooperation();
        let quantum = match context.execution().cpu_cooperation_policy() {
            crate::execution::CpuCooperationPolicy::NeverYield => 256,
            crate::execution::CpuCooperationPolicy::WorkQuantum(quantum) => quantum.get(),
        };
        let mut remaining = quantum;
        while values.len() < len {
            check()?;
            let count = (len - values.len()).min(remaining.min(256) as usize);
            values.resize(values.len() + count, S::zero());
            budget
                .checkpoint(count as u64)
                .await
                .map_err(|e| map_qr(e.into(), context))?;
            remaining -= count as u64;
            if remaining == 0 {
                remaining = quantum;
            }
            check()?;
        }
        check()?;
        Ok(Self {
            storage: Storage::Ordinary { values, shape },
            receipt: None,
        })
    }
    pub(in crate::builtins::math::linalg::ops::matrix_arithmetic::solve::host) fn write(
        &mut self,
        index: usize,
        value: S,
    ) -> Result<(), HostAllocationError> {
        match &mut self.storage {
            Storage::Managed(output) => {
                S::write(output, index, value).map_err(HostAllocationError::Numeric)
            }
            Storage::Ordinary { values, .. } => {
                *values.get_mut(index).ok_or(HostAllocationError::Overflow)? = value;
                Ok(())
            }
        }
    }
    pub(in crate::builtins::math::linalg::ops::matrix_arithmetic::solve::host) fn finish(
        self,
        context: &RuntimeContext,
        original_owner: impl Fn() -> Result<(), HostAllocationError>,
    ) -> BuiltinResult<Value> {
        original_owner().map_err(|e| map_qr(e.into(), context))?;
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(context)
                .map_err(|e| map_qr(e.into(), context))?;
        }
        let value = match self.storage {
            Storage::Managed(output) => S::finish(output)
                .map_err(|e| map_qr(HostAllocationError::Numeric(e).into(), context))?,
            Storage::Ordinary { values, shape } => {
                S::ordinary(values, shape.to_vec()).map_err(|detail| {
                    let mut error = crate::runtime_descriptor_error_with_detail(
                        "mldivide",
                        &runmat_builtins::MLDIVIDE_ERROR_INTERNAL,
                        detail,
                    );
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error
                })?
            }
        };
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(context)
                .map_err(|e| map_qr(e.into(), context))?;
        }
        original_owner().map_err(|e| map_qr(e.into(), context))?;
        Ok(value)
    }
}
