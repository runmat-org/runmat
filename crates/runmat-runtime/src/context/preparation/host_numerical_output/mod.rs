use super::RuntimeContext;
use crate::builtins::common::spec::FusionHostOutputDomain;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_builtins::BuiltinPreparationOperation;
use runmat_types::{NumericClass, NumericDomain};
use runmat_value::{ComplexTensor, ExactHostComplexTensorAllocation, NumericDType, Tensor, Value};

mod cast;
mod promoting;

/// A selected host numerical operation's physically allocated output.
///
/// The preparation service constructs this owner before its first readback
/// effect. Consuming it in the runtime body preserves the payload and shape
/// claims in the returned tensor without materializing an intermediate Vec.
pub struct PreparedHostNumericalOutput {
    operation: BuiltinPreparationOperation,
    dtype: NumericDType,
    domain: FusionHostOutputDomain,
    len: usize,
    allocation: NumericalAllocation,
    broadcast: Option<crate::builtins::common::broadcast::PreparedBroadcastPlan>,
    receipt: HostAllocationReceipt,
}

struct NumericalAllocation {
    real: Option<Tensor>,
    complex: Option<ComplexTensor>,
}

impl PreparedHostNumericalOutput {
    const COPY_CHUNK: usize = 4096;

    /// Prepare only operation-local scratch for an already-admitted real
    /// whole-Tensor slot. The slot is bound at the source-ordered VM take;
    /// its payload and shape claims are never charged a second time here.
    pub fn prepare_recycling_work(
        context: &RuntimeContext,
        resources: &super::AdmittedHostResources,
        slot_dtype: NumericDType,
        slot_shape: &[usize],
    ) -> Result<Self, HostAllocationError> {
        let prepared_scratch_bytes = validated_prepared_scratch(resources)?;
        let dtype = match resources.output_class {
            NumericClass::Single => NumericDType::F32,
            NumericClass::Double => NumericDType::F64,
            _ => return Err(unsupported_recycling_class()),
        };
        if resources.output_domain != FusionHostOutputDomain::Exact(NumericDomain::Real)
            || slot_dtype != dtype
            || slot_shape != resources.output_shape.as_slice()
            || resources.bound.retained_bytes
                != HostAllocationPlan::retained_tensor_bytes(dtype, slot_shape)?
        {
            return Err(unsupported_recycling_class());
        }
        let Some(receipt) = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(prepared_scratch_bytes, 0)?,
        )?
        else {
            return Err(HostAllocationError::MissingManagedOwner);
        };
        let broadcast = match resources.operation {
            BuiltinPreparationOperation::ElementwiseBinary
            | BuiltinPreparationOperation::RealPower => {
                let [left, right] = resources.operand_shapes.as_slice() else {
                    return Err(unsupported_recycling_class());
                };
                Some(
                    crate::builtins::common::broadcast::PreparedBroadcastPlan::prepare(
                        &receipt,
                        context,
                        left,
                        right,
                        &resources.output_shape,
                    )?,
                )
            }
            _ => None,
        };
        receipt.checkpoint(context)?;
        Ok(Self {
            operation: resources.operation,
            dtype,
            domain: resources.output_domain,
            len: allocation_len(slot_shape)?,
            allocation: NumericalAllocation {
                real: None,
                complex: None,
            },
            broadcast,
            receipt,
        })
    }

    pub(crate) fn bind_recycled_tensor(mut self, tensor: Tensor) -> Result<Self, Tensor> {
        if self.allocation.real.is_some()
            || self.allocation.complex.is_some()
            || tensor.numeric_dtype() != self.dtype
            || tensor.len() != self.len
        {
            return Err(tensor);
        }
        self.allocation.real = Some(tensor);
        Ok(self)
    }

    /// Prepare the exact selected numerical result before any provider
    /// readback. `RealOrComplex` reserves both possible physical results: the
    /// semantic kernel may inspect values to choose a domain, but it cannot
    /// discover an allocation failure after effects have begun.
    pub fn prepare_admitted(
        context: &RuntimeContext,
        resources: &super::AdmittedHostResources,
    ) -> Result<Option<Self>, HostAllocationError> {
        let prepared_scratch_bytes = validated_prepared_scratch(resources)?;
        let dtype = match resources.output_class {
            NumericClass::Single => NumericDType::F32,
            NumericClass::Double => NumericDType::F64,
            _ => {
                return Err(HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::UnsupportedClass,
                ))
            }
        };
        let Some(receipt) = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(prepared_scratch_bytes, resources.bound.retained_bytes)?,
        )?
        else {
            return Ok(None);
        };
        let allocation = match resources.output_domain {
            FusionHostOutputDomain::Exact(NumericDomain::Real) => NumericalAllocation {
                real: Some(
                    receipt
                        .exact_output_tensor(context, dtype, &resources.output_shape)?
                        .into_tensor()
                        .map_err(HostAllocationError::Numeric)?,
                ),
                complex: None,
            },
            FusionHostOutputDomain::Exact(NumericDomain::Complex) => NumericalAllocation {
                real: None,
                complex: Some(
                    receipt
                        .with_output_allocation(context, |permit| {
                            ExactHostComplexTensorAllocation::new(
                                dtype,
                                &resources.output_shape,
                                permit,
                            )
                        })?
                        .into_tensor()
                        .map_err(HostAllocationError::Numeric)?,
                ),
            },
            FusionHostOutputDomain::RealOrComplex => {
                let real = receipt
                    .exact_output_tensor(context, dtype, &resources.output_shape)?
                    .into_tensor()
                    .map_err(HostAllocationError::Numeric)?;
                let complex = receipt
                    .with_output_allocation(context, |permit| {
                        ExactHostComplexTensorAllocation::new(
                            dtype,
                            &resources.output_shape,
                            permit,
                        )
                    })?
                    .into_tensor()
                    .map_err(HostAllocationError::Numeric)?;
                NumericalAllocation {
                    real: Some(real),
                    complex: Some(complex),
                }
            }
        };
        let broadcast = match resources.operation {
            BuiltinPreparationOperation::ElementwiseBinary
            | BuiltinPreparationOperation::RealPower => {
                let [left, right] = resources.operand_shapes.as_slice() else {
                    return Err(HostAllocationError::Numeric(
                        runmat_value::ExactHostNumericAllocationError::ExceedsPermit,
                    ));
                };
                Some(
                    crate::builtins::common::broadcast::PreparedBroadcastPlan::prepare(
                        &receipt,
                        context,
                        left,
                        right,
                        &resources.output_shape,
                    )?,
                )
            }
            _ => None,
        };
        receipt.checkpoint(context)?;
        Ok(Some(Self {
            operation: resources.operation,
            dtype,
            domain: resources.output_domain,
            len: allocation_len(&resources.output_shape)?,
            allocation,
            broadcast,
            receipt,
        }))
    }

    pub fn operation(&self) -> BuiltinPreparationOperation {
        self.operation
    }

    pub fn dtype(&self) -> NumericDType {
        self.dtype
    }

    pub fn output_domain(&self) -> FusionHostOutputDomain {
        self.domain
    }

    pub fn shape(&self) -> &[usize] {
        self.allocation
            .real
            .as_ref()
            .map(Tensor::shape)
            .or_else(|| self.allocation.complex.as_ref().map(ComplexTensor::shape))
            .expect("prepared numerical output owns one physical result")
    }

    pub(crate) fn take_broadcast_plan(
        &mut self,
    ) -> Option<crate::builtins::common::broadcast::PreparedBroadcastPlan> {
        self.broadcast.take()
    }

    pub(crate) fn generate_real_f32(
        mut self,
        context: &RuntimeContext,
        mut generate: impl FnMut(usize) -> Result<f32, crate::RuntimeError>,
    ) -> Result<Value, crate::RuntimeError> {
        let (receipt, allocation) = (&self.receipt, &mut self.allocation);
        let output = real_allocation_mut(allocation)
            .map_err(|error| numerical_output_error(context, error))?
            .try_as_f32_slice_mut_unique()
            .ok_or_else(unsupported_output)?;
        for (index, output) in output.iter_mut().enumerate() {
            if index % Self::COPY_CHUNK == 0 {
                receipt
                    .checkpoint(context)
                    .map_err(|error| numerical_output_error(context, error))?;
            }
            *output = generate(index)?;
        }
        self.into_tensor(context)
            .map(Value::Tensor)
            .map_err(|error| numerical_output_error(context, error))
    }

    pub(crate) fn generate_real_f64(
        mut self,
        context: &RuntimeContext,
        mut generate: impl FnMut(usize) -> Result<f64, crate::RuntimeError>,
    ) -> Result<Value, crate::RuntimeError> {
        let (receipt, allocation) = (&self.receipt, &mut self.allocation);
        let output = real_allocation_mut(allocation)
            .map_err(|error| numerical_output_error(context, error))?
            .try_as_f64_slice_mut_unique()
            .ok_or_else(unsupported_output)?;
        for (index, output) in output.iter_mut().enumerate() {
            if index % Self::COPY_CHUNK == 0 {
                receipt
                    .checkpoint(context)
                    .map_err(|error| numerical_output_error(context, error))?;
            }
            *output = generate(index)?;
        }
        self.into_tensor(context)
            .map(Value::Tensor)
            .map_err(|error| numerical_output_error(context, error))
    }

    pub(crate) fn map_real_f32(
        mut self,
        context: &RuntimeContext,
        source: &[f32],
        mut map: impl FnMut(f32) -> Result<f32, crate::RuntimeError>,
    ) -> Result<Value, crate::RuntimeError> {
        self.ensure_source_len(source.len())?;
        let receipt = &self.receipt;
        let target = self
            .allocation
            .real
            .as_mut()
            .ok_or_else(unsupported_output)?
            .try_as_f32_slice_mut_unique()
            .ok_or_else(unsupported_output)?;
        for (offset, (target, source)) in target.iter_mut().zip(source).enumerate() {
            if offset % Self::COPY_CHUNK == 0 {
                receipt
                    .checkpoint(context)
                    .map_err(|error| numerical_output_error(context, error))?;
            }
            *target = map(*source)?;
        }
        self.into_tensor(context)
            .map(Value::Tensor)
            .map_err(|error| numerical_output_error(context, error))
    }

    pub(crate) fn map_real_f64(
        mut self,
        context: &RuntimeContext,
        source: &[f64],
        mut map: impl FnMut(f64) -> Result<f64, crate::RuntimeError>,
    ) -> Result<Value, crate::RuntimeError> {
        self.ensure_source_len(source.len())?;
        let receipt = &self.receipt;
        let target = self
            .allocation
            .real
            .as_mut()
            .ok_or_else(unsupported_output)?
            .try_as_f64_slice_mut_unique()
            .ok_or_else(unsupported_output)?;
        for (offset, (target, source)) in target.iter_mut().zip(source).enumerate() {
            if offset % Self::COPY_CHUNK == 0 {
                receipt
                    .checkpoint(context)
                    .map_err(|error| numerical_output_error(context, error))?;
            }
            *target = map(*source)?;
        }
        self.into_tensor(context)
            .map(Value::Tensor)
            .map_err(|error| numerical_output_error(context, error))
    }

    fn into_real_allocation(self) -> Result<Tensor, HostAllocationError> {
        self.allocation.real.ok_or(HostAllocationError::Numeric(
            runmat_value::ExactHostNumericAllocationError::UnsupportedClass,
        ))
    }

    fn ensure_source_len(&self, len: usize) -> Result<(), crate::RuntimeError> {
        if self.len != len {
            return Err(unsupported_output());
        }
        Ok(())
    }
}

fn validated_prepared_scratch(
    resources: &super::AdmittedHostResources,
) -> Result<u64, HostAllocationError> {
    let calculated = crate::builtins::math::host_resources::prepared_scratch_for_shapes(
        resources.operation,
        &resources.operand_shapes,
        &resources.output_shape,
    )
    .ok_or_else(unsupported_prepared_scratch)?;
    if calculated != resources.prepared_scratch_bytes || calculated > resources.bound.scratch_bytes
    {
        return Err(unsupported_prepared_scratch());
    }
    Ok(calculated)
}

fn unsupported_prepared_scratch() -> HostAllocationError {
    HostAllocationError::Numeric(runmat_value::ExactHostNumericAllocationError::ExceedsPermit)
}

fn unsupported_recycling_class() -> HostAllocationError {
    HostAllocationError::Numeric(runmat_value::ExactHostNumericAllocationError::UnsupportedClass)
}

fn allocation_len(shape: &[usize]) -> Result<usize, HostAllocationError> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or(HostAllocationError::Overflow)
}

fn real_allocation_mut(
    allocation: &mut NumericalAllocation,
) -> Result<&mut Tensor, HostAllocationError> {
    allocation.real.as_mut().ok_or(HostAllocationError::Numeric(
        runmat_value::ExactHostNumericAllocationError::UnsupportedClass,
    ))
}

fn numerical_output_error(
    context: &RuntimeContext,
    error: HostAllocationError,
) -> crate::RuntimeError {
    crate::context::host_allocation::host_allocation_runtime_error(
        Some(context),
        "RunMat:Placement:HostNumericalOutput",
        "selected host numerical output",
        error,
    )
}

fn unsupported_output() -> crate::RuntimeError {
    crate::build_runtime_error("selected host numerical output disagrees with its admitted body")
        .with_identifier("RunMat:Placement:InvalidDecision")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
