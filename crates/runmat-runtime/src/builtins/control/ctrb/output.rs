use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, ExactHostComplexTensorAllocation,
    ExactHostTensorAllocation, NumericDType, Tensor, Value,
};

use super::{admission, unsupported, BuiltinResult, RuntimeContext};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};

#[derive(Clone, Copy)]
enum OutputClass {
    RealF32,
    RealF64,
    ComplexF32,
    ComplexF64,
}

pub(super) struct OutputOwner {
    pub value: Value,
    pub receipt: Option<HostAllocationReceipt>,
}

pub(super) fn checkpoint(
    runtime: Option<&RuntimeContext>,
    receipt: Option<&HostAllocationReceipt>,
) -> BuiltinResult<()> {
    if let Some(receipt) = receipt {
        receipt
            .checkpoint_captured()
            .map_err(|error| admission(runtime, error))
    } else {
        match HostAllocationReceipt::policy(runtime).map_err(|error| admission(runtime, error))? {
            HostAllocationPolicy::Unbounded => Ok(()),
            HostAllocationPolicy::Managed => {
                Err(admission(runtime, HostAllocationError::MissingManagedOwner))
            }
        }
    }
}

fn zeros<T: Copy>(
    count: usize,
    zero: T,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<T>> {
    checkpoint(runtime, None)?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| {
        admission(
            runtime,
            HostAllocationError::Numeric(
                runmat_value::ExactHostNumericAllocationError::AllocationFailed,
            ),
        )
    })?;
    while values.len() < count {
        checkpoint(runtime, None)?;
        values.resize(values.len().saturating_add(4096).min(count), zero);
    }
    Ok(values)
}

fn allocate(
    class: OutputClass,
    rows: usize,
    cols: usize,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<OutputOwner> {
    let shape = [rows, cols];
    let (dtype, complex) = match class {
        OutputClass::RealF32 => (NumericDType::F32, false),
        OutputClass::RealF64 => (NumericDType::F64, false),
        OutputClass::ComplexF32 => (NumericDType::F32, true),
        OutputClass::ComplexF64 => (NumericDType::F64, true),
    };
    let retained = if complex {
        HostAllocationPlan::retained_complex_tensor_bytes(dtype, &shape)
    } else {
        HostAllocationPlan::retained_tensor_bytes(dtype, &shape)
    }
    .map_err(|error| admission(runtime, error))?;
    let plan = HostAllocationPlan::new(0, retained).map_err(|error| admission(runtime, error))?;
    let receipt =
        HostAllocationReceipt::admit(runtime, plan).map_err(|error| admission(runtime, error))?;
    let value = if let Some(receipt) = receipt.as_ref() {
        let runtime =
            runtime.ok_or_else(|| admission(runtime, HostAllocationError::MissingManagedOwner))?;
        receipt
            .with_output_allocation(runtime, |permit| {
                if complex {
                    ExactHostComplexTensorAllocation::new(dtype, &shape, permit)?
                        .into_tensor()
                        .map(Value::ComplexTensor)
                } else {
                    ExactHostTensorAllocation::new(dtype, &shape, permit)?
                        .into_tensor()
                        .map(Value::Tensor)
                }
            })
            .map_err(|error| admission(Some(runtime), error))?
    } else {
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
        match class {
            OutputClass::RealF32 => {
                Tensor::from_f32(zeros(count, 0.0_f32, runtime)?, shape.to_vec()).map(Value::Tensor)
            }
            OutputClass::RealF64 => {
                Tensor::new(zeros(count, 0.0_f64, runtime)?, shape.to_vec()).map(Value::Tensor)
            }
            OutputClass::ComplexF32 => ComplexTensor::from_complex_storage(
                ComplexStorage::F32(
                    zeros(count, ComplexElement(0.0_f32, 0.0_f32), runtime)?.into(),
                ),
                shape.to_vec(),
            )
            .map(Value::ComplexTensor),
            OutputClass::ComplexF64 => ComplexTensor::from_complex_storage(
                ComplexStorage::F64(
                    zeros(count, ComplexElement(0.0_f64, 0.0_f64), runtime)?.into(),
                ),
                shape.to_vec(),
            )
            .map(Value::ComplexTensor),
        }
        .map_err(|error| unsupported(format!("ctrb output shape: {error}")))?
    };
    checkpoint(runtime, receipt.as_ref())?;
    Ok(OutputOwner { value, receipt })
}

macro_rules! output_class {
    ($name:ident, $element:ty, $value:ident, $slice:ident, $finite:expr) => {
        pub(super) struct $name;
        impl $name {
            pub fn finite(value: $element) -> bool {
                ($finite)(value)
            }
            pub fn allocate(
                rows: usize,
                cols: usize,
                runtime: Option<&RuntimeContext>,
            ) -> BuiltinResult<OutputOwner> {
                allocate(OutputClass::$name, rows, cols, runtime)
            }
            pub fn values_mut(value: &mut Value) -> BuiltinResult<&mut [$element]> {
                let Value::$value(tensor) = value else {
                    return Err(unsupported(
                        "controllability result has the wrong storage class",
                    ));
                };
                tensor.$slice().ok_or_else(|| {
                    unsupported("controllability result must have unique native storage")
                })
            }
        }
    };
}

output_class!(
    RealF32,
    f32,
    Tensor,
    try_as_f32_slice_mut_unique,
    |value: f32| value.is_finite()
);
output_class!(
    RealF64,
    f64,
    Tensor,
    try_as_f64_slice_mut_unique,
    |value: f64| value.is_finite()
);
output_class!(
    ComplexF32,
    ComplexElement<f32>,
    ComplexTensor,
    try_as_f32_slice_mut_unique,
    |value: ComplexElement<f32>| value.0.is_finite() && value.1.is_finite()
);
output_class!(
    ComplexF64,
    ComplexElement<f64>,
    ComplexTensor,
    try_as_f64_slice_mut_unique,
    |value: ComplexElement<f64>| value.0.is_finite() && value.1.is_finite()
);
