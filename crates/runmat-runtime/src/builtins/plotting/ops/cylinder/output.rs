use super::{error, RadiusProfile, NAME};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::{ExactHostTensor2dAllocation, NumericDType, Tensor};

enum CoordinateBuffer {
    Exact(ExactHostTensor2dAllocation),
    Unbounded {
        data: Vec<f64>,
        rows: usize,
        cols: usize,
    },
}

impl CoordinateBuffer {
    fn new(
        rows: usize,
        cols: usize,
        receipt: Option<&HostAllocationReceipt>,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<Self> {
        if let (Some(receipt), Some(runtime)) = (receipt, runtime) {
            return receipt
                .with_output_allocation(runtime, |permit| {
                    ExactHostTensor2dAllocation::new(NumericDType::F64, rows, cols, permit)
                })
                .map(Self::Exact)
                .map_err(|source| host_error(Some(runtime), source));
        }
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| error(2, "grid extent overflow"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|source| allocation_error("coordinate allocation", source))?;
        data.resize(len, 0.0);
        Ok(Self::Unbounded { data, rows, cols })
    }

    fn set(&mut self, index: usize, value: f64) {
        match self {
            Self::Exact(allocation) => {
                allocation.as_f64_slice_mut().expect("Double output")[index] = value
            }
            Self::Unbounded { data, .. } => data[index] = value,
        }
    }

    fn into_tensor(self) -> BuiltinResult<Tensor> {
        match self {
            Self::Exact(allocation) => allocation
                .into_tensor()
                .map_err(|source| allocation_error("coordinate publication", source)),
            Self::Unbounded { data, rows, cols } => {
                Tensor::new(data, vec![rows, cols]).map_err(|source| error(3, source))
            }
        }
    }
}

fn allocation_error(
    context: &str,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    build_runtime_error(format!("cylinder: {context}: {source}"))
        .with_builtin(NAME)
        .with_identifier("RunMat:cylinder:OutOfMemory")
        .with_source(source)
        .build()
}

fn host_error(
    runtime: Option<&crate::context::RuntimeContext>,
    source: HostAllocationError,
) -> RuntimeError {
    let cancellation = runtime.map(crate::context::RuntimeContext::cancellation_context);
    crate::builtin::builtin_control_error(cancellation.as_ref(), NAME, source)
}

fn cooperation_error(
    runtime: Option<&crate::context::RuntimeContext>,
    source: crate::execution::ExecutionServiceError,
) -> RuntimeError {
    let error = source.into_runtime_error_in_context(runtime);
    crate::builtins::common::map_control_flow_with_builtin(error, NAME)
}

pub(super) fn grid_size(rows: usize, n: usize) -> BuiltinResult<usize> {
    let cols = n.checked_add(1).ok_or_else(|| error(2, "n+1 overflowed"))?;
    rows.checked_mul(cols)
        .ok_or_else(|| error(2, "grid extent overflow"))?;
    Ok(cols)
}

pub(super) async fn make_coordinates(
    profile: &RadiusProfile,
    n: usize,
    requested: usize,
    plot: bool,
) -> BuiltinResult<(Tensor, Option<Tensor>, Option<Tensor>)> {
    let rows = profile.len();
    let cols = grid_size(rows, n)?;
    let outputs = if plot { 3 } else { requested.clamp(1, 3) };
    let one = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[rows, cols])
        .map_err(|source| allocation_error("grid geometry", source))?;
    let retained = one
        .checked_mul(outputs as u64)
        .ok_or_else(|| error(2, "retained grid extent overflow"))?;
    let runtime = crate::context::legacy::active();
    let mut cooperation = runtime
        .as_ref()
        .map(crate::context::RuntimeContext::cpu_cooperation);
    if let Some(budget) = cooperation.as_mut() {
        budget
            .checkpoint(0)
            .await
            .map_err(|source| cooperation_error(runtime.as_ref(), source))?;
    }
    let plan = HostAllocationPlan::new(0, retained)
        .map_err(|source| host_error(runtime.as_ref(), source))?;
    let receipt = HostAllocationReceipt::admit(runtime.as_ref(), plan)
        .map_err(|source| host_error(runtime.as_ref(), source))?;
    let mut x = CoordinateBuffer::new(rows, cols, receipt.as_ref(), runtime.as_ref())?;
    let mut y = if outputs >= 2 {
        Some(CoordinateBuffer::new(
            rows,
            cols,
            receipt.as_ref(),
            runtime.as_ref(),
        )?)
    } else {
        None
    };
    let mut z = if outputs >= 3 {
        Some(CoordinateBuffer::new(
            rows,
            cols,
            receipt.as_ref(),
            runtime.as_ref(),
        )?)
    } else {
        None
    };
    let mut pending = 0_u64;
    for col in 0..cols {
        let theta = if col == n {
            0.0
        } else {
            std::f64::consts::TAU * col as f64 / n as f64
        };
        let (sine, cosine) = theta.sin_cos();
        for row in 0..rows {
            let index = row + col * rows;
            let radius = profile.at(row);
            x.set(index, radius * cosine);
            if let Some(y) = y.as_mut() {
                y.set(index, radius * sine);
            }
            if let Some(z) = z.as_mut() {
                z.set(
                    index,
                    if rows <= 1 {
                        0.0
                    } else {
                        row as f64 / (rows - 1) as f64
                    },
                );
            }
            pending += 1;
            if pending == 4096 {
                if let Some(budget) = cooperation.as_mut() {
                    budget
                        .checkpoint(pending)
                        .await
                        .map_err(|source| cooperation_error(runtime.as_ref(), source))?;
                }
                pending = 0;
            }
        }
    }
    if let Some(budget) = cooperation.as_mut() {
        budget
            .checkpoint(pending)
            .await
            .map_err(|source| cooperation_error(runtime.as_ref(), source))?;
    }
    if let (Some(receipt), Some(runtime)) = (receipt.as_ref(), runtime.as_ref()) {
        receipt
            .checkpoint(runtime)
            .map_err(|source| host_error(Some(runtime), source))?;
    }
    let x = x.into_tensor()?;
    let y = y.map(CoordinateBuffer::into_tensor).transpose()?;
    let z = z.map(CoordinateBuffer::into_tensor).transpose()?;
    if let (Some(receipt), Some(runtime)) = (receipt.as_ref(), runtime.as_ref()) {
        receipt
            .checkpoint(runtime)
            .map_err(|source| host_error(Some(runtime), source))?;
    }
    Ok((x, y, z))
}
