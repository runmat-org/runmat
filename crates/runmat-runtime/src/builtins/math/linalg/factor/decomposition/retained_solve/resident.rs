//! Retained host computation restores its result to the original live owner.
//! This is not a device factor kernel, and never submits the original matrix
//! to an unrelated provider solve/refactor path.
use super::*;
use crate::builtins::common::gpu_helpers::{self, host_upload::GpuHostUpload};
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
    NumericElementType,
};

pub(super) async fn restore(
    result: &NativeDenseFactorMatrix,
    source: &GpuTensorHandle,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    let (shape, dtype, complex, len) = match result {
        NativeDenseFactorMatrix::Real(t) => (t.shape(), t.numeric_dtype(), false, t.len()),
        NativeDenseFactorMatrix::Complex(t) => (t.shape(), t.numeric_dtype(), true, t.len()),
    };
    let class = if dtype == NumericDType::F32 {
        NumericElementType::F32
    } else {
        NumericElementType::F64
    };
    let lanes = if complex {
        len.checked_mul(2)
            .ok_or_else(|| native::allocation(HostAllocationError::Overflow, runtime))?
    } else {
        0
    };
    let scratch = if complex {
        HostAllocationPlan::numeric_workspace_bytes(dtype, lanes)
            .map_err(|e| native::allocation(e, runtime))?
    } else {
        0
    };
    let storage = if complex {
        GpuTensorStorage::ComplexInterleaved
    } else {
        GpuTensorStorage::Real
    };
    let mut upload = GpuHostUpload::prepare_borrowed_native_output_in_context(
        runtime, NAME, source, shape, class, storage, scratch,
    )?;
    let output = match result {
        NativeDenseFactorMatrix::Real(t) => upload.publish(&HostNumericTensorView {
            data: gpu_helpers::tensor_numeric_view(t),
            shape,
            storage,
        })?,
        NativeDenseFactorMatrix::Complex(t) => {
            let mut workspace = upload.sample_workspace(dtype, lanes)?;
            let mut work = OperatorWork::new(runtime).map_err(|e| work_error(e, runtime))?;
            work.flush().await.map_err(|e| work_error(e, runtime))?;
            macro_rules! interleave {
                ($real:ty,$read:ident,$write:ident,$variant:ident) => {{
                    let pairs = t.$read().expect("native complex output");
                    let mut ordinary = Vec::<$real>::new();
                    if workspace.is_none() {
                        ordinary.try_reserve_exact(lanes).map_err(|_| {
                            native::allocation(
                                HostAllocationError::Numeric(
                                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                                ),
                                runtime,
                            )
                        })?;
                    }
                    for (index, pair) in pairs.iter().enumerate() {
                        upload.checkpoint()?;
                        if let Some(workspace) = &mut workspace {
                            let values = workspace.$write().expect("native upload workspace");
                            values[2 * index] = pair.0;
                            values[2 * index + 1] = pair.1;
                        } else {
                            ordinary.push(pair.0);
                            ordinary.push(pair.1);
                        }
                        work.tick().await.map_err(|e| work_error(e, runtime))?;
                    }
                    work.flush().await.map_err(|e| work_error(e, runtime))?;
                    let values = if let Some(workspace) = &mut workspace {
                        workspace.$write().expect("native upload workspace")
                    } else {
                        ordinary.as_mut_slice()
                    };
                    upload.publish(&HostNumericTensorView {
                        data: HostNumericDataView::$variant(values),
                        shape,
                        storage,
                    })?
                }};
            }
            if dtype == NumericDType::F32 {
                interleave!(f32, as_f32_slice, as_f32_slice_mut, F32)
            } else {
                interleave!(f64, as_f64_slice, as_f64_slice_mut, F64)
            }
        }
    };
    gpu_helpers::resident_gpu_value(output)
}
