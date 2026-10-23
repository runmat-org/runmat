//! GPU child-stream permutation with O(k) host scratch and one protected
//! Double upload. The host route is selected before any computation/effect.

use super::builtin_error;
use crate::builtins::common::{
    gpu_helpers::host_upload::GpuHostUpload, random, random::gpu_effect::after_stream_commit,
};
use crate::context::host_allocation::{HostAllocationPlan, HostNumericWorkspace};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::random::GpuPermutationRequest;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
    NumericElementType,
};
use runmat_value::{NumericDType, Value};

enum PermutationSamples {
    Exact(HostNumericWorkspace),
    Unbounded(Vec<f64>),
}

impl PermutationSamples {
    fn set(&mut self, index: usize, value: f64) {
        match self {
            Self::Exact(workspace) => workspace.as_f64_slice_mut().unwrap()[index] = value,
            Self::Unbounded(values) => values[index] = value,
        }
    }

    fn swap(&mut self, left: usize, right: usize) {
        match self {
            Self::Exact(workspace) => workspace.as_f64_slice_mut().unwrap().swap(left, right),
            Self::Unbounded(values) => values.swap(left, right),
        }
    }

    fn view<'a>(&'a self, shape: &'a [usize]) -> HostNumericTensorView<'a> {
        HostNumericTensorView {
            data: match self {
                Self::Exact(workspace) => workspace.data_view(),
                Self::Unbounded(values) => HostNumericDataView::F64(values),
            },
            shape,
            storage: GpuTensorStorage::Real,
        }
    }
}

async fn sample(
    n: usize,
    k: usize,
    upload: &GpuHostUpload<'_>,
) -> BuiltinResult<PermutationSamples> {
    let mut cooperation = upload
        .runtime()
        .map(crate::context::RuntimeContext::cpu_cooperation);
    if let Some(budget) = cooperation.as_mut() {
        budget.checkpoint(0).await.map_err(|source| {
            build_runtime_error(format!("randperm: {source}"))
                .with_builtin("randperm")
                .with_source(source)
                .build()
        })?;
    }
    let n = u64::try_from(n).map_err(|_| builtin_error("randperm: N exceeds GPU stream width"))?;
    let k = u64::try_from(k).map_err(|_| builtin_error("randperm: K exceeds GPU stream width"))?;
    let mut values =
        if let Some(workspace) = upload.sample_workspace(NumericDType::F64, k as usize)? {
            PermutationSamples::Exact(workspace)
        } else {
            let mut values = Vec::new();
            values.try_reserve_exact(k as usize).map_err(|source| {
                builtin_error(format!("randperm: cannot reserve result: {source}"))
            })?;
            values.resize(k as usize, 0.0);
            PermutationSamples::Unbounded(values)
        };
    let mut initialization_work = 0_u64;
    for index in 1..=k {
        values.set((index - 1) as usize, index as f64);
        initialization_work += 1;
        if initialization_work == 512 {
            if let Some(budget) = cooperation.as_mut() {
                budget
                    .checkpoint(initialization_work)
                    .await
                    .map_err(|source| {
                        build_runtime_error(format!("randperm: {source}"))
                            .with_builtin("randperm")
                            .with_source(source)
                            .build()
                    })?;
            }
            initialization_work = 0;
        }
    }
    if let Some(budget) = cooperation.as_mut() {
        budget
            .checkpoint(initialization_work)
            .await
            .map_err(|source| {
                build_runtime_error(format!("randperm: {source}"))
                    .with_builtin("randperm")
                    .with_source(source)
                    .build()
            })?;
    }
    let reservation = random::gpu::reserve(if k == 0 { 0 } else { n as usize }, "randperm")?;
    let request = GpuPermutationRequest { n, k, reservation };
    let sampled = async move {
        request
            .validate()
            .ok_or_else(|| builtin_error("randperm: invalid GPU child request"))?;
        if k == 0 {
            return Ok(values);
        }
        let mut pending_work = 0_u64;
        for total in (k + 1)..=n {
            let mut attempt = 0_u64;
            let offset = loop {
                let accepted = request
                    .reservoir_attempt(total, attempt)
                    .ok_or_else(|| builtin_error("randperm: reservoir attempt overflow"))?;
                pending_work += 1;
                if pending_work == 512 {
                    if let Some(budget) = cooperation.as_mut() {
                        budget.checkpoint(pending_work).await.map_err(|source| {
                            build_runtime_error(format!("randperm: {source}"))
                                .with_builtin("randperm")
                                .with_source(source)
                                .build()
                        })?;
                    }
                    pending_work = 0;
                }
                if let Some(offset) = accepted {
                    break offset;
                }
                attempt = attempt
                    .checked_add(1)
                    .ok_or_else(|| builtin_error("randperm: reservoir attempt overflow"))?;
            };
            if offset < k {
                values.set(offset as usize, total as f64);
            }
        }
        for index in 0..k {
            let mut attempt = 0_u64;
            let offset = loop {
                let accepted = request
                    .shuffle_attempt(index, attempt)
                    .ok_or_else(|| builtin_error("randperm: shuffle attempt overflow"))?;
                pending_work += 1;
                if pending_work == 512 {
                    if let Some(budget) = cooperation.as_mut() {
                        budget.checkpoint(pending_work).await.map_err(|source| {
                            build_runtime_error(format!("randperm: {source}"))
                                .with_builtin("randperm")
                                .with_source(source)
                                .build()
                        })?;
                    }
                    pending_work = 0;
                }
                if let Some(offset) = accepted {
                    break offset;
                }
                attempt = attempt
                    .checked_add(1)
                    .ok_or_else(|| builtin_error("randperm: shuffle attempt overflow"))?;
            };
            values.swap(index as usize, (index + offset) as usize);
        }
        if let Some(budget) = cooperation.as_mut() {
            budget.checkpoint(pending_work).await.map_err(|source| {
                build_runtime_error(format!("randperm: {source}"))
                    .with_builtin("randperm")
                    .with_source(source)
                    .build()
            })?;
        }
        Ok(values)
    };
    sampled.await.map_err(|error| {
        if k == 0 {
            error
        } else {
            after_stream_commit("randperm", error)
        }
    })
}

pub(super) async fn like(source: &GpuTensorHandle, n: usize, k: usize) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_is_logical(source)
        || runmat_accelerate_api::handle_storage(source) != GpuTensorStorage::Real
    {
        return Err(builtin_error(
            "randperm: GPU prototype must be real numeric",
        ));
    }
    let shape = [1, k];
    let exact_bytes = HostAllocationPlan::numeric_workspace_bytes(NumericDType::F64, k)
        .map_err(|source| builtin_error(format!("randperm: sample extent: {source}")))?;
    let native_bytes = u64::try_from(k)
        .ok()
        .and_then(|count| count.checked_mul(8))
        .ok_or_else(|| builtin_error("randperm: sample extent overflow"))?;
    let compute_scratch = exact_bytes
        .checked_sub(native_bytes)
        .ok_or_else(|| builtin_error("randperm: sample metadata underflow"))?;
    let mut upload = GpuHostUpload::prepare(
        "randperm",
        source,
        &shape,
        NumericElementType::F64,
        compute_scratch,
    )?;
    let values = sample(n, k, &upload).await?;
    let output = upload.publish(&values.view(&shape)).map_err(|error| {
        if k == 0 {
            error
        } else {
            after_stream_commit("randperm", error)
        }
    })?;
    Ok(Value::GpuTensor(output))
}
