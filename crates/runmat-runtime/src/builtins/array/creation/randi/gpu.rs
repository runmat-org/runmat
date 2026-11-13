//! Native-class GPU Like generation from the independent GPU child stream.
//! Host arithmetic and protected upload are one preselected, admitted route;
//! provider failure never reruns generation on the CPU language stream.

use super::{builtin_error, integer_storage_from_values, integer_tensor_view, Bounds};
use crate::builtins::common::{
    gpu_helpers::host_upload::GpuHostUpload, random, random::gpu_effect::after_stream_commit,
};
use crate::context::host_allocation::{HostAllocationPlan, HostNumericWorkspace};
use crate::context::host_presentation::select_prepared_host_upload;
use crate::context::provider_operation::ExactProviderOrHostSelection;
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::random::GpuBoundedIntegerDomain;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
    NumericElementType,
};
use runmat_execution::Digest;
use runmat_value::{IntegerStorage, NumericDType, NumericScalar, Value};

enum NativeSamples {
    Exact(HostNumericWorkspace),
    F64(Vec<f64>),
    F32(Vec<f32>),
    Integer(IntegerStorage),
}

impl NativeSamples {
    fn view<'a>(&'a self, shape: &'a [usize]) -> HostNumericTensorView<'a> {
        match self {
            Self::Exact(workspace) => HostNumericTensorView {
                data: workspace.data_view(),
                shape,
                storage: GpuTensorStorage::Real,
            },
            Self::F64(values) => HostNumericTensorView {
                data: HostNumericDataView::F64(values),
                shape,
                storage: GpuTensorStorage::Real,
            },
            Self::F32(values) => HostNumericTensorView {
                data: HostNumericDataView::F32(values),
                shape,
                storage: GpuTensorStorage::Real,
            },
            Self::Integer(storage) => integer_tensor_view(storage, shape).into(),
        }
    }
}

fn dtype(class: NumericElementType) -> NumericDType {
    match class {
        NumericElementType::F64 => NumericDType::F64,
        NumericElementType::F32 => NumericDType::F32,
        NumericElementType::I8 => NumericDType::I8,
        NumericElementType::I16 => NumericDType::I16,
        NumericElementType::I32 => NumericDType::I32,
        NumericElementType::I64 => NumericDType::I64,
        NumericElementType::U8 => NumericDType::U8,
        NumericElementType::U16 => NumericDType::U16,
        NumericElementType::U32 => NumericDType::U32,
        NumericElementType::U64 => NumericDType::U64,
    }
}

fn native_scalar(value: i128, class: NumericElementType) -> NumericScalar {
    match class {
        NumericElementType::F64 => NumericScalar::F64(value as f64),
        NumericElementType::F32 => NumericScalar::F32(value as f32),
        NumericElementType::I8 => {
            NumericScalar::I8(value.clamp(i8::MIN as i128, i8::MAX as i128) as i8)
        }
        NumericElementType::I16 => {
            NumericScalar::I16(value.clamp(i16::MIN as i128, i16::MAX as i128) as i16)
        }
        NumericElementType::I32 => {
            NumericScalar::I32(value.clamp(i32::MIN as i128, i32::MAX as i128) as i32)
        }
        NumericElementType::I64 => {
            NumericScalar::I64(value.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
        }
        NumericElementType::U8 => NumericScalar::U8(value.clamp(0, u8::MAX as i128) as u8),
        NumericElementType::U16 => NumericScalar::U16(value.clamp(0, u16::MAX as i128) as u16),
        NumericElementType::U32 => NumericScalar::U32(value.clamp(0, u32::MAX as i128) as u32),
        NumericElementType::U64 => NumericScalar::U64(value.clamp(0, u64::MAX as i128) as u64),
    }
}

async fn sample(
    bounds: Bounds,
    len: usize,
    class: NumericElementType,
    upload: &GpuHostUpload<'_>,
) -> BuiltinResult<NativeSamples> {
    let domain = GpuBoundedIntegerDomain::new(bounds.lower, bounds.span)
        .ok_or_else(|| builtin_error("randi: invalid GPU integer range"))?;
    let mut cooperation = upload
        .runtime()
        .map(crate::context::RuntimeContext::cpu_cooperation);
    if let Some(budget) = cooperation.as_mut() {
        budget.checkpoint(0).await.map_err(|source| {
            build_runtime_error(format!("randi: {source}"))
                .with_builtin("randi")
                .with_source(source)
                .build()
        })?;
    }
    let mut exact = upload.sample_workspace(dtype(class), len)?;
    let mut values = Vec::new();
    if exact.is_none() {
        values
            .try_reserve_exact(len)
            .map_err(|source| builtin_error(format!("randi: cannot reserve samples: {source}")))?;
    }
    // Allocate the temporary under the admitted host receipt before a range
    // is committed. Every subsequent error carries the committed effect.
    let reservation = random::gpu::reserve(len, "randi")?;
    let sampled = async move {
        let mut pending_work = 0_u64;
        for index in 0..len {
            let item =
                u64::try_from(index).map_err(|_| builtin_error("randi: stream index overflow"))?;
            let mut attempt = 0_u64;
            let value = loop {
                let accepted = domain
                    .sample_attempt(reservation, item, attempt)
                    .ok_or_else(|| builtin_error("randi: stream attempt overflow"))?;
                pending_work += 1;
                if pending_work == 512 {
                    if let Some(budget) = cooperation.as_mut() {
                        budget.checkpoint(pending_work).await.map_err(|source| {
                            build_runtime_error(format!("randi: {source}"))
                                .with_builtin("randi")
                                .with_source(source)
                                .build()
                        })?;
                    }
                    pending_work = 0;
                }
                if let Some(value) = accepted {
                    break value;
                }
                attempt = attempt
                    .checked_add(1)
                    .ok_or_else(|| builtin_error("randi: stream attempt overflow"))?;
            };
            if let Some(workspace) = exact.as_mut() {
                workspace
                    .set_value(index, native_scalar(value, class))
                    .map_err(|source| builtin_error(format!("randi: sample store: {source}")))?;
            } else {
                values.push(value);
            }
        }
        if let Some(budget) = cooperation.as_mut() {
            budget.checkpoint(pending_work).await.map_err(|source| {
                build_runtime_error(format!("randi: {source}"))
                    .with_builtin("randi")
                    .with_source(source)
                    .build()
            })?;
        }
        if let Some(workspace) = exact {
            return Ok(NativeSamples::Exact(workspace));
        }
        match class {
            NumericElementType::F64 => Ok(NativeSamples::F64(
                values.into_iter().map(|value| value as f64).collect(),
            )),
            NumericElementType::F32 => Ok(NativeSamples::F32(
                values.into_iter().map(|value| value as f32).collect(),
            )),
            integer => {
                let class = integer
                    .integer_type()
                    .ok_or_else(|| builtin_error("randi: expected integer class"))?;
                Ok(NativeSamples::Integer(integer_storage_from_values(
                    class.into(),
                    values,
                )?))
            }
        }
    };
    sampled.await.map_err(|error| {
        if len == 0 {
            error
        } else {
            after_stream_commit("randi", error)
        }
    })
}

pub(super) async fn like(
    source: &GpuTensorHandle,
    bounds: &Bounds,
    shape: &[usize],
) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(source) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_is_logical(source)
    {
        return Err(builtin_error(
            "randi: unsupported GPU prototype representation",
        ));
    }
    let class = source.physical().element_type();
    if source.descriptor().element_type != Some(class)
        || runmat_accelerate_api::handle_class_identity(source) != Some(class.class_identity())
    {
        return Err(builtin_error(
            "randi: GPU prototype class does not match storage",
        ));
    }
    let len = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1_usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| builtin_error("randi: output extent overflow"))?
    };
    let domain = GpuBoundedIntegerDomain::new(bounds.lower, bounds.span)
        .ok_or_else(|| builtin_error("randi: invalid GPU integer range"))?;
    if let Some(output) = random::gpu_integer::try_fixed_work(source, domain, shape, class)? {
        return Ok(Value::GpuTensor(output));
    }
    // The finite route uses one exact native workspace, and the admitted
    // scratch includes its allocator/claim metadata. Unbounded legacy calls
    // may still use Vec storage but do not claim finite admission.
    let native_bytes = u64::try_from(len)
        .ok()
        .and_then(|n| n.checked_mul(u64::try_from(class.element_size()).ok()?))
        .ok_or_else(|| builtin_error("randi: sample extent overflow"))?;
    let exact_bytes = HostAllocationPlan::numeric_workspace_bytes(dtype(class), len)
        .map_err(|source| builtin_error(format!("randi: sample extent: {source}")))?;
    let compute_scratch = exact_bytes
        .checked_sub(native_bytes)
        .ok_or_else(|| builtin_error("randi: sample metadata underflow"))?;
    let host_quote =
        GpuHostUpload::quote_candidate("randi", source, shape, class, compute_scratch)?;
    let chosen = random::gpu_integer_batch::select_initial_route(
        source,
        domain,
        shape,
        class,
        host_quote.as_ref(),
    )?;
    let host_selection = match chosen {
        Some(random::gpu_integer_batch::InitialIntegerRoute {
            selection: ExactProviderOrHostSelection::Provider(selected),
            snapshot,
            physical,
        }) => {
            // The host alternative was not selected. Its query preparation
            // claim must not survive the provider continuation or consume
            // headroom needed by a later acknowledged batch.
            drop(host_quote);
            let selected = random::gpu_integer_batch::SelectedInitialNativeBatch {
                selected,
                snapshot,
                physical,
            };
            return random::gpu_integer_batch::execute_selected(
                source, domain, shape, class, selected,
            )
            .await
            .map(|output| Value::GpuTensor(output));
        }
        Some(random::gpu_integer_batch::InitialIntegerRoute {
            selection: ExactProviderOrHostSelection::HostUpload(selected),
            ..
        }) => Some(selected),
        None => {
            let host_quote = host_quote
                .as_ref()
                .ok_or_else(|| builtin_error("randi: no admitted native or upload route"))?;
            let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(source)
                .ok_or_else(|| builtin_error("randi: GPU prototype has no exact provider"))?;
            let facts = Digest::sha256_serialized(&(
                bounds.lower,
                bounds.span,
                class,
                shape,
                source.view_identity(),
            ))
            .expect("typed GPU integer host route serializes");
            select_prepared_host_upload(
                crate::context::legacy::active().as_ref(),
                provider,
                "randi",
                "random.gpu_child.integer.host.v1",
                facts,
                host_quote.candidate.clone(),
                &host_quote.query,
                host_quote.output.clone(),
                host_quote.requested_bytes,
            )?
        }
    };
    let selected_upload_bytes = host_quote
        .as_ref()
        .map(|quote| quote.requested_bytes)
        .ok_or_else(|| builtin_error("randi: host selection lacks an upload quote"))?;
    // The candidate's feasibility Vecs and their preparation claim retire
    // before the selected upload constructs a fresh, separately admitted
    // query. Keeping both would charge overlapping but unused descriptions.
    drop(host_quote);
    let mut upload = GpuHostUpload::prepare("randi", source, shape, class, compute_scratch)?;
    if let Some(selected) = host_selection {
        upload = upload.with_selected_host_route(selected, selected_upload_bytes)?;
    }
    let samples = sample(*bounds, len, class, &upload).await?;
    let output = upload.publish(&samples.view(shape)).map_err(|error| {
        if len == 0 {
            error
        } else {
            after_stream_commit("randi", error)
        }
    })?;
    Ok(Value::GpuTensor(output))
}
