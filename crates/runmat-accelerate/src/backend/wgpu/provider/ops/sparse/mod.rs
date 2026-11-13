use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{NumericElementType, WgpuBufferOwner, WgpuEmission};

use crate::backend::wgpu::dispatch::observation::ProfiledEncoder;
use crate::backend::wgpu::pipelines::PipelineBundle;

mod cg;
mod contract;
mod jacobi;
mod jacobi_kernels;
mod multigrid;
mod pipelines;
mod spmv;
mod storage;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub(super) use cg::execute as execute_krylov;
pub(super) use contract::{capabilities, cost, feasibility};
pub(super) use jacobi::{apply as apply_preconditioner, prepare as prepare_preconditioner};
pub(in crate::backend::wgpu::provider) use multigrid::MultigridState;
pub(super) use multigrid::{
    free as free_multigrid, prepare as prepare_multigrid, solve as solve_multigrid,
};
pub(super) use pipelines::SparsePipelines;
pub(super) use spmv::execute as execute_spmv;
pub(super) use storage::SparseState;

fn numeric_element_type(precision: super::NumericPrecision) -> NumericElementType {
    match precision {
        super::NumericPrecision::F32 => NumericElementType::F32,
        super::NumericPrecision::F64 => NumericElementType::F64,
    }
}

fn storage_binding_limit(provider: &super::WgpuProvider) -> u64 {
    provider.adapter_limits.max_buffer_size.min(u64::from(
        provider.adapter_limits.max_storage_buffer_binding_size,
    ))
}

/// One workgroup per `workgroup_config.scalar` rows, within the adapter limit.
fn dispatch_groups(provider: &super::WgpuProvider, count: usize) -> anyhow::Result<u32> {
    let workgroup_size = provider.workgroup_config.scalar.max(1);
    let groups = u32::try_from(count)
        .ok()
        .and_then(|count| count.checked_add(workgroup_size - 1))
        .map(|count| count / workgroup_size)
        .ok_or_else(|| anyhow!("sparse dispatch count exceeds WGPU limits"))?;
    ensure!(
        groups <= provider.adapter_limits.max_compute_workgroups_per_dimension,
        "sparse dispatch exceeds the adapter workgroup limit"
    );
    Ok(groups)
}

/// Bind `buffers` in binding order and record one compute pass.
fn dispatch(
    emission: &WgpuEmission<'_>,
    encoder: &mut ProfiledEncoder<'_>,
    pipeline: &PipelineBundle,
    label: &'static str,
    buffers: &[&wgpu::Buffer],
    groups: u32,
) -> anyhow::Result<()> {
    let compiled = pipeline.get(emission)?;
    let entries = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();
    let bind_group = emission
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &compiled.layout,
            entries: &entries,
        });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(&compiled.pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
    Ok(())
}

/// A host-written buffer with `usage`; empty contents occupy `minimum_size`
/// zero bytes so the buffer remains bindable.
fn initialized_buffer(
    provider: &super::WgpuProvider,
    emission: &WgpuEmission<'_>,
    label: &str,
    bytes: &[u8],
    minimum_size: u64,
    usage: wgpu::BufferUsages,
) -> anyhow::Result<WgpuBufferOwner> {
    let zeroes = [0_u8; 8];
    let minimum_size = usize::try_from(minimum_size)
        .map_err(|_| anyhow!("{label} minimum allocation exceeds the host address space"))?;
    ensure!(
        (1..=zeroes.len()).contains(&minimum_size),
        "{label} has an unsupported minimum allocation"
    );
    let contents = if bytes.is_empty() {
        &zeroes[..minimum_size]
    } else {
        bytes
    };
    ensure!(
        contents.len() as u64 <= storage_binding_limit(provider),
        "{label} exceeds the WGPU storage-buffer binding limit"
    );
    emission.ensure_device(&provider.access)?;
    let buffer = WgpuBufferOwner::new(emission.create_initialized_buffer(
        &wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        },
    ));
    provider
        .telemetry
        .record_upload_bytes(contents.len() as u64);
    Ok(buffer)
}

/// Read-only operand storage uploaded once and retained by sparse handles.
fn upload_operand(
    provider: &super::WgpuProvider,
    emission: &WgpuEmission<'_>,
    label: &str,
    bytes: &[u8],
    minimum_size: u64,
) -> anyhow::Result<WgpuBufferOwner> {
    initialized_buffer(
        provider,
        emission,
        label,
        bytes,
        minimum_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    )
}

/// Device-written status or state copied back to the host at checkpoints.
fn readback_source(
    provider: &super::WgpuProvider,
    emission: &WgpuEmission<'_>,
    label: &str,
    bytes: &[u8],
) -> anyhow::Result<WgpuBufferOwner> {
    initialized_buffer(
        provider,
        emission,
        label,
        bytes,
        4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )
}

/// Copy `size` bytes of `source` into a fresh mappable staging buffer.
fn stage_readback(
    emission: &WgpuEmission<'_>,
    encoder: &mut ProfiledEncoder<'_>,
    source: &wgpu::Buffer,
    size: u64,
    label: &str,
) -> WgpuBufferOwner {
    let staging = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    ));
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    staging
}
