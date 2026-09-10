use std::sync::Arc;

use anyhow::{anyhow, ensure};
use runmat_accelerate_api::NumericElementType;

mod cg;
mod contract;
mod jacobi;
mod jacobi_kernels;
mod pipelines;
mod spmv;
mod storage;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub(super) use cg::execute as execute_krylov;
pub(super) use contract::{capabilities, cost, feasibility};
pub(super) use jacobi::{apply as apply_preconditioner, prepare as prepare_preconditioner};
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

fn upload_u32(
    provider: &super::WgpuProvider,
    label: &str,
    values: &[u32],
) -> anyhow::Result<Arc<wgpu::Buffer>> {
    upload_bytes(provider, label, bytemuck::cast_slice(values), 4)
}

fn upload_bytes(
    provider: &super::WgpuProvider,
    label: &str,
    bytes: &[u8],
    minimum_size: u64,
) -> anyhow::Result<Arc<wgpu::Buffer>> {
    use wgpu::util::DeviceExt;

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
    let binding_limit = storage_binding_limit(provider);
    ensure!(
        contents.len() as u64 <= binding_limit,
        "{label} exceeds the WGPU storage-buffer binding limit"
    );
    let buffer = Arc::new(provider.device_ref().create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    ));
    provider
        .telemetry
        .record_upload_bytes(contents.len() as u64);
    Ok(buffer)
}
