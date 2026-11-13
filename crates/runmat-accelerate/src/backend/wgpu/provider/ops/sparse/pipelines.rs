use runmat_accelerate_api::WgpuDeviceAccess;

use crate::backend::wgpu::bindings::{storage_read_entry, storage_read_write_entry};
use crate::backend::wgpu::pipelines::{create_pipeline, PipelineBundle};
use crate::backend::wgpu::types::NumericPrecision;

use super::cg::pipelines::SparseCgPipelines;
use super::{jacobi_kernels, spmv};

/// Sparse kernels for one device and precision, compiled on first use.
pub(crate) struct SparsePipelines {
    pub(super) spmv: PipelineBundle,
    pub(super) jacobi_prepare: PipelineBundle,
    pub(super) jacobi_apply: PipelineBundle,
    pub(super) cg: SparseCgPipelines,
}

impl SparsePipelines {
    pub(crate) fn new(
        access: &WgpuDeviceAccess,
        precision: NumericPrecision,
        workgroup_size: u32,
    ) -> Self {
        let scalar = precision.as_str();
        let finite_limit = finite_limit(precision);
        Self {
            spmv: bundle(
                access,
                "runmat-sparse-spmv",
                &[false, false, false, false, true],
                spmv::shader_source(scalar, workgroup_size),
            ),
            jacobi_prepare: bundle(
                access,
                "runmat-sparse-jacobi-prepare",
                &[false, false, false, true, true],
                jacobi_kernels::prepare_shader(scalar, finite_limit, workgroup_size),
            ),
            jacobi_apply: bundle(
                access,
                "runmat-sparse-jacobi-apply",
                &[false, false, true],
                jacobi_kernels::apply_shader(scalar, workgroup_size),
            ),
            cg: SparseCgPipelines::new(access, scalar, finite_limit, workgroup_size),
        }
    }
}

/// Largest finite magnitude of the provider's scalar type, as a WGSL literal.
fn finite_limit(precision: NumericPrecision) -> &'static str {
    match precision {
        NumericPrecision::F32 => "3.402823466e38",
        NumericPrecision::F64 => "1.7976931348623157e308",
    }
}

/// One storage binding per entry, in binding order; `true` marks read-write.
pub(super) fn bundle(
    access: &WgpuDeviceAccess,
    label: &'static str,
    writable: &[bool],
    source: String,
) -> PipelineBundle {
    let entries = writable
        .iter()
        .enumerate()
        .map(|(binding, writable)| {
            if *writable {
                storage_read_write_entry(binding as u32)
            } else {
                storage_read_entry(binding as u32)
            }
        })
        .collect();
    create_pipeline(access, label, label, label, entries, source)
}
