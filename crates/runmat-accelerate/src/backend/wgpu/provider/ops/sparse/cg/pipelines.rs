use std::borrow::Cow;
use std::sync::Arc;

use super::super::super::WgpuProvider;
use super::shaders;

pub(super) struct ComputeStage {
    pub(super) layout: Arc<wgpu::BindGroupLayout>,
    pub(super) pipeline: wgpu::ComputePipeline,
}

pub(crate) struct SparseCgPipelines {
    pub(super) spmv: ComputeStage,
    pub(super) initialize_vectors: ComputeStage,
    pub(super) dot_partials: ComputeStage,
    pub(super) dot_finalize: ComputeStage,
    pub(super) initialize_state: ComputeStage,
    pub(super) alpha: ComputeStage,
    pub(super) update_solution_residual: ComputeStage,
    pub(super) apply_preconditioner: ComputeStage,
    pub(super) update_state: ComputeStage,
    pub(super) update_direction: ComputeStage,
}

pub(super) fn get(provider: &WgpuProvider) -> Arc<SparseCgPipelines> {
    let mut cache = provider
        .device_caches
        .sparse_pipelines
        .cg
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    cache
        .get_or_insert_with(|| Arc::new(create(provider)))
        .clone()
}

fn create(provider: &WgpuProvider) -> SparseCgPipelines {
    let finite_limit = match provider.precision {
        super::super::super::NumericPrecision::F32 => "3.402823466e38",
        super::super::super::NumericPrecision::F64 => "1.7976931348623157e308",
    };
    let sources = shaders::sources(
        provider.precision.as_str(),
        finite_limit,
        provider.workgroup_config.scalar.max(1),
    );
    SparseCgPipelines {
        spmv: stage(
            provider,
            "runmat-sparse-cg-spmv",
            &[false, false, false, false, true, false],
            sources.spmv,
        ),
        initialize_vectors: stage(
            provider,
            "runmat-sparse-cg-initialize-vectors",
            &[false, false, false, true, true, true, false],
            sources.initialize_vectors,
        ),
        dot_partials: stage(
            provider,
            "runmat-sparse-cg-dot-partials",
            &[false, false, true, false],
            sources.dot_partials,
        ),
        dot_finalize: stage(
            provider,
            "runmat-sparse-cg-dot-finalize",
            &[false, true, false],
            sources.dot_finalize,
        ),
        initialize_state: stage(
            provider,
            "runmat-sparse-cg-initialize-state",
            &[false, false, true],
            sources.initialize_state,
        ),
        alpha: stage(
            provider,
            "runmat-sparse-cg-alpha",
            &[false, true],
            sources.alpha,
        ),
        update_solution_residual: stage(
            provider,
            "runmat-sparse-cg-update-solution-residual",
            &[true, true, false, false, false],
            sources.update_solution_residual,
        ),
        apply_preconditioner: stage(
            provider,
            "runmat-sparse-cg-apply-preconditioner",
            &[false, false, true, false],
            sources.apply_preconditioner,
        ),
        update_state: stage(
            provider,
            "runmat-sparse-cg-update-state",
            &[false, false, true],
            sources.update_state,
        ),
        update_direction: stage(
            provider,
            "runmat-sparse-cg-update-direction",
            &[false, true, false],
            sources.update_direction,
        ),
    }
}

fn stage(
    provider: &WgpuProvider,
    label: &'static str,
    writable: &[bool],
    source: String,
) -> ComputeStage {
    let entries = writable
        .iter()
        .enumerate()
        .map(|(binding, writable)| {
            if *writable {
                crate::backend::wgpu::bindings::storage_read_write_entry(binding as u32)
            } else {
                crate::backend::wgpu::bindings::storage_read_entry(binding as u32)
            }
        })
        .collect::<Vec<_>>();
    let layout = Arc::new(provider.device_ref().create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        },
    ));
    let pipeline_layout =
        provider
            .device_ref()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout.as_ref()],
                push_constant_ranges: &[],
            });
    let module = provider
        .device_ref()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
    let pipeline = provider.device_ref().create_compute_pipeline(
        &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor!(
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
        ),
    );
    ComputeStage { layout, pipeline }
}
