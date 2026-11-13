pub fn create_shader_module(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    label: &str,
    wgsl: &str,
) -> anyhow::Result<wgpu::ShaderModule> {
    super::super::shader_compilation::create_shader_module(emission, label, wgsl, None)
}

pub fn create_pipeline_layout_single(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    label: &str,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    emission
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[bgl],
            push_constant_ranges: &[],
        })
}

pub fn create_compute_pipeline(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    emission.device().create_compute_pipeline(
        &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
            label: Some(label),
            layout: Some(layout),
            module: module,
            entry_point: "main",
        },
    )
}

pub fn create_pipeline_layout_empty(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    label: &str,
) -> wgpu::PipelineLayout {
    emission
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        })
}
