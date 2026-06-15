#[cfg(target_arch = "wasm32")]
pub(crate) fn instance_new(desc: wgpu::InstanceDescriptor) -> wgpu::Instance {
    wgpu::Instance::new(&desc)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn instance_new(desc: wgpu::InstanceDescriptor) -> wgpu::Instance {
    wgpu::Instance::new(desc)
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn device_descriptor<'a>(
    label: Option<&'a str>,
    required_features: wgpu::Features,
    required_limits: wgpu::Limits,
) -> wgpu::DeviceDescriptor<'a> {
    wgpu::DeviceDescriptor {
        label,
        required_features,
        required_limits,
        memory_hints: wgpu::MemoryHints::default(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn device_descriptor<'a>(
    label: Option<&'a str>,
    required_features: wgpu::Features,
    required_limits: wgpu::Limits,
) -> wgpu::DeviceDescriptor<'a> {
    wgpu::DeviceDescriptor {
        label,
        required_features,
        required_limits,
    }
}

macro_rules! wgpu_compute_pipeline_descriptor {
    (
        label: $label:expr,
        layout: $layout:expr,
        module: $module:expr,
        entry_point: $entry_point:expr $(,)?
    ) => {{
        #[cfg(target_arch = "wasm32")]
        {
            wgpu::ComputePipelineDescriptor {
                label: $label,
                layout: $layout,
                module: $module,
                entry_point: Some($entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            wgpu::ComputePipelineDescriptor {
                label: $label,
                layout: $layout,
                module: $module,
                entry_point: $entry_point,
            }
        }
    }};
}
pub(crate) use wgpu_compute_pipeline_descriptor;
