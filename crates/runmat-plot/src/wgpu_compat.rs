pub(crate) type TexelCopyTextureInfo<'a> = wgpu::TexelCopyTextureInfo<'a>;

pub(crate) type TexelCopyBufferInfo<'a> = wgpu::TexelCopyBufferInfo<'a>;

pub(crate) type TexelCopyBufferLayout = wgpu::TexelCopyBufferLayout;

pub(crate) fn instance_new(desc: wgpu::InstanceDescriptor) -> wgpu::Instance {
    wgpu::Instance::new(&desc)
}

#[cfg(any(test, feature = "web", feature = "gui"))]
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

pub(crate) fn default_device_descriptor<'a>() -> wgpu::DeviceDescriptor<'a> {
    wgpu::DeviceDescriptor {
        memory_hints: wgpu::MemoryHints::default(),
        ..Default::default()
    }
}

#[cfg(feature = "egui-overlay")]
pub(crate) fn egui_renderer_new(
    device: &wgpu::Device,
    output_color_format: wgpu::TextureFormat,
    output_depth_format: Option<wgpu::TextureFormat>,
    msaa_samples: u32,
) -> egui_wgpu::Renderer {
    egui_wgpu::Renderer::new(
        device,
        output_color_format,
        output_depth_format,
        msaa_samples,
        false,
    )
}

macro_rules! wgpu_vertex_state {
    ($module:expr, $entry_point:expr, $buffers:expr $(,)?) => {{
        wgpu::VertexState {
            module: $module,
            entry_point: Some($entry_point),
            buffers: $buffers,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }
    }};
}
pub(crate) use wgpu_vertex_state;

macro_rules! wgpu_fragment_state {
    ($module:expr, $entry_point:expr, $targets:expr $(,)?) => {{
        wgpu::FragmentState {
            module: $module,
            entry_point: Some($entry_point),
            targets: $targets,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }
    }};
}
pub(crate) use wgpu_fragment_state;

macro_rules! wgpu_render_pipeline_descriptor {
    (
        label: $label:expr,
        layout: $layout:expr,
        vertex: $vertex:expr,
        fragment: $fragment:expr,
        primitive: $primitive:expr,
        depth_stencil: $depth_stencil:expr,
        multisample: $multisample:expr,
        multiview: $multiview:expr $(,)?
    ) => {{
        wgpu::RenderPipelineDescriptor {
            label: $label,
            layout: $layout,
            vertex: $vertex,
            fragment: $fragment,
            primitive: $primitive,
            depth_stencil: $depth_stencil,
            multisample: $multisample,
            multiview: $multiview,
            cache: None,
        }
    }};
}
pub(crate) use wgpu_render_pipeline_descriptor;

macro_rules! wgpu_compute_pipeline_descriptor {
    (
        label: $label:expr,
        layout: $layout:expr,
        module: $module:expr,
        entry_point: $entry_point:expr $(,)?
    ) => {{
        wgpu::ComputePipelineDescriptor {
            label: $label,
            layout: $layout,
            module: $module,
            entry_point: Some($entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        }
    }};
}
pub(crate) use wgpu_compute_pipeline_descriptor;
