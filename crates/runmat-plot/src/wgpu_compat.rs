#[cfg(target_arch = "wasm32")]
pub(crate) type TexelCopyTextureInfo<'a> = wgpu::TexelCopyTextureInfo<'a>;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type TexelCopyTextureInfo<'a> = wgpu::ImageCopyTexture<'a>;

#[cfg(target_arch = "wasm32")]
pub(crate) type TexelCopyBufferInfo<'a> = wgpu::TexelCopyBufferInfo<'a>;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type TexelCopyBufferInfo<'a> = wgpu::ImageCopyBuffer<'a>;

#[cfg(target_arch = "wasm32")]
pub(crate) type TexelCopyBufferLayout = wgpu::TexelCopyBufferLayout;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type TexelCopyBufferLayout = wgpu::ImageDataLayout;

#[cfg(target_arch = "wasm32")]
pub(crate) fn instance_new(desc: wgpu::InstanceDescriptor) -> wgpu::Instance {
    wgpu::Instance::new(&desc)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn instance_new(desc: wgpu::InstanceDescriptor) -> wgpu::Instance {
    wgpu::Instance::new(desc)
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
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

#[cfg(all(
    not(target_arch = "wasm32"),
    any(test, feature = "gui", feature = "web")
))]
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

#[cfg(target_arch = "wasm32")]
pub(crate) fn default_device_descriptor<'a>() -> wgpu::DeviceDescriptor<'a> {
    wgpu::DeviceDescriptor {
        memory_hints: wgpu::MemoryHints::default(),
        ..Default::default()
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn default_device_descriptor<'a>() -> wgpu::DeviceDescriptor<'a> {
    wgpu::DeviceDescriptor::default()
}

#[cfg(all(feature = "egui-overlay", target_arch = "wasm32"))]
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

#[cfg(all(feature = "egui-overlay", not(target_arch = "wasm32")))]
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
    )
}

macro_rules! wgpu_vertex_state {
    ($module:expr, $entry_point:expr, $buffers:expr $(,)?) => {{
        #[cfg(target_arch = "wasm32")]
        {
            wgpu::VertexState {
                module: $module,
                entry_point: Some($entry_point),
                buffers: $buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            wgpu::VertexState {
                module: $module,
                entry_point: $entry_point,
                buffers: $buffers,
            }
        }
    }};
}
pub(crate) use wgpu_vertex_state;

macro_rules! wgpu_fragment_state {
    ($module:expr, $entry_point:expr, $targets:expr $(,)?) => {{
        #[cfg(target_arch = "wasm32")]
        {
            wgpu::FragmentState {
                module: $module,
                entry_point: Some($entry_point),
                targets: $targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            wgpu::FragmentState {
                module: $module,
                entry_point: $entry_point,
                targets: $targets,
            }
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
        #[cfg(target_arch = "wasm32")]
        {
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
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            wgpu::RenderPipelineDescriptor {
                label: $label,
                layout: $layout,
                vertex: $vertex,
                fragment: $fragment,
                primitive: $primitive,
                depth_stencil: $depth_stencil,
                multisample: $multisample,
                multiview: $multiview,
            }
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
