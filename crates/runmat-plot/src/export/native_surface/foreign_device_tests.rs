//! GPU-resident plot data made on one device draws the same on a surface with its own device.

use super::*;
use crate::core::{BoundingBox, PipelineType, Vertex};
use crate::plots::{LineGpuStyle, LinePlot, LineStyle};
use crate::GpuVertexBuffer;
use glam::{Vec3, Vec4};
use runmat_accelerate_api::WgpuBufferOwner;
use wgpu::util::DeviceExt;

async fn device(adapter: &wgpu::Adapter) -> WgpuDeviceAccess {
    let (device, queue) = adapter
        .request_device(&crate::wgpu_compat::default_device_descriptor(), None)
        .await
        .expect("GPU device");
    WgpuDeviceAccess::new(device, queue)
}

/// A filled triangle whose vertices live in a buffer on `provider`, recorded as its device.
fn gpu_triangle_figure(provider: &WgpuDeviceAccess) -> Figure {
    let color = Vec4::new(0.9, 0.3, 0.1, 1.0);
    let vertices = [
        Vertex::new(Vec3::new(0.0, 0.0, 0.0), color),
        Vertex::new(Vec3::new(1.0, 0.0, 0.0), color),
        Vertex::new(Vec3::new(0.5, 1.0, 0.0), color),
    ];
    let buffer = provider
        .with_shared(|emission| {
            emission
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("provider plot vertices"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_SRC,
                })
        })
        .unwrap();
    let line = LinePlot::from_gpu_buffer(
        GpuVertexBuffer::new(WgpuBufferOwner::on_device(buffer, provider), 3),
        3,
        LineGpuStyle {
            color,
            line_width: 2.0,
            line_style: LineStyle::Solid,
            marker: None,
            handle_visibility: "on".to_string(),
        },
        BoundingBox::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0)),
        PipelineType::Triangles,
        None,
    );
    let mut figure = Figure::new();
    figure.add_line_plot(line);
    figure
}

async fn render(
    access: &WgpuDeviceAccess,
    figure: &Figure,
) -> (NativeSurfaceRenderContext, Vec<u8>) {
    let mut context =
        NativeSurfaceRenderContext::new(access.clone(), 96, 72, wgpu::TextureFormat::Rgba8Unorm)
            .await
            .unwrap();
    let pixels = context.render_to_rgba(figure, None, None).await.unwrap();
    (context, pixels)
}

#[tokio::test]
async fn provider_buffers_draw_the_same_on_a_surface_device() {
    if std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to exercise the real renderer");
        return;
    }
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("GPU adapter");
    let provider = device(&adapter).await;
    let surface = device(&adapter).await;
    assert!(!provider.same_device(&surface));
    let figure = gpu_triangle_figure(&provider);

    let (same, expected) = render(&provider, &figure).await;
    assert_eq!(same.renderer.foreign_buffer_copy_count(), 0);
    let (_, empty) = render(&provider, &Figure::new()).await;
    assert_ne!(expected, empty, "the GPU-resident triangle must be drawn");

    let (mut other, actual) = render(&surface, &figure).await;
    assert_eq!(other.renderer.foreign_buffer_copy_count(), 1);
    assert_eq!(
        actual, expected,
        "a copy on the surface device draws the same pixels"
    );

    // A later frame of the same data binds the kept copy.
    let again = other.render_to_rgba(&figure, None, None).await.unwrap();
    assert_eq!(again, expected);
    assert_eq!(other.renderer.foreign_buffer_copy_count(), 1);
}
