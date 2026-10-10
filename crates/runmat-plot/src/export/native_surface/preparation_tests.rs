use super::*;
use pollster::FutureExt;
use std::sync::Arc;

struct ClearInPrepare {
    buffer: Arc<wgpu::Buffer>,
}

impl egui_wgpu::CallbackTrait for ClearInPrepare {
    fn prepare(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("plot-test-egui-callback-preparation"),
        });
        encoder.clear_buffer(&self.buffer, 0, None);
        vec![encoder.finish()]
    }

    fn paint(
        &self,
        _info: egui::epaint::PaintCallbackInfo,
        _render_pass: &mut wgpu::RenderPass<'static>,
        _resources: &egui_wgpu::CallbackResources,
    ) {
    }
}

#[test]
#[ignore = "requires an actual native WGPU adapter"]
fn callback_preparation_executes_before_native_surface_main_encoder() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .block_on()
        .expect("this hardware gate requires an actual WGPU adapter");
    let (device, queue) = adapter
        .request_device(&crate::wgpu_compat::default_device_descriptor(), None)
        .block_on()
        .expect("this hardware gate requires an actual WGPU device");
    let source = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("plot-test-prepared-source"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("plot-test-prepared-readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    queue.write_buffer(&source, 0, &0x1234_5678_u32.to_le_bytes());

    let mut renderer =
        egui_wgpu::Renderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, None, 1, false);
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(4.0, 4.0));
    let paint_jobs = [egui::epaint::ClippedPrimitive {
        clip_rect: rect,
        primitive: egui::epaint::Primitive::Callback(egui_wgpu::Callback::new_paint_callback(
            rect,
            ClearInPrepare {
                buffer: source.clone(),
            },
        )),
    }];
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("plot-test-native-surface-main"),
    });
    let preparation = renderer.update_buffers(
        &device,
        &queue,
        &mut encoder,
        &paint_jobs,
        &ScreenDescriptor {
            size_in_pixels: [4, 4],
            pixels_per_point: 1.0,
        },
    );
    assert_eq!(
        preparation.len(),
        1,
        "callback returned one preparation buffer"
    );
    encoder.copy_buffer_to_buffer(&source, 0, &readback, 0, 4);
    let result = PreparedOverlayRender {
        result: RenderResult {
            success: true,
            data_bounds: None,
            axes_viewports_px: Vec::new(),
            vertex_count: 0,
            triangle_count: 0,
            render_time_ms: 0.0,
        },
        preparation,
    }
    .submit(&queue, encoder);
    assert!(result.success);

    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |mapped| {
            tx.send(mapped).expect("readback receiver remains live");
        });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .expect("readback callback ran")
        .expect("readback mapped");
    let mapped = readback.slice(..).get_mapped_range();
    assert_eq!(&mapped[..4], &0_u32.to_le_bytes());
    drop(mapped);
    readback.unmap();
}
