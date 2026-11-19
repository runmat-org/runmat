//! A camera-only frame copies no geometry, a present keeps a frame copy only when asked, and
//! the offscreen renderer is created once per host surface.

use super::*;
use crate::core::{Material, Vertex};
use crate::geometry_scene::GeometrySceneChunk;
use glam::{Vec3, Vec4};

fn gpu_tests_enabled() -> bool {
    if std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").as_deref() == Ok("1") {
        return true;
    }
    eprintln!("set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to exercise the real renderer");
    false
}

fn triangle_scene(scene_id: &str) -> Arc<GeometryScene> {
    let color = Vec4::new(0.4, 0.6, 0.9, 1.0);
    let vertices = vec![
        Vertex::new(Vec3::ZERO, color),
        Vertex::new(Vec3::X, color),
        Vertex::new(Vec3::Y, color),
    ];
    let chunk =
        GeometrySceneChunk::indexed_triangles("tri", vertices, vec![0, 1, 2], Material::default());
    Arc::new(GeometryScene::new(scene_id, 1, vec![chunk]))
}

fn target(context: &NativeSurfaceRenderContext) -> wgpu::Texture {
    context
        .renderer
        .wgpu_renderer
        .access
        .with_shared(|emission| {
            emission.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("frame reuse target"),
                size: wgpu::Extent3d {
                    width: context.config.width,
                    height: context.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: context.renderer.wgpu_renderer.surface_config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        })
        .unwrap()
}

#[tokio::test]
async fn camera_only_frames_share_the_scene_and_keep_no_frame_copy() {
    if !gpu_tests_enabled() {
        return;
    }
    let mut context = create_headless_context(96, 64).await.unwrap();
    let texture = target(&context);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let scene = triangle_scene("shared");
    let mut camera = Camera::new();
    for step in 0..3 {
        camera.position = Vec3::new(2.0 + step as f32, -2.0, 2.0);
        context
            .render_shared_geometry_scene_to_view(&scene, &view, Some(&camera), None)
            .unwrap();
        let held = context.renderer.shared_geometry_scene().unwrap();
        assert!(Arc::ptr_eq(held, &scene), "frame {step} copied the scene");
        // The default retention keeps no frame copy.
        context.retain_presented_frame(&texture);
        assert!(context.presented_frame().is_none());
    }
    assert_eq!(Arc::strong_count(&scene), 2);

    // A new scene revision replaces the held scene; the old one is released.
    let next = triangle_scene("shared-next");
    context
        .render_shared_geometry_scene_to_view(&next, &view, Some(&camera), None)
        .unwrap();
    assert_eq!(Arc::strong_count(&scene), 1);
    assert!(Arc::ptr_eq(
        context.renderer.shared_geometry_scene().unwrap(),
        &next
    ));
}

#[tokio::test]
async fn presented_frame_retention_follows_the_policy() {
    if !gpu_tests_enabled() {
        return;
    }
    let mut context = create_headless_context(32, 32).await.unwrap();
    let texture = target(&context);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let scene = triangle_scene("retention");
    let present = |context: &mut NativeSurfaceRenderContext| {
        context
            .render_shared_geometry_scene_to_view(&scene, &view, None, None)
            .unwrap();
        context.retain_presented_frame(&texture);
    };

    assert_eq!(
        context.presented_frame_retention(),
        PresentedFrameRetention::None
    );
    present(&mut context);
    assert!(context.presented_frame().is_none());

    context.set_presented_frame_retention(PresentedFrameRetention::NextPresent);
    present(&mut context);
    let kept = context
        .presented_frame()
        .expect("the requested frame is kept");
    assert_eq!(kept.size(), (32, 32));
    assert_eq!(kept.rgba().await.unwrap().len(), 32 * 32 * 4);
    assert_eq!(
        context.presented_frame_retention(),
        PresentedFrameRetention::None
    );
    // The next present drops the copy: it no longer shows the surface.
    present(&mut context);
    assert!(context.presented_frame().is_none());

    context.set_presented_frame_retention(PresentedFrameRetention::EveryPresent);
    present(&mut context);
    present(&mut context);
    assert!(context.presented_frame().is_some());
}

#[tokio::test]
async fn headless_frame_renderer_keeps_one_context_across_frames() {
    if !gpu_tests_enabled() {
        return;
    }
    let mut renderer = HeadlessFrameRenderer::new();
    assert!(!renderer.is_ready());
    let scene = triangle_scene("fallback");
    let first = renderer
        .render_geometry_scene(&scene, 40, 30, None, None)
        .await
        .unwrap();
    assert_eq!((first.width, first.height), (40, 30));
    assert_eq!(first.rgba.len(), 40 * 30 * 4);
    assert!(renderer.is_ready());
    let device_before = renderer
        .context(40, 30)
        .await
        .unwrap()
        .renderer
        .wgpu_renderer
        .access
        .clone();

    let resized = renderer
        .render_geometry_scene(&scene, 64, 48, None, None)
        .await
        .unwrap();
    assert_eq!((resized.width, resized.height), (64, 48));
    assert_eq!(resized.rgba.len(), 64 * 48 * 4);
    let context = renderer.context(64, 48).await.unwrap();
    assert!(
        context
            .renderer
            .wgpu_renderer
            .access
            .same_device(&device_before),
        "a resize must reuse the kept context's device"
    );
    assert!(Arc::ptr_eq(
        context.renderer.shared_geometry_scene().unwrap(),
        &scene
    ));

    renderer.reset();
    assert!(!renderer.is_ready());
}
