//! Regression for scene-only surfaces: fit/reset must not flatten imported geometry.
use glam::Vec3;
use runmat_plot::{
    core::{camera::ProjectionType, CameraViewPreset, PlotRenderer},
    event::FigureScene,
    plots::{Figure, MeshPlot},
};

#[tokio::test]
async fn imported_geometry_remains_spatial_after_fit_and_reset() {
    if std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to exercise the real renderer");
        return;
    }
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("GPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: 760,
        height: 475,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    let access = runmat_accelerate_api::WgpuDeviceAccess::new(device, queue);
    let mut renderer = PlotRenderer::new(access.clone(), config).await.unwrap();
    let mut sources = Vec::new();
    if let Ok(directory) = std::env::var("RUNMAT_DOCS_SNAPSHOT_PROJECT") {
        for view in ["geometry", "displacement", "stress"] {
            let bytes = std::fs::read(format!("{directory}/.snapshot/{view}.scene.json")).unwrap();
            let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            sources.push(serde_json::from_value::<FigureScene>(payload["figure"].clone()).unwrap());
        }
    } else {
        let mut figure = Figure::new();
        figure.add_mesh_plot(
            MeshPlot::new(
                vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z],
                vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
            )
            .unwrap(),
        );
        sources.push(FigureScene::capture(&figure));
    }
    for (index, payload) in sources.into_iter().enumerate() {
        let scene = payload
            .into_geometry_scene(format!("cold-{index}"), index as u64)
            .unwrap();
        renderer.set_geometry_scene(scene.clone());
        assert!(renderer.axes_has_3d_content(0));
        assert!(!renderer.axes_has_3d_content(1));
        for _ in 0..2 {
            renderer.fit_extents();
            let camera = renderer.axes_camera(0).unwrap();
            assert!(matches!(
                camera.projection,
                ProjectionType::Perspective { .. }
            ));
            assert!(camera.position.is_finite());
            assert!((camera.position.z - camera.target.z).abs() > 0.01);
            assert_eq!(camera.up, Vec3::Z);
            renderer.set_camera_view_preset(CameraViewPreset::Top);
            renderer.reset_geometry_view();
        }
        if let Ok(directory) = std::env::var("RUNMAT_DOCS_REPLAY_IMAGES") {
            use runmat_plot::export::image::{ImageExportSettings, ImageExporter};
            let exporter = ImageExporter::with_settings(ImageExportSettings {
                width: 760,
                height: 475,
                ..Default::default()
            })
            .await
            .unwrap();
            let bytes = exporter
                .render_geometry_scene_png_bytes_with_camera(
                    &scene,
                    renderer.axes_camera(0).unwrap(),
                )
                .await
                .unwrap();
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(format!("{directory}/{index}.png"), bytes).unwrap();
        }
    }
    // Switching back to an ordinary 2D figure must recover its planar behavior.
    let mut figure = Figure::new();
    figure
        .add_line_plot(runmat_plot::plots::LinePlot::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap());
    access
        .with_shared(|emission| renderer.set_figure(emission, figure))
        .unwrap();
    assert!(!renderer.axes_has_3d_content(0));
}

/// A geometry scene is true-scale: fitting a 180 x 50 x 10 box keeps its proportions (one scale
/// on every axis) instead of stretching each extent to a cube.
#[tokio::test]
async fn geometry_scene_fit_keeps_true_proportions() {
    if std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to exercise the real renderer");
        return;
    }
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("GPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: 760,
        height: 475,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    let access = runmat_accelerate_api::WgpuDeviceAccess::new(device, queue);
    let mut renderer = PlotRenderer::new(access, config).await.unwrap();
    let (x, y, z) = (180.0, 50.0, 10.0);
    let corners: Vec<Vec3> = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 == 0 { 0.0 } else { x },
                if i & 2 == 0 { 0.0 } else { y },
                if i & 4 == 0 { 0.0 } else { z },
            )
        })
        .collect();
    let quads = [
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 5, 7, 3],
    ];
    let triangles: Vec<[u32; 3]> = quads
        .iter()
        .flat_map(|[a, b, c, d]| [[*a, *b, *c], [*a, *c, *d]])
        .collect();
    let mut figure = Figure::new();
    figure.add_mesh_plot(MeshPlot::new(corners, triangles).unwrap());
    let scene = FigureScene::capture(&figure)
        .into_geometry_scene("box".to_string(), 1)
        .unwrap();
    renderer.set_geometry_scene(scene);
    renderer.fit_extents();
    let camera = renderer.axes_camera(0).unwrap();
    let s = camera.axis_scale;
    assert!(
        (s.x - s.y).abs() <= 1e-6 * s.x && (s.x - s.z).abs() <= 1e-6 * s.x,
        "{s:?}"
    );
    assert!((s.x - 1.0 / x).abs() <= 1e-6 / x * 10.0, "{s:?}");
    // The fitted box keeps its 180:50:10 shape in scaled space.
    let scaled = Vec3::new(x, y, z) * s;
    assert!((scaled.x / scaled.y - x / y).abs() < 1e-4, "{scaled:?}");
    assert!((scaled.y / scaled.z - y / z).abs() < 1e-4, "{scaled:?}");
}
