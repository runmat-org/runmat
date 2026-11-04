//! 3D framing of exported figures: `surf`/`mesh` with `view(az, el)` and CAD meshes in source units.
//!
//! Each case renders to PNG through the same exporter `print -dpng` uses. Set
//! `RUNMAT_PLOT_FRAMING_OUT=<dir>` to keep the PNGs for inspection.

use glam::Vec3;
use runmat_plot::export::image::{ImageExportSettings, ImageExporter};
use runmat_plot::plots::surface::ShadingMode;
use runmat_plot::plots::{Figure, MeshPlot, SurfacePlot};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

fn finance_figure() -> Figure {
    // Black-Scholes-like call surface: spot 60..140 (span 80), expiry 0.05..2 (span ~2), price 0..48.
    let spots: Vec<f64> = (0..=40).map(|i| 60.0 + 2.0 * i as f64).collect();
    let years: Vec<f64> = (1..=40).map(|i| 0.05 * i as f64).collect();
    let z = spots
        .iter()
        .map(|s| {
            years
                .iter()
                .map(|t| ((s - 100.0) + 20.0 * t.sqrt()).max(0.0) * (1.0 + 0.1 * t))
                .collect()
        })
        .collect();
    let surface = SurfacePlot::new(spots, years, z)
        .unwrap()
        .with_shading(ShadingMode::Smooth);
    let mut figure = Figure::new();
    figure.add_surface_plot(surface);
    figure.set_axes_title(0, "Call price");
    figure.set_axes_xlabel(0, "Spot");
    figure.set_axes_ylabel(0, "Years to expiry");
    figure.set_axes_zlabel(0, "Price");
    figure.set_axes_colorbar_enabled(0, true);
    figure.set_axes_view(0, -35.0, 28.0);
    figure
}

fn wire_mesh_figure() -> Figure {
    let x: Vec<f64> = (0..=40).map(|i| -2.0 + 0.1 * i as f64).collect();
    let y = x.clone();
    let z = x
        .iter()
        .map(|x| y.iter().map(|y| x * (-x * x - y * y).exp()).collect())
        .collect();
    let surface = SurfacePlot::new(x, y, z)
        .unwrap()
        .with_wireframe(true)
        .with_shading(ShadingMode::Faceted);
    let mut figure = Figure::new();
    figure.add_surface_plot(surface);
    figure.set_axes_view(0, -35.0, 30.0);
    figure
}

/// A closed box in millimetre coordinates, offset from the origin like an exported assembly.
fn cad_box_figure(min: Vec3, max: Vec3) -> Figure {
    let corners: Vec<Vec3> = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 == 0 { min.x } else { max.x },
                if i & 2 == 0 { min.y } else { max.y },
                if i & 4 == 0 { min.z } else { max.z },
            )
        })
        .collect();
    let triangles = vec![
        [0, 2, 1],
        [1, 2, 3],
        [4, 5, 6],
        [5, 7, 6],
        [0, 1, 4],
        [1, 5, 4],
        [2, 6, 3],
        [3, 6, 7],
        [0, 4, 2],
        [2, 4, 6],
        [1, 3, 5],
        [3, 7, 5],
    ];
    let mut figure = Figure::new();
    figure.add_mesh_plot(MeshPlot::new(corners, triangles).unwrap());
    figure.set_axes_view(0, -35.0, 30.0);
    figure
}

async fn render(name: &str, mut figure: Figure) -> Vec<u8> {
    let exporter = ImageExporter::with_settings(ImageExportSettings {
        width: WIDTH,
        height: HEIGHT,
        ..Default::default()
    })
    .await
    .unwrap();
    let rgba = exporter.render_rgba_bytes(&mut figure).await.unwrap();
    if let Ok(dir) = std::env::var("RUNMAT_PLOT_FRAMING_OUT") {
        let png = exporter.encode_png_bytes(&rgba).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.png")), png).unwrap();
    }
    rgba
}

/// Bounding box (x0, y0, x1, y1) of pixels whose colour differs strongly from the background
/// sampled at the image centre-top edge, ignoring the lower-left orientation triad.
fn content_box(rgba: &[u8]) -> Option<(u32, u32, u32, u32)> {
    let background = &rgba[((WIDTH / 2) * 4) as usize..((WIDTH / 2) * 4 + 3) as usize];
    let mut bounds: Option<(u32, u32, u32, u32)> = None;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if x < 140 && y > HEIGHT - 160 {
                continue;
            }
            let index = ((y * WIDTH + x) * 4) as usize;
            let delta: i32 = (0..3)
                .map(|c| (rgba[index + c] as i32 - background[c] as i32).abs())
                .sum();
            if delta > 90 {
                bounds = Some(match bounds {
                    None => (x, y, x, y),
                    Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                });
            }
        }
    }
    bounds
}

#[tokio::test]
async fn three_d_figures_frame_their_data_in_the_exported_image() {
    let cases = [
        ("surf-finance", finance_figure()),
        ("mesh-wireframe", wire_mesh_figure()),
        (
            "cad-box-mm",
            cad_box_figure(
                Vec3::new(-1160.0, -660.0, -40.0),
                Vec3::new(1160.0, 660.0, 180.0),
            ),
        ),
        (
            "cad-box-m",
            cad_box_figure(Vec3::new(0.01, 0.02, 0.0), Vec3::new(0.09, 0.07, 0.008)),
        ),
    ];
    for (name, figure) in cases {
        let rgba = render(name, figure).await;
        let (x0, y0, x1, y1) =
            content_box(&rgba).unwrap_or_else(|| panic!("{name}: nothing was drawn"));
        let width = (x1 - x0) as f32 / WIDTH as f32;
        let height = (y1 - y0) as f32 / HEIGHT as f32;
        println!("{name}: content box ({x0},{y0})-({x1},{y1}) fills {width:.2} x {height:.2}");
        assert!(
            width > 0.4 || height > 0.4,
            "{name}: content fills only {width:.2} x {height:.2} of the image"
        );
    }
}
