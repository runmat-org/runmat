//! A 2D figure exported offscreen draws its frame: the plot area sits inside margins that hold
//! the tick labels, axis titles and figure title, the legend draws, and each series is one
//! stroke. Runs with `RUNMAT_PLOT_FORCE_GPU_TESTS=1`, like the other headless export tests.

use glam::Vec4;
use runmat_plot::export::image::{ImageExportSettings, ImageExporter};
use runmat_plot::plots::line::LineStyle;
use runmat_plot::plots::{Figure, Line3Plot, LinePlot};
use runmat_plot::styling::{PlotThemeConfig, ThemeVariant};

const W: u32 = 800;
const H: u32 = 500;

fn gpu_tests_on() -> bool {
    std::env::var("RUNMAT_PLOT_FORCE_GPU_TESTS")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn figure(legend: bool) -> Figure {
    let mut fig = Figure::new();
    fig.set_title("Step response");
    fig.set_axis_labels("Time (us)", "Voltage (V)");
    fig.set_legend(legend);
    let x: Vec<f64> = (0..=200).map(|i| i as f64 * 0.05).collect();
    let y: Vec<f64> = x.iter().map(|t| 1.0 - (-t).exp()).collect();
    let line = LinePlot::new(x, y)
        .unwrap()
        .with_style(Vec4::new(1.0, 0.0, 0.0, 1.0), 2.0, LineStyle::Solid)
        .with_label("AIN");
    fig.add_line_plot(line);
    fig
}

async fn render(legend: bool) -> Vec<u8> {
    let mut exporter = ImageExporter::with_settings(ImageExportSettings {
        width: W,
        height: H,
        ..Default::default()
    })
    .await
    .unwrap();
    exporter.set_theme_config(PlotThemeConfig {
        variant: ThemeVariant::ClassicLight,
        ..Default::default()
    });
    exporter
        .render_rgba_bytes(&mut figure(legend))
        .await
        .unwrap()
}

fn px(rgba: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2]]
}

fn is_series(p: [u8; 3]) -> bool {
    p[0] > 170 && p[1] < 110 && p[2] < 110
}

fn is_ink(p: [u8; 3]) -> bool {
    p.iter().all(|&c| c < 140)
}

#[tokio::test]
async fn positioned_axes_export_keeps_3d_and_2d_series_in_their_own_panels() {
    if !gpu_tests_on() {
        eprintln!("skipping; set RUNMAT_PLOT_FORCE_GPU_TESTS=1 for native viewport acceptance");
        return;
    }
    let mut figure = Figure::new();
    for (axes, position) in [
        [0.03, 0.06, 0.59, 0.88],
        [0.70, 0.56, 0.28, 0.38],
        [0.70, 0.08, 0.28, 0.38],
    ]
    .into_iter()
    .enumerate()
    {
        figure.ensure_axes(axes);
        figure.set_axes_position(axes, position);
        figure.set_axes_legend_enabled(axes, false);
    }
    let map = Line3Plot::new(vec![-1., 0., 1.], vec![0., 1., 0.], vec![-0.5, 1., 0.25])
        .unwrap()
        .with_style(Vec4::new(1., 0., 1., 1.), 4., LineStyle::Solid);
    figure.add_line3_plot_on_axes(map, 0);
    figure.set_axes_view(0, 32., 34.);
    for (axes, color, y) in [
        (1, Vec4::new(0., 1., 1., 1.), vec![1., 3., 2.]),
        (2, Vec4::new(1., 1., 0., 1.), vec![2., 1., 4.]),
    ] {
        let line =
            LinePlot::new(vec![0., 1., 2.], y)
                .unwrap()
                .with_style(color, 4., LineStyle::Solid);
        figure.add_line_plot_on_axes(line, axes);
    }
    // Use the GPU context directly: a CPU fallback cannot qualify the GPU
    // viewport path or its incremental scene updates.
    let mut context = runmat_plot::export::native_surface::create_headless_context(W, H)
        .await
        .unwrap();
    context.set_theme_config(PlotThemeConfig {
        variant: ThemeVariant::ClassicLight,
        ..Default::default()
    });
    let rgba = context.render_to_rgba(&figure, None, None).await.unwrap();
    let _ = image::save_buffer(
        std::env::temp_dir().join("runtime072-positioned-axes.png"),
        &rgba,
        W,
        H,
        image::ColorType::Rgba8,
    );
    let series_panel = |[red, green, blue]: [u8; 3]| {
        if red > 160 && green < 100 && blue > 160 {
            Some(0)
        } else if red < 60 && green > 160 && blue > 160 {
            // Cyan series cores have less red than the blue 3D axis labels.
            Some(1)
        } else if red > 160 && green > 160 && blue < 100 {
            Some(2)
        } else {
            None
        }
    };
    let mut counts = [0_usize; 3];
    for y in 0..H {
        for x in 0..W {
            let Some(panel) = series_panel(px(&rgba, x, y)) else {
                continue;
            };
            counts[panel] += 1;
            let in_panel = match panel {
                0 => x < W * 65 / 100,
                1 => x > W * 65 / 100 && y < H * 55 / 100,
                2 => x > W * 65 / 100 && y > H * 45 / 100,
                _ => unreachable!(),
            };
            assert!(
                in_panel,
                "series for panel {panel} escaped to pixel ({x}, {y}): {:?}",
                px(&rgba, x, y)
            );
        }
    }
    assert!(
        counts.iter().all(|count| *count > 25),
        "missing series: {counts:?}"
    );

    figure.set_axes_view(0, 115., 12.);
    let Some(runmat_plot::plots::figure::PlotElement::Line(line)) = figure.get_plot_mut(1) else {
        panic!("the upper-right panel must contain its original line");
    };
    line.update_data(vec![0., 0.5, 2.], vec![3., 1., 2.])
        .unwrap();
    let updated = context.render_to_rgba(&figure, None, None).await.unwrap();
    let mut changed = [0_usize; 3];
    for y in 0..H {
        for x in 0..W {
            let before = series_panel(px(&rgba, x, y));
            let after = series_panel(px(&updated, x, y));
            for (panel, count) in changed.iter_mut().enumerate() {
                *count += usize::from((before == Some(panel)) != (after == Some(panel)));
            }
        }
    }
    assert!(changed[0] > 25, "the 3D view did not update: {changed:?}");
    assert!(changed[1] > 25, "the line data did not update: {changed:?}");
    assert_eq!(changed[2], 0, "the untouched panel changed: {changed:?}");
}

#[tokio::test]
async fn offscreen_2d_export_draws_axes_text_legend_and_one_stroke_per_series() {
    if !gpu_tests_on() {
        eprintln!(
            "skipping; set RUNMAT_PLOT_FORCE_GPU_TESTS=1 to enable headless GPU export tests"
        );
        return;
    }
    let with = render(true).await;
    assert_eq!(with.len(), (W * H * 4) as usize);
    // For a look when a check fails.
    let _ = image::save_buffer(
        std::env::temp_dir().join("export_overlay.png"),
        &with,
        W,
        H,
        image::ColorType::Rgba8,
    );

    // The plot area is inset: the series never reaches the left margin, where the y tick labels
    // and the y axis title are, nor the bottom margin of the x tick labels and title.
    let margin_x = W / 14;
    let margin_y = H / 14;
    let mut series_in_margin = 0;
    for y in 0..H {
        for x in 0..margin_x {
            series_in_margin += is_series(px(&with, x, y)) as u32;
        }
    }
    for y in H - margin_y..H {
        for x in 0..W {
            series_in_margin += is_series(px(&with, x, y)) as u32;
        }
    }
    assert_eq!(
        series_in_margin, 0,
        "the series runs into the axis margins: no plot frame"
    );

    // Text in the left and bottom margins (tick labels, axis titles) and in the top one (title).
    let ink = |x0: u32, x1: u32, y0: u32, y1: u32| -> u32 {
        let mut n = 0;
        for y in y0..y1 {
            for x in x0..x1 {
                n += is_ink(px(&with, x, y)) as u32;
            }
        }
        n
    };
    assert!(
        ink(0, margin_x, 0, H) > 80,
        "no y tick labels or y axis title"
    );
    assert!(
        ink(0, W, H - margin_y, H) > 80,
        "no x tick labels or x axis title"
    );
    assert!(ink(0, W, 0, margin_y) > 40, "no title");

    // The legend adds its frame, swatch and label.
    let without = render(false).await;
    let changed = with
        .chunks_exact(4)
        .zip(without.chunks_exact(4))
        .filter(|(a, b)| a[..3].iter().zip(&b[..3]).any(|(p, q)| p.abs_diff(*q) > 40))
        .count();
    assert!(
        changed > 80,
        "the legend does not draw ({changed} pixels change)"
    );

    // One stroke per series: down a column through the middle of the curve (clear of the legend) there is one run of
    // series pixels, at most a few pixels tall.
    let x = W / 2;
    let mut runs = Vec::new();
    let mut start: Option<u32> = None;
    for y in 0..H {
        let on = is_series(px(&with, x, y));
        match (on, start) {
            (true, None) => start = Some(y),
            (false, Some(s)) => {
                runs.push((s, y - s));
                start = None;
            }
            _ => {}
        }
    }
    assert_eq!(
        runs.len(),
        1,
        "the series draws as {} strokes at x = {x}: {runs:?}",
        runs.len()
    );
    assert!(runs[0].1 <= 8, "the stroke is {} px tall", runs[0].1);
}
