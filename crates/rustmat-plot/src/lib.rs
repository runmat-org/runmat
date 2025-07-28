use plotters::prelude::*;
use rustmat_macros::matlab_fn;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::Path;

/// Environment variable specifying the path to the optional YAML config file.
pub const CONFIG_ENV: &str = "RUSTMAT_PLOT_CONFIG";

/// Style configuration loaded from YAML.
#[derive(Clone, Deserialize)]
pub struct PlotConfig {
    /// Width of the output image in pixels.
    #[serde(default = "default_width")]
    pub width: u32,
    /// Height of the output image in pixels.
    #[serde(default = "default_height")]
    pub height: u32,
    /// Color of line plots in hex form (`#rrggbb`).
    #[serde(default = "default_line_color")]
    pub line_color: String,
    /// Line width in pixels.
    #[serde(default = "default_line_width")]
    pub line_width: u32,
    /// Color of scatter plot points.
    #[serde(default = "default_scatter_color")]
    pub scatter_color: String,
    /// Radius of scatter plot points in pixels.
    #[serde(default = "default_marker_size")]
    pub marker_size: u32,
    /// Fill color for bar charts.
    #[serde(default = "default_bar_color")]
    pub bar_color: String,
    /// Fill color for histograms.
    #[serde(default = "default_hist_color")]
    pub hist_color: String,
    /// Background color of the drawing area.
    #[serde(default = "default_background")]
    pub background: String,
}

fn default_line_color() -> String {
    String::from("#0000ff")
}

fn default_scatter_color() -> String {
    String::from("#ff0000")
}

fn default_bar_color() -> String {
    String::from("#0088cc")
}

fn default_hist_color() -> String {
    String::from("#aaaaaa")
}

fn default_width() -> u32 {
    800
}

fn default_height() -> u32 {
    600
}

fn default_line_width() -> u32 {
    2
}

fn default_marker_size() -> u32 {
    4
}

fn default_background() -> String {
    String::from("#ffffff")
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: default_width(),
            height: default_height(),
            line_color: default_line_color(),
            scatter_color: default_scatter_color(),
            bar_color: default_bar_color(),
            hist_color: default_hist_color(),
            line_width: default_line_width(),
            marker_size: default_marker_size(),
            background: default_background(),
        }
    }
}

/// Load a configuration from the given path.
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<PlotConfig, String> {
    let data = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    serde_yaml::from_str(&data).map_err(|e| e.to_string())
}

fn current_config() -> PlotConfig {
    if let Ok(path) = env::var(CONFIG_ENV) {
        load_config(&path).unwrap_or_default()
    } else {
        PlotConfig::default()
    }
}

fn parse_color(hex: &str) -> Result<RGBColor, String> {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        return Err("invalid color".into());
    }
    let r = u8::from_str_radix(&hex[0..2], 16).map_err(|_| "invalid color")?;
    let g = u8::from_str_radix(&hex[2..4], 16).map_err(|_| "invalid color")?;
    let b = u8::from_str_radix(&hex[4..6], 16).map_err(|_| "invalid color")?;
    Ok(RGBColor(r, g, b))
}

/// Plot a line series and write to PNG or SVG depending on the file extension.
#[matlab_fn(name = "plot")]
pub fn plot_line(xs: &[f64], ys: &[f64], path: &str) -> Result<(), String> {
    if xs.len() != ys.len() {
        return Err("input length mismatch".into());
    }
    let config = current_config();
    let color = parse_color(&config.line_color)?;
    let bg = parse_color(&config.background)?;

    if path.ends_with(".svg") {
        let backend = SVGBackend::new(path, (config.width, config.height));
        draw_line(backend, xs, ys, &config, color, bg)
    } else if path.ends_with(".png") {
        let backend = BitMapBackend::new(path, (config.width, config.height));
        draw_line(backend, xs, ys, &config, color, bg)
    } else {
        Err("unsupported extension".into())
    }
}

fn draw_line<B: DrawingBackend>(
    backend: B,
    xs: &[f64],
    ys: &[f64],
    config: &PlotConfig,
    color: RGBColor,
    bg: RGBColor,
) -> Result<(), String> {
    let root = backend.into_drawing_area();
    root.fill(&bg).map_err(|e| e.to_string())?;
    let (xmin, xmax) = xs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
            (min.min(x), max.max(x))
        });
    let (ymin, ymax) = ys
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &y| {
            (min.min(y), max.max(y))
        });
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)
        .map_err(|e| e.to_string())?;
    chart.configure_mesh().draw().map_err(|e| e.to_string())?;
    chart
        .draw_series(LineSeries::new(
            xs.iter().zip(ys).map(|(&x, &y)| (x, y)),
            ShapeStyle::from(&color).stroke_width(config.line_width),
        ))
        .map_err(|e| e.to_string())?;
    root.present().map_err(|e| e.to_string())
}

/// Plot an XY scatter plot using circular markers.
#[matlab_fn(name = "scatter")]
pub fn plot_scatter(xs: &[f64], ys: &[f64], path: &str) -> Result<(), String> {
    if xs.len() != ys.len() {
        return Err("input length mismatch".into());
    }
    let config = current_config();
    let color = parse_color(&config.scatter_color)?;
    let bg = parse_color(&config.background)?;
    if path.ends_with(".svg") {
        let backend = SVGBackend::new(path, (config.width, config.height));
        draw_scatter(backend, xs, ys, &config, color, bg)
    } else if path.ends_with(".png") {
        let backend = BitMapBackend::new(path, (config.width, config.height));
        draw_scatter(backend, xs, ys, &config, color, bg)
    } else {
        Err("unsupported extension".into())
    }
}

fn draw_scatter<B: DrawingBackend>(
    backend: B,
    xs: &[f64],
    ys: &[f64],
    config: &PlotConfig,
    color: RGBColor,
    bg: RGBColor,
) -> Result<(), String> {
    let root = backend.into_drawing_area();
    root.fill(&bg).map_err(|e| e.to_string())?;
    let (xmin, xmax) = xs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
            (min.min(x), max.max(x))
        });
    let (ymin, ymax) = ys
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &y| {
            (min.min(y), max.max(y))
        });
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)
        .map_err(|e| e.to_string())?;
    chart.configure_mesh().draw().map_err(|e| e.to_string())?;
    chart
        .draw_series(xs.iter().zip(ys).map(|(&x, &y)| {
            Circle::new(
                (x, y),
                config.marker_size,
                ShapeStyle::from(&color).filled(),
            )
        }))
        .map_err(|e| e.to_string())?;
    root.present().map_err(|e| e.to_string())
}

/// Plot a vertical bar chart.
#[matlab_fn(name = "bar")]
pub fn plot_bar(labels: &[&str], values: &[f64], path: &str) -> Result<(), String> {
    if labels.len() != values.len() {
        return Err("input length mismatch".into());
    }
    let config = current_config();
    let color = parse_color(&config.bar_color)?;
    let bg = parse_color(&config.background)?;
    if path.ends_with(".svg") {
        let backend = SVGBackend::new(path, (config.width, config.height));
        draw_bar(backend, labels, values, &config, color, bg)
    } else if path.ends_with(".png") {
        let backend = BitMapBackend::new(path, (config.width, config.height));
        draw_bar(backend, labels, values, &config, color, bg)
    } else {
        Err("unsupported extension".into())
    }
}

fn draw_bar<B: DrawingBackend>(
    backend: B,
    labels: &[&str],
    values: &[f64],
    _config: &PlotConfig,
    color: RGBColor,
    bg: RGBColor,
) -> Result<(), String> {
    let root = backend.into_drawing_area();
    root.fill(&bg).map_err(|e| e.to_string())?;
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_val = if max_val.is_finite() { max_val } else { 0.0 };
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(0..labels.len() as i32, 0f64..max_val)
        .map_err(|e| e.to_string())?;
    chart
        .configure_mesh()
        .x_labels(labels.len())
        .x_label_formatter(&|idx| labels.get(*idx as usize).unwrap_or(&"").to_string())
        .draw()
        .map_err(|e| e.to_string())?;
    chart
        .draw_series(values.iter().enumerate().map(|(i, &v)| {
            Rectangle::new(
                [(i as i32, 0.0), (i as i32 + 1, v)],
                ShapeStyle::from(&color).filled(),
            )
        }))
        .map_err(|e| e.to_string())?;
    root.present().map_err(|e| e.to_string())
}

/// Plot a histogram given a set of samples and number of bins.
#[matlab_fn(name = "histogram")]
pub fn plot_histogram(values: &[f64], bins: usize, path: &str) -> Result<(), String> {
    if bins == 0 {
        return Err("bins must be > 0".into());
    }
    let config = current_config();
    let color = parse_color(&config.hist_color)?;
    let bg = parse_color(&config.background)?;
    if path.ends_with(".svg") {
        let backend = SVGBackend::new(path, (config.width, config.height));
        draw_hist(backend, values, bins, &config, color, bg)
    } else if path.ends_with(".png") {
        let backend = BitMapBackend::new(path, (config.width, config.height));
        draw_hist(backend, values, bins, &config, color, bg)
    } else {
        Err("unsupported extension".into())
    }
}

fn draw_hist<B: DrawingBackend>(
    backend: B,
    values: &[f64],
    bins: usize,
    _config: &PlotConfig,
    color: RGBColor,
    bg: RGBColor,
) -> Result<(), String> {
    let root = backend.into_drawing_area();
    root.fill(&bg).map_err(|e| e.to_string())?;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_width = (max - min) / bins as f64;
    let mut counts = vec![0usize; bins];
    for &v in values {
        let idx = ((v - min) / bin_width).floor() as usize;
        let idx = if idx >= bins { bins - 1 } else { idx };
        counts[idx] += 1;
    }
    let max_count = *counts.iter().max().unwrap_or(&0) as f64;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(min..max, 0f64..max_count)
        .map_err(|e| e.to_string())?;
    chart.configure_mesh().draw().map_err(|e| e.to_string())?;
    chart
        .draw_series((0..bins).map(|i| {
            let x0 = min + i as f64 * bin_width;
            let x1 = x0 + bin_width;
            let y = counts[i] as f64;
            Rectangle::new([(x0, 0.0), (x1, y)], ShapeStyle::from(&color).filled())
        }))
        .map_err(|e| e.to_string())?;
    root.present().map_err(|e| e.to_string())
}

/// Convert a 3D point into 2D screen coordinates using a simple perspective
/// projection with a fixed camera orientation. This replaces the earlier
/// isometric approach and provides a more natural view.
const ROT_X: f64 = std::f64::consts::FRAC_PI_4; // 45 deg tilt
const ROT_Y: f64 = std::f64::consts::FRAC_PI_4; // 45 deg rotation
const CAMERA_DIST: f64 = 5.0;

fn project_perspective(x: f64, y: f64, z: f64) -> (f64, f64) {
    // rotate around X axis
    let (sin_x, cos_x) = ROT_X.sin_cos();
    let y1 = cos_x * y - sin_x * z;
    let z1 = sin_x * y + cos_x * z;

    // rotate around Y axis
    let (sin_y, cos_y) = ROT_Y.sin_cos();
    let x2 = cos_y * x + sin_y * z1;
    let z2 = -sin_y * x + cos_y * z1;

    // perspective divide
    let scale = CAMERA_DIST / (CAMERA_DIST + z2);
    (x2 * scale, y1 * scale)
}

/// Plot a 3D scatter plot using a perspective projection.
#[matlab_fn(name = "scatter3")]
pub fn plot_3d_scatter(xs: &[f64], ys: &[f64], zs: &[f64], path: &str) -> Result<(), String> {
    if xs.len() != ys.len() || xs.len() != zs.len() {
        return Err("input length mismatch".into());
    }
    let config = current_config();
    let color = parse_color(&config.scatter_color)?;
    let bg = parse_color(&config.background)?;
    let points: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys)
        .zip(zs)
        .map(|((&x, &y), &z)| project_perspective(x, y, z))
        .collect();
    let (xmin, xmax) = points
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| {
            (min.min(*x), max.max(*x))
        });
    let (ymin, ymax) = points
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (_, y)| {
            (min.min(*y), max.max(*y))
        });
    if path.ends_with(".svg") {
        let backend = SVGBackend::new(path, (config.width, config.height));
        draw_scatter2d(backend, &points, &config, color, bg, xmin, xmax, ymin, ymax)
    } else if path.ends_with(".png") {
        let backend = BitMapBackend::new(path, (config.width, config.height));
        draw_scatter2d(backend, &points, &config, color, bg, xmin, xmax, ymin, ymax)
    } else {
        Err("unsupported extension".into())
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_scatter2d<B: DrawingBackend>(
    backend: B,
    points: &[(f64, f64)],
    config: &PlotConfig,
    color: RGBColor,
    bg: RGBColor,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
) -> Result<(), String> {
    let root = backend.into_drawing_area();
    root.fill(&bg).map_err(|e| e.to_string())?;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)
        .map_err(|e| e.to_string())?;
    chart.configure_mesh().draw().map_err(|e| e.to_string())?;
    chart
        .draw_series(points.iter().map(|&(x, y)| {
            Circle::new((x, y), config.marker_size, ShapeStyle::from(color).filled())
        }))
        .map_err(|e| e.to_string())?;
    root.present().map_err(|e| e.to_string())
}

/// Plot a surface defined on a square grid using a perspective projection.
/// The number of points must form an `n x n` grid.
#[matlab_fn(name = "surf")]
pub fn plot_surface(xs: &[f64], ys: &[f64], zs: &[f64], path: &str) -> Result<(), String> {
    if xs.len() != ys.len() || xs.len() != zs.len() {
        return Err("input length mismatch".into());
    }
    let n = (xs.len() as f64).sqrt() as usize;
    if n * n != xs.len() {
        return Err("surface data must form a square grid".into());
    }
    let config = current_config();
    let line_color = parse_color(&config.line_color)?;
    let bg = parse_color(&config.background)?;
    let mut projected = Vec::with_capacity(xs.len());
    for ((&x, &y), &z) in xs.iter().zip(ys).zip(zs) {
        projected.push(project_perspective(x, y, z));
    }
    let (xmin, xmax) = projected
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (x, _)| {
            (min.min(*x), max.max(*x))
        });
    let (ymin, ymax) = projected
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (_, y)| {
            (min.min(*y), max.max(*y))
        });
    if path.ends_with(".svg") {
        let backend = SVGBackend::new(path, (config.width, config.height));
        draw_surface2d(
            backend, &projected, n, line_color, bg, xmin, xmax, ymin, ymax,
        )
    } else if path.ends_with(".png") {
        let backend = BitMapBackend::new(path, (config.width, config.height));
        draw_surface2d(
            backend, &projected, n, line_color, bg, xmin, xmax, ymin, ymax,
        )
    } else {
        Err("unsupported extension".into())
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_surface2d<B: DrawingBackend>(
    backend: B,
    points: &[(f64, f64)],
    n: usize,
    color: RGBColor,
    bg: RGBColor,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
) -> Result<(), String> {
    let root = backend.into_drawing_area();
    root.fill(&bg).map_err(|e| e.to_string())?;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)
        .map_err(|e| e.to_string())?;
    chart.configure_mesh().draw().map_err(|e| e.to_string())?;
    // draw grid lines
    for i in 0..n {
        for j in 0..n - 1 {
            let p1 = points[i * n + j];
            let p2 = points[i * n + j + 1];
            chart
                .draw_series(std::iter::once(PathElement::new(vec![p1, p2], color)))
                .map_err(|e| e.to_string())?;
        }
    }
    for j in 0..n {
        for i in 0..n - 1 {
            let p1 = points[i * n + j];
            let p2 = points[(i + 1) * n + j];
            chart
                .draw_series(std::iter::once(PathElement::new(vec![p1, p2], color)))
                .map_err(|e| e.to_string())?;
        }
    }
    root.present().map_err(|e| e.to_string())
}
