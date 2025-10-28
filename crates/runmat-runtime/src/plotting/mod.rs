//! Modern plotting builtin functions using the world-class runmat-plot system
//!
//! These functions integrate with the configuration system to provide
//! interactive GUI plotting, static exports, and Jupyter notebook support.

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
// no extra imports needed; Value/Tensor already imported above
use runmat_plot::plots::*;
use std::env;
use once_cell::sync::OnceCell;
use std::sync::Mutex;

mod style;
use style::{is_style_string, parse_line_spec, parse_color_value, SeriesStyle};

/// Determine the plotting mode from environment/config
#[cfg(not(test))]
fn get_plotting_mode() -> PlottingMode {
    let mode_env = env::var("RUNMAT_PLOT_MODE")
        .or_else(|_| env::var("RUSTMAT_PLOT_MODE"))
        .unwrap_or_else(|_| "auto".to_string());
    match mode_env.to_lowercase().as_str() {
        "gui" => PlottingMode::Interactive,
        "headless" => PlottingMode::Static,
        "jupyter" => PlottingMode::Jupyter,
        _ => PlottingMode::Auto,
    }
}

#[cfg(test)]
fn get_plotting_mode() -> PlottingMode { PlottingMode::Static }

/// Plotting mode configuration
#[derive(Debug, Clone, Copy)]
enum PlottingMode {
    Auto,
    Interactive,
    Static,
    Jupyter,
}

/// Execute a plot based on the current mode
fn execute_plot(mut figure: Figure) -> Result<String, String> {
    match get_plotting_mode() {
        PlottingMode::Interactive => interactive_export(&mut figure),
        PlottingMode::Static => static_export(&mut figure, "plot.png"),
        PlottingMode::Jupyter => jupyter_export(&mut figure),
        PlottingMode::Auto => {
            if env::var("JPY_PARENT_PID").is_ok() || env::var("JUPYTER_RUNTIME_DIR").is_ok() {
                jupyter_export(&mut figure)
            } else {
                interactive_export(&mut figure)
            }
        }
    }
}

/// Launch interactive GUI window using platform-optimal approach
fn interactive_export(figure: &mut Figure) -> Result<String, String> {
    let figure_clone = figure.clone();
    match runmat_plot::show_interactive_platform_optimal(figure_clone) {
        Ok(result) => Ok(result),
        Err(e) => {
            let msg = format!("{e}");
            if msg.contains("EventLoop can't be recreated") {
                Ok("Plot window closed successfully".to_string())
            } else {
                Err(format!(
                    "Interactive plotting failed: {e}. Please check GPU/GUI system setup."
                ))
            }
        }
    }
}

/// Export plot as static file using the figure data
fn static_export(figure: &mut Figure, filename: &str) -> Result<String, String> {
    if figure.is_empty() { return Err("No plots found in figure to export".to_string()); }
    runmat_plot::show_plot_unified(figure.clone(), Some(filename))
}

/// Export plot for Jupyter
#[cfg(feature = "jupyter")]
fn jupyter_export(figure: &mut Figure) -> Result<String, String> {
    use runmat_plot::jupyter::JupyterBackend;
    let mut backend = JupyterBackend::new();
    backend.display_figure(figure)
}

#[cfg(not(feature = "jupyter"))]
fn jupyter_export(_figure: &mut Figure) -> Result<String, String> {
    Err("Jupyter feature not enabled".to_string())
}

#[cfg(test)]
mod plotting_mode_lint_sanity {
    use super::PlottingMode;
    #[test]
    fn constructs_all_variants() {
        // Construct all variants so dead-code lints do not trigger in test builds
        let _ = PlottingMode::Auto;
        let _ = PlottingMode::Interactive;
        let _ = PlottingMode::Static;
        let _ = PlottingMode::Jupyter;
    }
}

/// Extract numeric vector from a Matrix
fn extract_numeric_vector(matrix: &Tensor) -> Vec<f64> {
    matrix.data.clone()
}

// Module-level helpers reusable across builtins
fn get_string_value(v: &Value) -> Option<String> {
    if let Value::String(s) = v { Some(s.clone()) } else { None }
}

// Unified MATLAB-compatible plot dispatcher
#[runtime_builtin(name = "plot")]
fn plot_varargs(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty() { return Err("plot: not enough input arguments".to_string()); }

    let mut figure = take_or_new_figure()
        .with_title("Plot")
        .with_labels("X", "Y")
        .with_grid(true);

    {
        let st = state().lock().unwrap();
        figure = figure.with_xlog(st.x_log).with_ylog(st.y_log);
    }

    let colors = [
        glam::Vec4::new(0.0, 0.4470, 0.7410, 1.0),
        glam::Vec4::new(0.8500, 0.3250, 0.0980, 1.0),
        glam::Vec4::new(0.9290, 0.6940, 0.1250, 1.0),
        glam::Vec4::new(0.4940, 0.1840, 0.5560, 1.0),
        glam::Vec4::new(0.4660, 0.6740, 0.1880, 1.0),
        glam::Vec4::new(0.3010, 0.7450, 0.9330, 1.0),
        glam::Vec4::new(0.6350, 0.0780, 0.1840, 1.0),
    ];

    fn to_tensor(v: &Value) -> Result<Tensor, String> { match v { Value::Tensor(t) => Ok(t.clone()), _ => Err("plot: expected numeric array".to_string()), } }
    fn is_string(v: &Value) -> bool { matches!(v, Value::String(_)) }
    fn get_string(v: &Value) -> Option<String> { match v { Value::String(s) => Some(s.clone()), _ => None } }
    fn extract_column(t: &Tensor, col: usize) -> Vec<f64> { let rows = t.rows(); (0..rows).map(|r| t.data[r + col * rows]).collect() }
    fn sequence_1_to(n: usize) -> Vec<f64> { (1..=n).map(|i| i as f64).collect() }
    fn is_vector_tensor(t: &Tensor) -> bool { t.shape.len() <= 1 || t.rows() == 1 || t.cols() == 1 }
    fn apply_log_xy(mut x: Vec<f64>, mut y: Vec<f64>, x_log: bool, y_log: bool) -> Result<(Vec<f64>, Vec<f64>), String> {
        if x_log { if x.iter().any(|v| *v <= 0.0 || !v.is_finite()) { return Err("semilogx/loglog: X data must be positive and finite".to_string()); } for v in &mut x { *v = v.log10(); } }
        if y_log { if y.iter().any(|v| *v <= 0.0 || !v.is_finite()) { return Err("semilogy/loglog: Y data must be positive and finite".to_string()); } for v in &mut y { *v = v.log10(); } }
        Ok((x, y))
    }

    let (x_log_flag, y_log_flag) = { let st = state().lock().unwrap(); (st.x_log, st.y_log) };

    let mut i = 0usize; let mut series_count = 0usize;
    while i < rest.len() {
        if rest.len() == 1 {
            let y = to_tensor(&rest[0])?;
            if is_vector_tensor(&y) {
                let n = y.data.len();
                let x = sequence_1_to(n);
                let (color, style) = (colors[series_count % colors.len()], runmat_plot::plots::line::LineStyle::Solid);
                let line = LinePlot::new(x, y.data.clone()).map_err(|e| format!("plot: {e}"))?.with_style(color, 1.0, style);
                let idx = figure.add_line_plot(line);
                if state().lock().unwrap().current_axes > 0 { let _ = figure.assign_plot_to_axes(idx, state().lock().unwrap().current_axes); }
            } else {
                let rows = y.rows(); let cols = y.cols(); let x = sequence_1_to(rows);
                for c in 0..cols { let yc = extract_column(&y, c); let color = colors[series_count % colors.len()];
                    let line = LinePlot::new(x.clone(), yc).map_err(|e| format!("plot: {e}"))?.with_style(color, 1.0, runmat_plot::plots::line::LineStyle::Solid).with_label(format!("Series {}", c + 1));
                    let idx = figure.add_line_plot(line);
                    if state().lock().unwrap().current_axes > 0 { let _ = figure.assign_plot_to_axes(idx, state().lock().unwrap().current_axes); }
                    series_count += 1; }
            }
            break;
        }

        if i + 1 >= rest.len() { return Err("plot: expected X,Y pairs".to_string()); }
        let x = to_tensor(&rest[i])?; let y = to_tensor(&rest[i+1])?; i += 2;
        let mut style = SeriesStyle::default(); let mut label_opt: Option<String> = None; let mut marker_face_color_opt: Option<glam::Vec4> = None; let mut marker_edge_color_opt: Option<glam::Vec4> = None;
        if i < rest.len() && is_string(&rest[i]) { if let Some(s) = get_string(&rest[i]) { if is_style_string(&s) { style = parse_line_spec(&s); i += 1; } } }
        while i + 1 < rest.len() {
            if let Some(key) = get_string(&rest[i]) {
                let key_l = key.to_ascii_lowercase();
                match key_l.as_str() {
                    "displayname" => { if let Some(v) = get_string(&rest[i+1]) { label_opt = Some(v); i += 2; continue; } }
                    "linewidth" => { let w: f64 = std::convert::TryInto::try_into(&rest[i+1])?; style.line_width = w as f32; i += 2; continue; }
                    "markersize" => { let s: f64 = std::convert::TryInto::try_into(&rest[i+1])?; style.marker_size = s as f32; i += 2; continue; }
                    "color" => { if let Some(c) = parse_color_value(&rest[i+1]) { style.color = c; i += 2; continue; } }
                    "markerfacecolor" | "facecolor" => { if let Some(c) = parse_color_value(&rest[i+1]) { marker_face_color_opt = Some(c); i += 2; continue; } }
                    "markeredgecolor" | "edgecolor" => { if let Some(c) = parse_color_value(&rest[i+1]) { marker_edge_color_opt = Some(c); i += 2; continue; } }
                    "linestyle" => { if let Some(v) = get_string(&rest[i+1]) { let v = v.as_str(); use runmat_plot::plots::line::LineStyle; style.line_style = match v { "-"=>Some(LineStyle::Solid), "--"=>Some(LineStyle::Dashed), ":"=>Some(LineStyle::Dotted), "-."=>Some(LineStyle::DashDot), "none"=>None, _=>style.line_style }; i += 2; continue; } }
                    "marker" => { if let Some(v) = get_string(&rest[i+1]) { use runmat_plot::plots::scatter::MarkerStyle; style.marker = match v.as_str() { "."|"o"=>Some(MarkerStyle::Circle), "x"|"+"=>Some(MarkerStyle::Plus), "*"=>Some(MarkerStyle::Star), "s"=>Some(MarkerStyle::Square), "d"=>Some(MarkerStyle::Diamond), "^"|"v"|"<"|">"=>Some(MarkerStyle::Triangle), "h"=>Some(MarkerStyle::Hexagon), "none"=>None, _=>style.marker }; i += 2; continue; } }
                    _ => {}
                }
            }
            break;
        }

        let rows = y.rows(); let cols = y.cols();
        if is_vector_tensor(&y) {
            let yv = y.data.clone(); let xv = if is_vector_tensor(&x) { x.data.clone() } else { return Err("plot: X must be a vector when Y is a vector".to_string()); };
            if xv.len() != yv.len() { return Err("plot: X and Y must have the same length".to_string()); }
            let (xv, yv) = apply_log_xy(xv, yv, x_log_flag, y_log_flag)?;
            let st = if series_count == 0 { let mut s = style.clone(); if s.color == SeriesStyle::default().color { s.color = colors[series_count % colors.len()]; } s } else { style.clone() };
            if let Some(ls) = st.line_style { let mut line = LinePlot::new(xv.clone(), yv.clone()).map_err(|e| format!("plot: {e}"))?.with_style(st.color, st.line_width, ls); if let Some(lbl) = label_opt.clone() { line = line.with_label(lbl.clone()); } let idx = figure.add_line_plot(line); if state().lock().unwrap().current_axes > 0 { let _ = figure.assign_plot_to_axes(idx, state().lock().unwrap().current_axes); } }
            if let Some(m) = st.marker { let mut scatter = ScatterPlot::new(xv, yv).map_err(|e| format!("plot: {e}"))?.with_style(st.color, st.marker_size, m); if let Some(fc)=marker_face_color_opt { scatter.set_face_color(fc); } if let Some(ec)=marker_edge_color_opt { scatter.set_edge_color(ec); } scatter.set_edge_thickness(st.line_width.max(0.0)); if let Some(lbl) = label_opt.clone() { scatter = scatter.with_label(lbl); } let idx = figure.add_scatter_plot(scatter); if state().lock().unwrap().current_axes > 0 { let _ = figure.assign_plot_to_axes(idx, state().lock().unwrap().current_axes); } }
            series_count += 1;
        } else {
            if is_vector_tensor(&x) {
                if x.data.len() != rows { return Err("plot: length(X) must equal rows(Y)".to_string()); }
                for c in 0..cols {
                    let yc = extract_column(&y, c); let st = if series_count == 0 { let mut s = style.clone(); if s.color == SeriesStyle::default().color { s.color = colors[series_count % colors.len()]; } s } else { style.clone() };
                    let xv = x.data.clone(); let (xv, yc) = apply_log_xy(xv, yc, x_log_flag, y_log_flag)?;
                    if let Some(ls) = st.line_style { let mut line = LinePlot::new(xv.clone(), yc.clone()).map_err(|e| format!("plot: {e}"))?.with_style(st.color, st.line_width, ls); if let Some(lbl) = label_opt.clone() { line = line.with_label(lbl.clone()); } let idx = figure.add_line_plot(line); if state().lock().unwrap().current_axes > 0 { let _ = figure.assign_plot_to_axes(idx, state().lock().unwrap().current_axes); } }
                    if let Some(m) = st.marker { let mut scatter = ScatterPlot::new(xv, yc).map_err(|e| format!("plot: {e}"))?.with_style(st.color, st.marker_size, m); if let Some(fc)=marker_face_color_opt { scatter.set_face_color(fc); } if let Some(ec)=marker_edge_color_opt { scatter.set_edge_color(ec); } scatter.set_edge_thickness(st.line_width.max(0.0)); if let Some(lbl) = label_opt.clone() { scatter = scatter.with_label(lbl); } let idx = figure.add_scatter_plot(scatter); if state().lock().unwrap().current_axes > 0 { let _ = figure.assign_plot_to_axes(idx, state().lock().unwrap().current_axes); } }
                    series_count += 1;
                }
            } else if x.rows() == rows && x.cols() == cols {
                for c in 0..cols {
                    let yc = extract_column(&y, c); let xc = extract_column(&x, c);
                    let st = if series_count == 0 { let mut s = style.clone(); if s.color == SeriesStyle::default().color { s.color = colors[series_count % colors.len()]; } s } else { style.clone() };
                    let (xc, yc) = apply_log_xy(xc, yc, x_log_flag, y_log_flag)?;
                    if let Some(ls) = st.line_style { let mut line = LinePlot::new(xc.clone(), yc.clone()).map_err(|e| format!("plot: {e}"))?.with_style(st.color, st.line_width, ls); if let Some(lbl) = label_opt.clone() { line = line.with_label(lbl.clone()); } let idx = figure.add_line_plot(line); if state().lock().unwrap().current_axes > 0 { let _ = figure.assign_plot_to_axes(idx, state().lock().unwrap().current_axes); } }
                    if let Some(m) = st.marker { let mut scatter = ScatterPlot::new(xc, yc).map_err(|e| format!("plot: {e}"))?.with_style(st.color, st.marker_size, m); if let Some(fc)=marker_face_color_opt { scatter.set_face_color(fc); } if let Some(ec)=marker_edge_color_opt { scatter.set_edge_color(ec); } scatter.set_edge_thickness(st.line_width.max(0.0)); if let Some(lbl) = label_opt.clone() { scatter = scatter.with_label(lbl); } let idx = figure.add_scatter_plot(scatter); if state().lock().unwrap().current_axes > 0 { let _ = figure.assign_plot_to_axes(idx, state().lock().unwrap().current_axes); } }
                    series_count += 1;
                }
            } else { return Err("plot: X must be vector length rows(Y) or same size as Y".to_string()); }
        }
    }

    let hold = state().lock().unwrap().hold_on;
    if hold { store_current_figure(figure.clone()); }
    let res = execute_plot(figure.clone());
    if !hold { store_current_figure(figure); }
    res
}

// ---------- MATLAB state: current figure/axes and hold ----------
#[derive(Debug, Default)]
struct PlotState {
    current_figure: Option<Figure>,
    hold_on: bool,
    x_log: bool,
    y_log: bool,
    current_axes: usize,
}

static GLOBAL_PLOT_STATE: OnceCell<Mutex<PlotState>> = OnceCell::new();
fn state() -> &'static Mutex<PlotState> { GLOBAL_PLOT_STATE.get_or_init(|| Mutex::new(PlotState::default())) }
fn take_or_new_figure() -> Figure {
    let mut st = state().lock().unwrap();
    if let Some(fig) = st.current_figure.take() {
        // Keep a copy in state and return the existing figure to continue working on it
        let clone = fig.clone();
        st.current_figure = Some(clone);
        return fig;
    }
    Figure::new()
}
fn store_current_figure(fig: Figure) { let mut st = state().lock().unwrap(); st.current_figure = Some(fig); }

#[runtime_builtin(name = "gcf")]
fn gcf_builtin() -> Result<String, String> { let has_fig = state().lock().unwrap().current_figure.is_some(); Ok(if has_fig { "figure exists" } else { "no figure" }.to_string()) }

#[runtime_builtin(name = "hold")]
fn hold_builtin(rest: Vec<runmat_builtins::Value>) -> Result<String, String> {
    let mut st = state().lock().unwrap();
    match rest.len() {
        0 => { st.hold_on = !st.hold_on; }
        1 => { let mode: String = std::convert::TryInto::try_into(&rest[0])?; let m = mode.to_ascii_lowercase(); if m == "on" || m == "all" { st.hold_on = true; } else if m == "off" || m == "none" { st.hold_on = false; } else if m == "toggle" { st.hold_on = !st.hold_on; } else { return Err(format!("hold: unknown mode '{mode}', expected on/off/toggle")); } }
        n => { return Err(format!("hold: expected 0 or 1 arguments, got {n}")); }
    }
    Ok(if st.hold_on { "hold on".to_string() } else { "hold off".to_string() })
}

#[runtime_builtin(name = "semilogx")]
fn semilogx_builtin(rest: Vec<Value>) -> Result<String, String> { { let mut st = state().lock().unwrap(); st.x_log = true; st.y_log = false; } plot_varargs(rest) }
#[runtime_builtin(name = "semilogy")]
fn semilogy_builtin(rest: Vec<Value>) -> Result<String, String> { { let mut st = state().lock().unwrap(); st.x_log = false; st.y_log = true; } plot_varargs(rest) }
#[runtime_builtin(name = "loglog")]
fn loglog_builtin(rest: Vec<Value>) -> Result<String, String> { { let mut st = state().lock().unwrap(); st.x_log = true; st.y_log = true; } plot_varargs(rest) }

fn mutate_figure_and_render<F>(mutator: F) -> Result<String, String>
where F: FnOnce(&mut Figure) { let mut fig = { let mut st = state().lock().unwrap(); st.current_figure.take().unwrap_or_else(Figure::new) }; mutator(&mut fig); let clone = fig.clone(); store_current_figure(fig); execute_plot(clone) }

#[runtime_builtin(name = "title")]
fn title_builtin(text: String) -> Result<String, String> { mutate_figure_and_render(|f| { f.title = Some(text); }) }
#[runtime_builtin(name = "xlabel")]
fn xlabel_builtin(text: String) -> Result<String, String> { mutate_figure_and_render(|f| { f.x_label = Some(text); }) }
#[runtime_builtin(name = "ylabel")]
fn ylabel_builtin(text: String) -> Result<String, String> { mutate_figure_and_render(|f| { f.y_label = Some(text); }) }

#[runtime_builtin(name = "axis")]
fn axis_builtin(rest: Vec<Value>) -> Result<String, String> {
    use runmat_builtins::Value::*;
    if rest.is_empty() { return Err("axis: expected args".to_string()); }
    if rest.len() == 1 {
        if let Tensor(t) = &rest[0] {
            if t.data.len() == 4 { let (xmin, xmax, ymin, ymax) = (t.data[0], t.data[1], t.data[2], t.data[3]); return mutate_figure_and_render(|f| { f.x_limits = Some((xmin, xmax)); f.y_limits = Some((ymin, ymax)); }); }
            return Err("axis: numeric vector must have 4 elements [xmin xmax ymin ymax]".to_string());
        }
        if let Ok(s) = <std::string::String as std::convert::TryFrom<&Value>>::try_from(&rest[0]) { let m = s.to_ascii_lowercase(); return mutate_figure_and_render(|f| { match m.as_str() { "tight" => {
                // Compute data bounds from render_data
                let mut xmin = f64::INFINITY; let mut xmax = f64::NEG_INFINITY; let mut ymin = f64::INFINITY; let mut ymax = f64::NEG_INFINITY;
                for rd in f.render_data() {
                    for v in &rd.vertices { xmin = xmin.min(v.position[0] as f64); xmax = xmax.max(v.position[0] as f64); ymin = ymin.min(v.position[1] as f64); ymax = ymax.max(v.position[1] as f64); }
                }
                if xmin.is_finite() && xmax.is_finite() && ymin.is_finite() && ymax.is_finite() {
                    f.x_limits = Some((xmin, xmax)); f.y_limits = Some((ymin, ymax));
                }
            }, "equal" => { f.x_log = false; f.y_log = false; f.axis_equal = true; }, "auto" => { f.x_limits = None; f.y_limits = None; }, _ => {} } }); }
    }
    if rest.len() == 4 { let xmin: f64 = std::convert::TryInto::try_into(&rest[0])?; let xmax: f64 = std::convert::TryInto::try_into(&rest[1])?; let ymin: f64 = std::convert::TryInto::try_into(&rest[2])?; let ymax: f64 = std::convert::TryInto::try_into(&rest[3])?; return mutate_figure_and_render(|f| { f.x_limits = Some((xmin, xmax)); f.y_limits = Some((ymin, ymax)); }); }
    Err("axis: expected [xmin xmax ymin ymax], four scalars, or keyword".to_string())
}

#[runtime_builtin(name = "grid")]
fn grid_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty() { return mutate_figure_and_render(|f| f.grid_enabled = !f.grid_enabled); }
    if rest.len() == 1 {
        let s: String = std::convert::TryInto::try_into(&rest[0])?;
        let m = s.to_ascii_lowercase();
        return mutate_figure_and_render(|f| {
            f.grid_enabled = matches!(m.as_str(), "on" | "all" | "show");
        });
    }
    Err("grid: expected 0 or 1 argument".to_string())
}

#[runtime_builtin(name = "legend")]
fn legend_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty() { return mutate_figure_and_render(|f| f.legend_enabled = !f.legend_enabled); }
    if rest.len() == 1 {
        if let Ok(s) = <String as std::convert::TryFrom<&Value>>::try_from(&rest[0]) { let m = s.to_ascii_lowercase(); return mutate_figure_and_render(|f| { f.legend_enabled = matches!(m.as_str(), "on" | "show"); }); }
        if let Value::Cell(ca) = &rest[0] { let mut labels = Vec::new(); for h in &ca.data { if let Value::String(s) = unsafe { &*h.as_raw() } { labels.push(s.clone()); } } return mutate_figure_and_render(|f| { f.set_labels(&labels); f.legend_enabled = true; }); }
    }
    let mut labels = Vec::new(); for v in rest.iter() { if let Ok(s) = <String as std::convert::TryFrom<&Value>>::try_from(v) { labels.push(s); } }
    if labels.is_empty() { return Err("legend: expected on/off or labels".to_string()); }
    mutate_figure_and_render(|f| { f.set_labels(&labels); f.legend_enabled = true; })
}

#[runtime_builtin(name = "colormap")]
fn colormap_builtin(name: String) -> Result<String, String> {
    use runmat_plot::plots::surface::ColorMap;
    let cmap = match name.to_ascii_lowercase().as_str() {
        "parula" => ColorMap::Parula, "jet" => ColorMap::Jet, "hot" => ColorMap::Hot, "cool" => ColorMap::Cool, "spring" => ColorMap::Spring, "summer" => ColorMap::Summer, "autumn" => ColorMap::Autumn, "winter" => ColorMap::Winter, "gray" | "grey" => ColorMap::Gray, "viridis" => ColorMap::Viridis, "plasma" => ColorMap::Plasma, "inferno" => ColorMap::Inferno, "magma" => ColorMap::Magma, "turbo" => ColorMap::Turbo, _ => return Err(format!("Unknown colormap: {name}")), };
    mutate_figure_and_render(|f| { f.colormap = cmap; })
}

#[runtime_builtin(name = "colorbar")]
fn colorbar_builtin(rest: Vec<Value>) -> Result<String, String> { if rest.is_empty() { return mutate_figure_and_render(|f| { f.colorbar_enabled = true; }); } if rest.len() == 1 { if let Ok(s) = <std::string::String as std::convert::TryFrom<&Value>>::try_from(&rest[0]) { let m = s.to_ascii_lowercase(); return mutate_figure_and_render(|f| { f.colorbar_enabled = matches!(m.as_str(), "on" | "show"); }); } } Err("colorbar: expected on/off or no args".to_string()) }

#[runtime_builtin(name = "caxis")]
fn caxis_builtin(rest: Vec<Value>) -> Result<String, String> {
    match rest.len() {
        0 => { let st = state().lock().unwrap(); if let Some(fig) = &st.current_figure { if let Some((a,b)) = fig.color_limits { return Ok(format!("[{}, {}]", a, b)); } } Err("caxis: no figure or limits".to_string()) }
        1 => { if let Ok(s) = <String as std::convert::TryFrom<&Value>>::try_from(&rest[0]) { let m = s.to_ascii_lowercase(); return mutate_figure_and_render(|f| { if m.as_str() == "auto" { f.set_color_limits(None); } }); } let v: Tensor = std::convert::TryInto::try_into(&rest[0])?; if v.data.len() != 2 { return Err("caxis: expected [cmin cmax]".to_string()); } let lo = v.data[0]; let hi = v.data[1]; mutate_figure_and_render(|f| f.set_color_limits(Some((lo, hi)))) }
        2 => { let lo: f64 = std::convert::TryInto::try_into(&rest[0])?; let hi: f64 = std::convert::TryInto::try_into(&rest[1])?; mutate_figure_and_render(|f| f.set_color_limits(Some((lo, hi)))) }
        _ => Err("caxis: expected 0, 1 or 2 arguments".to_string()),
    }
}

// imagesc / imshow
#[runtime_builtin(name = "imagesc")]
fn imagesc_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty() { return Err("imagesc: expected at least Z".to_string()); }
    let (x_vec, y_vec, z_grid, clim_opt): (Option<Vec<f64>>, Option<Vec<f64>>, Vec<Vec<f64>>, Option<(f64,f64)>) = {
        let mut args = rest.clone(); let mut clim: Option<(f64,f64)> = None; if let Some(Value::Tensor(t)) = args.last() { if t.data.len() == 2 { let t = t.clone(); args.pop(); clim = Some((t.data[0], t.data[1])); } }
        match args.len() { 1 => { let z: Tensor = std::convert::TryInto::try_into(&args[0])?; let rows = z.rows(); let cols = z.cols(); if rows == 0 || cols == 0 { return Err("imagesc: Z must be non-empty".to_string()); } let x: Vec<f64> = (1..=cols).map(|i| i as f64).collect(); let y: Vec<f64> = (1..=rows).map(|i| i as f64).collect(); let mut z_grid = vec![vec![0.0; cols]; rows]; for r in 0..rows { for c in 0..cols { z_grid[r][c] = z.data[r + c * rows]; } } (Some(x), Some(y), z_grid, clim) }
            3 => { let x: Tensor = std::convert::TryInto::try_into(&args[0])?; let y: Tensor = std::convert::TryInto::try_into(&args[1])?; let z: Tensor = std::convert::TryInto::try_into(&args[2])?; let rows = z.rows(); let cols = z.cols(); let x_vec = if x.cols() == 1 { x.data.clone() } else { return Err("imagesc: X must be a vector".to_string()) }; let y_vec = if y.cols() == 1 { y.data.clone() } else { return Err("imagesc: Y must be a vector".to_string()) }; if x_vec.len() != cols || y_vec.len() != rows { return Err("imagesc: size mismatch between X/Y and Z".to_string()); } let mut z_grid = vec![vec![0.0; cols]; rows]; for r in 0..rows { for c in 0..cols { z_grid[r][c] = z.data[r + c * rows]; } } (Some(x_vec), Some(y_vec), z_grid, clim) }
            _ => return Err("imagesc: expected Z or X,Y,Z (optional [clow chigh])".to_string()), }
    };
    let x = x_vec.unwrap(); let mut y = y_vec.unwrap(); if y.windows(2).all(|w| w[1] - w[0] == 1.0) { y.reverse(); }
    let mut _z_min = f64::INFINITY; let mut _z_max = f64::NEG_INFINITY; for row in &z_grid { for &z in row { if z.is_finite() { _z_min = _z_min.min(z); _z_max = _z_max.max(z); } } } if let Some((_lo, _hi)) = clim_opt { }
    let image_plot = runmat_plot::ImagePlot::from_grayscale(x.clone(), y.clone(), z_grid, runmat_plot::plots::surface::ColorMap::Parula, clim_opt).map_err(|e| format!("imagesc: {e}"))?.with_label("Image");
    let mut figure = take_or_new_figure().with_title("Image").with_labels("X", "Y").with_axis_equal(true).with_colorbar(true);
    figure.add_image_plot(image_plot);
    let hold = state().lock().unwrap().hold_on; if hold { store_current_figure(figure.clone()); } let res = execute_plot(figure.clone()); if !hold { store_current_figure(figure); } res
}

#[runtime_builtin(name = "imshow")]
fn imshow_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty() { return Err("imshow: expected image matrix".to_string()); }
    if let Value::Tensor(t) = &rest[0] {
        match t.shape.as_slice() {
            [_, _] => { return imagesc_builtin(vec![rest[0].clone()]); }
            [r, c, 3] => {
                let rows = *r; let cols = *c; let data = t.data.clone(); let maxv = data.iter().cloned().fold(0.0f64, f64::max); let scale = if maxv > 1.0 { 255.0 } else { 1.0 };
                let mut grid: Vec<Vec<glam::Vec4>> = vec![vec![glam::Vec4::ZERO; cols]; rows];
                for r in 0..rows { for c in 0..cols { let idx = r + c * rows; let rch = data[idx] / scale; let gch = data[idx + rows * cols] / scale; let bch = data[idx + 2 * rows * cols] / scale; grid[r][c] = glam::Vec4::new(rch as f32, gch as f32, bch as f32, 1.0); } }
                let x: Vec<f64> = (1..=cols).map(|i| i as f64).collect(); let mut y: Vec<f64> = (1..=rows).map(|i| i as f64).collect(); y.reverse();
                let image_plot = runmat_plot::ImagePlot::from_color_grid(x.clone(), y.clone(), grid).map_err(|e| format!("imshow: {e}"))?.with_label("Image");
                let mut figure = take_or_new_figure().with_title("Image").with_axis_equal(true).with_colorbar(false);
                figure.add_image_plot(image_plot);
                let hold = state().lock().unwrap().hold_on; if hold { store_current_figure(figure.clone()); } let res = execute_plot(figure.clone()); if !hold { store_current_figure(figure); } return res;
            }
            _ => {}
        }
    }
    imagesc_builtin(rest)
}

// Axis limits
#[runtime_builtin(name = "xlim")]
fn xlim_builtin(rest: Vec<Value>) -> Result<String, String> {
    match rest.len() {
        1 => { let v: Tensor = std::convert::TryInto::try_into(&rest[0])?; if v.data.len() != 2 { return Err("xlim: expected [xmin xmax]".to_string()); } let xmin = v.data[0]; let xmax = v.data[1]; mutate_figure_and_render(|f| { f.x_limits = Some((xmin, xmax)); }) }
        2 => { let xmin: f64 = std::convert::TryInto::try_into(&rest[0])?; let xmax: f64 = std::convert::TryInto::try_into(&rest[1])?; mutate_figure_and_render(|f| { f.x_limits = Some((xmin, xmax)); }) }
        0 => { let st = state().lock().unwrap(); if let Some(fig) = &st.current_figure { if let Some((a,b)) = fig.x_limits { return Ok(format!("[{}, {}]", a, b)); } } Err("xlim: no figure or limits".to_string()) }
        _ => Err("xlim: expected 0, 1 or 2 arguments".to_string()),
    }
}

#[runtime_builtin(name = "ylim")]
fn ylim_builtin(rest: Vec<Value>) -> Result<String, String> {
    match rest.len() {
        1 => { let v: Tensor = std::convert::TryInto::try_into(&rest[0])?; if v.data.len() != 2 { return Err("ylim: expected [ymin ymax]".to_string()); } let ymin = v.data[0]; let ymax = v.data[1]; mutate_figure_and_render(|f| { f.y_limits = Some((ymin, ymax)); }) }
        2 => { let ymin: f64 = std::convert::TryInto::try_into(&rest[0])?; let ymax: f64 = std::convert::TryInto::try_into(&rest[1])?; mutate_figure_and_render(|f| { f.y_limits = Some((ymin, ymax)); }) }
        0 => { let st = state().lock().unwrap(); if let Some(fig) = &st.current_figure { if let Some((a,b)) = fig.y_limits { return Ok(format!("[{}, {}]", a, b)); } } Err("ylim: no figure or limits".to_string()) }
        _ => Err("ylim: expected 0, 1 or 2 arguments".to_string()),
    }
}

#[runtime_builtin(name = "scatter")]
fn scatter_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty() { return Err("scatter: expected at least Y".to_string()); }
    fn to_vec(v: &Value) -> Result<Vec<f64>, String> { match v { Value::Tensor(t) => Ok(t.data.clone()), _ => Err("scatter: expected numeric vector".to_string()) } }
    fn to_scalar(v: &Value) -> Option<f64> { std::convert::TryInto::try_into(v).ok() }

    // Parse X and Y
    let (mut x, y, mut idx) = if rest.len() >= 2 && matches!(&rest[0], Value::Tensor(_)) && matches!(&rest[1], Value::Tensor(_)) {
        (to_vec(&rest[0])?, to_vec(&rest[1])?, 2usize)
    } else {
        (Vec::<f64>::new(), to_vec(&rest[0])?, 1usize)
    };
    let n = y.len(); if n == 0 { return Err("scatter: empty Y".to_string()); }
    if x.is_empty() { x = (1..=n).map(|k| k as f64).collect(); }
    if x.len() != n { return Err("scatter: length(X) must equal length(Y)".to_string()); }

    // Optional S and C
    let mut sizes_opt: Option<Vec<f32>> = None; // MATLAB S is area in points^2; approximate sqrt to px
    let mut colors_opt: Option<Vec<glam::Vec4>> = None; // per-point RGB(A)
    let mut values_opt: Option<Vec<f64>> = None; // colormap values
    let mut filled = false;

    // S (area in pt^2)
    if idx < rest.len() {
        match &rest[idx] {
            Value::Tensor(t) => {
                let is_vector = t.rows() == 1 || t.cols() == 1;
                if is_vector {
                    if t.data.len() == 1 {
                        let s = t.data[0].max(0.0);
                        let s_px = (s as f32).sqrt().max(1.0);
                        sizes_opt = Some(vec![s_px; n]);
                        idx += 1;
                    } else if t.data.len() == n {
                        // Ambiguous vector: decide S vs C by value range.
                        let all_in_0_1 = t.data.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0);
                        if all_in_0_1 {
                            // Likely color values; leave for C parsing below (do not consume idx)
                        } else {
                            // Treat as sizes
                            let mut out = Vec::with_capacity(n);
                            for i in 0..n { let s = t.data[i].max(0.0); out.push((s as f32).sqrt().max(1.0)); }
                            sizes_opt = Some(out);
                            idx += 1;
                        }
                    }
                }
            }
            _ => { if let Some(s) = to_scalar(&rest[idx]) { let s_px = (s as f32).sqrt().max(1.0); sizes_opt = Some(vec![s_px; n]); idx += 1; } }
        }
    }

    // C or 'filled'
    if idx < rest.len() {
        // 'filled' may appear directly
        if let Some(s) = get_string_value(&rest[idx]) { if s.eq_ignore_ascii_case("filled") { filled = true; idx += 1; } }
    }
    if idx < rest.len() {
        match &rest[idx] {
            Value::Tensor(t) => {
                // Prefer 3xN (channels in rows) if ambiguous
                if t.rows() == 3 && t.cols() >= 1 {
                    let rows = t.rows(); let cols = t.cols();
                    let mut cols_vec = Vec::with_capacity(cols);
                    for cix in 0..cols {
                        let rch = t.data[0 + cix * rows] as f32;
                        let gch = t.data[1 + cix * rows] as f32;
                        let bch = t.data[2 + cix * rows] as f32;
                        cols_vec.push(glam::Vec4::new(rch, gch, bch, 1.0));
                    }
                    if cols == 1 { colors_opt = Some(vec![cols_vec[0]; n]); } else { colors_opt = Some(cols_vec); }
                    idx += 1;
                } else if t.cols() == 3 && t.rows() >= 1 {
                    let rows = t.rows();
                    let mut cols_vec = Vec::with_capacity(rows);
                    for r in 0..rows { let rch = t.data[r] as f32; let gch = t.data[r + rows*1] as f32; let bch = t.data[r + rows*2] as f32; cols_vec.push(glam::Vec4::new(rch, gch, bch, 1.0)); }
                    if rows == 1 { colors_opt = Some(vec![cols_vec[0]; n]); } else { colors_opt = Some(cols_vec); }
                    idx += 1;
                } else if t.cols() == 1 && t.rows() >= 1 { values_opt = Some(t.data.clone()); idx += 1; }
                else if t.data.len() == 3 { colors_opt = Some(vec![glam::Vec4::new(t.data[0] as f32, t.data[1] as f32, t.data[2] as f32, 1.0); n]); idx += 1; }
            }
            Value::String(s) => {
                if s.eq_ignore_ascii_case("filled") { filled = true; idx += 1; }
                else if let Some(c) = parse_color_value(&rest[idx]) { colors_opt = Some(vec![c; n]); idx += 1; }
            }
            _ => {}
        }
    }

    // Remaining: 'filled' and name-value pairs
    if idx < rest.len() { if let Some(s) = get_string_value(&rest[idx]) { if s.eq_ignore_ascii_case("filled") { filled = true; idx += 1; } } }

    // Name-Value
    let mut edge_color_opt: Option<glam::Vec4> = None;
    let mut face_color_opt: Option<glam::Vec4> = None;
    let mut label_opt: Option<String> = None;
    let mut marker_opt: Option<runmat_plot::plots::scatter::MarkerStyle> = None;
    let mut line_width_opt: Option<f32> = None; // edge thickness
    while idx + 1 <= rest.len() {
        if idx < rest.len() {
            if let Some(k) = get_string_value(&rest[idx]) {
                let kl = k.to_ascii_lowercase();
                if kl == "filled" { filled = true; idx += 1; continue; }
                if idx + 1 < rest.len() {
                    match kl.as_str() {
                        "displayname" => { if let Some(s) = get_string_value(&rest[idx+1]) { label_opt = Some(s); idx += 2; continue; } }
                        "markersize" => { if let Some(v) = to_scalar(&rest[idx+1]) { let s_px = (v as f32).sqrt().max(1.0); sizes_opt = Some(vec![s_px; n]); idx += 2; continue; } }
                        "marker" => { if let Some(s) = get_string_value(&rest[idx+1]) { use runmat_plot::plots::scatter::MarkerStyle; let m = match s.as_str() { "."|"o"=>Some(MarkerStyle::Circle),
                                "x"|"+"=>Some(MarkerStyle::Plus), "*"=>Some(MarkerStyle::Star),
                                "s"=>Some(MarkerStyle::Square), "d"=>Some(MarkerStyle::Diamond),
                                "^"|"v"|"<"|">"=>Some(MarkerStyle::Triangle), "h"=>Some(MarkerStyle::Hexagon),
                                "none"=>None, _=>marker_opt }; marker_opt = m; idx += 2; continue; } }
                        "markerfacecolor"|"facecolor" => { if let Some(c) = parse_color_value(&rest[idx+1]) { face_color_opt = Some(c); idx += 2; continue; } }
                        "markeredgecolor"|"edgecolor" => { if let Some(c) = parse_color_value(&rest[idx+1]) { edge_color_opt = Some(c); idx += 2; continue; } }
                        "linewidth" => { if let Some(v) = to_scalar(&rest[idx+1]) { line_width_opt = Some(v as f32); idx += 2; continue; } }
                        "cdata" => { if let Value::Tensor(t) = &rest[idx+1] { if t.cols()==1 { values_opt = Some(t.data.clone()); } else if t.cols()==3 { let rows=t.rows(); let mut cols_vec=Vec::with_capacity(rows); for r in 0..rows { let rch=t.data[r] as f32; let gch=t.data[r+rows] as f32; let bch=t.data[r+rows*2] as f32; cols_vec.push(glam::Vec4::new(rch,gch,bch,1.0)); } colors_opt=Some(cols_vec); } idx += 2; continue; } }
                        _ => {}
                    }
                }
            }
        }
        break;
    }

    let mut figure = take_or_new_figure().with_title("Scatter Plot").with_labels("X","Y").with_grid(true);
    // Use figure's current colormap for scalar mapping
    let fig_colormap = figure.colormap;

    let mut sp = ScatterPlot::new(x, y).map_err(|e| format!("Failed to create scatter plot: {e}"))?;
    if let Some(m) = marker_opt { sp.set_marker_style(m); }
    if let Some(sizes) = sizes_opt { sp.set_sizes(sizes); } else { /* default size already set */ }
    if let Some(cols) = colors_opt { sp.set_colors(cols); }
    if let Some(vals) = values_opt { sp.set_color_values(vals, None); sp = sp.with_colormap(fig_colormap); }
    if let Some(fc) = face_color_opt { sp.set_face_color(fc); }
    if let Some(ec) = edge_color_opt { sp.set_edge_color(ec); }
    if let Some(lw) = line_width_opt { sp.set_edge_thickness(lw); }
    if let Some(lbl) = label_opt { sp = sp.with_label(lbl); }
    if filled { sp.set_filled(true); }

    figure.add_scatter_plot(sp);
    let hold = state().lock().unwrap().hold_on; if hold { store_current_figure(figure.clone()); }
    let res = execute_plot(figure.clone()); if !hold { store_current_figure(figure); }
    res
}

#[runtime_builtin(name = "bar")]
fn bar_builtin(rest: Vec<Value>) -> Result<String, String> {
    // MATLAB-like forms supported:
    // bar(Y) vertical, categories 1..n
    // bar(X,Y) with explicit categories/numeric X
    // bar(..., 'stacked'|'grouped')
    // name-value: 'DisplayName','Color','BarWidth','Orientation'('vertical'|'horizontal')
    if rest.is_empty() { return Err("bar: expected data".to_string()); }

    // Parse X (optional), Y (required), and trailing options
    #[allow(unused_assignments)]
    let mut i = 0usize;
    let mut x_opt: Option<Vec<f64>> = None;
    let mut x_labels_opt: Option<Vec<String>> = None; // categorical labels
    let y: Tensor;
    // Handle X,Y forms: numeric or categorical X
    if rest.len() >= 2 {
        match (&rest[0], &rest[1]) {
            (Value::Tensor(_), Value::Tensor(_)) => {
                let x_t: Tensor = std::convert::TryInto::try_into(&rest[0])?;
                let y_t: Tensor = std::convert::TryInto::try_into(&rest[1])?;
                x_opt = Some(x_t.data.clone());
                y = y_t; i = 2;
            }
            (Value::Cell(ca), Value::Tensor(y_t)) => {
                // Categorical X as cell array of strings
                let mut labels: Vec<String> = Vec::with_capacity(ca.data.len());
                for h in &ca.data { if let Value::String(s) = unsafe { &*h.as_raw() } { labels.push(s.clone()); } else { return Err("bar: categorical X must be array of strings".to_string()); } }
                x_labels_opt = Some(labels);
                y = y_t.clone(); i = 2;
            }
            _ => {
                y = std::convert::TryInto::try_into(&rest[0])?; i = 1;
            }
        }
    } else {
        y = std::convert::TryInto::try_into(&rest[0])?; i = 1;
    }

    // Options
    let mut stacked = false;
    let mut _grouped = true;
    let mut orientation = runmat_plot::plots::bar::Orientation::Vertical;
    let mut color: Option<glam::Vec4> = None;
    let mut edge_color: Option<glam::Vec4> = None;
    let mut edge_width: Option<f32> = None;
    let mut bar_width: Option<f32> = None;
    let mut display_name: Option<String> = None;

    #[allow(unused_assignments)]
    while i < rest.len() {
        if let Some(s) = get_string_value(&rest[i]) {
            let k = s.to_ascii_lowercase();
            match k.as_str() {
                "stacked" => { stacked = true; _grouped = false; i += 1; continue; }
                "grouped" => { _grouped = true; stacked = false; i += 1; continue; }
                _ => {
                    if i + 1 < rest.len() {
                        match k.as_str() {
                            "displayname" => { if let Some(v) = get_string_value(&rest[i+1]) { display_name = Some(v); i += 2; continue; } }
                            "color" | "facecolor" => { if let Some(c) = parse_color_value(&rest[i+1]) { color = Some(c); i += 2; continue; } }
                            "edgecolor" => { if let Some(c) = parse_color_value(&rest[i+1]) { edge_color = Some(c); i += 2; continue; } }
                            "barwidth" => { let w: f64 = std::convert::TryInto::try_into(&rest[i+1])?; bar_width = Some(w as f32); i += 2; continue; }
                            "orientation" => { if let Some(v) = get_string_value(&rest[i+1]) { let v = v.to_ascii_lowercase(); orientation = if v == "horizontal" { runmat_plot::plots::bar::Orientation::Horizontal } else { runmat_plot::plots::bar::Orientation::Vertical }; i += 2; continue; } }
                            "linewidth" => { let w: f64 = std::convert::TryInto::try_into(&rest[i+1])?; edge_width = Some(w as f32); i += 2; continue; }
                            "linestyle" => { /* not applicable to filled bars; ignore or store later */ i += 2; continue; }
                            _ => {}
                        }
                    }
                }
            }
        }
        break;
    }

    // Build bars
    let mut figure = take_or_new_figure().with_title("Bar Chart").with_labels("Categories", "Values").with_grid(true);

    // If Y is a vector → single series
    if y.cols() == 1 || y.shape.len() <= 1 {
        let data = y.data.clone();
        let labels: Vec<String> = if let Some(lbls) = x_labels_opt.clone() { if lbls.len() != data.len() { return Err("bar: length(X) must equal length(Y)".to_string()); } lbls } else if let Some(xv) = &x_opt { xv.iter().map(|v| format!("{v}")).collect() } else { (1..=data.len()).map(|i| format!("{i}")).collect() };
        let mut bar = BarChart::new(labels, data).map_err(|e| format!("bar: {e}"))?;
        if let Some(c) = color { bar.set_color(c); }
        if let Some(w) = bar_width { bar.set_bar_width(w); }
        bar = bar.with_orientation(orientation);
        if let Some(name) = display_name { bar = bar.with_label(name); }
        if let Some(ec) = edge_color { bar.set_outline_color(ec); }
        if let Some(ew) = edge_width { bar.set_outline_width(ew); }
        figure.add_bar_chart(bar);
    } else {
        // Matrix Y: columns are series
        let rows = y.rows(); let cols = y.cols();
        let base_labels: Vec<String> = if let Some(lbls) = x_labels_opt.clone() { if lbls.len() != rows { return Err("bar: length(X) must equal rows(Y)".to_string()); } lbls } else if let Some(xv) = &x_opt { xv.iter().map(|v| format!("{v}")).collect() } else { (1..=rows).map(|i| format!("{i}")).collect() };

        if stacked {
            // Build stacked with separate positive/negative accumulators
            let mut pos_acc = vec![0.0f64; rows];
            let mut neg_acc = vec![0.0f64; rows];
            for c in 0..cols {
                let series: Vec<f64> = (0..rows).map(|r| y.data[r + c * rows]).collect();
                // Compute offsets per row depending on sign
                let mut offsets = vec![0.0f64; rows];
                for r in 0..rows {
                    let v = series[r];
                    if v.is_sign_negative() { offsets[r] = neg_acc[r]; neg_acc[r] += v; } else { offsets[r] = pos_acc[r]; pos_acc[r] += v; }
                }
                let mut bar = BarChart::new(base_labels.clone(), series).map_err(|e| format!("bar: {e}"))?;
                bar = bar.with_orientation(orientation).with_stack_offsets(offsets);
                if let Some(ca) = color { bar.set_color(ca); }
                if let Some(w) = bar_width { bar.set_bar_width(w); }
                if let Some(ec) = edge_color { bar.set_outline_color(ec); }
                if let Some(ew) = edge_width { bar.set_outline_width(ew); }
                let name = display_name.clone().unwrap_or_else(|| format!("Series {}", c + 1));
                bar = bar.with_label(name);
                figure.add_bar_chart(bar);
            }
        } else {
            // Grouped bars: one series per column with grouped index
            for c in 0..cols {
                let series: Vec<f64> = (0..rows).map(|r| y.data[r + c * rows]).collect();
                let mut bar = BarChart::new(base_labels.clone(), series).map_err(|e| format!("bar: {e}"))?;
                bar = bar.with_orientation(orientation).with_group(c, cols);
                if let Some(ca) = color { bar.set_color(ca); }
                if let Some(w) = bar_width { bar.set_bar_width(w); }
                if let Some(ec) = edge_color { bar.set_outline_color(ec); }
                if let Some(ew) = edge_width { bar.set_outline_width(ew); }
                let name = display_name.clone().unwrap_or_else(|| format!("Series {}", c + 1));
                bar = bar.with_label(name);
                figure.add_bar_chart(bar);
            }
        }
    }

    execute_plot(figure)
}

#[runtime_builtin(name = "barh")]
fn barh_builtin(rest: Vec<Value>) -> Result<String, String> {
    // Forward to bar with Orientation set to Horizontal if not provided
    // If user passes Orientation explicitly, it will override below; otherwise we append it
    let mut args = rest.clone();
    // Only add orientation if not already present
    let mut has_orientation = false; let mut i = 0usize; while i + 1 < args.len() { if let Value::String(s) = &args[i] { if s.eq_ignore_ascii_case("orientation") { has_orientation = true; break; } } i += 1; }
    if !has_orientation { args.push(Value::String("Orientation".to_string())); args.push(Value::String("horizontal".to_string())); }
    bar_builtin(args)
}

#[runtime_builtin(name = "hist")]
fn hist_builtin(values: Tensor) -> Result<String, String> {
    // Delegate to histogram with defaults to honor name-value variations in one codepath
    histogram_builtin(vec![Value::Tensor(values)])
}

// 3D Plotting Functions (placeholders retained)
#[runtime_builtin(name = "surf")]
fn surf_builtin(x: Tensor, y: Tensor, z: Tensor) -> Result<String, String> {
    let x_data = extract_numeric_vector(&x); let y_data = extract_numeric_vector(&y); let z_data_flat = extract_numeric_vector(&z); let grid_size = (z_data_flat.len() as f64).sqrt() as usize; if grid_size * grid_size != z_data_flat.len() { return Err("Z data must form a square grid".to_string()); }
    let mut z_grid = Vec::new(); for i in 0..grid_size { let mut row = Vec::new(); for j in 0..grid_size { row.push(z_data_flat[i * grid_size + j]); } z_grid.push(row); }
    let surface = SurfacePlot::new(x_data, y_data, z_grid).map_err(|e| format!("Failed to create surface plot: {e}"))?.with_colormap(ColorMap::Viridis).with_label("Surface");
    Ok(format!("3D Surface plot created with {} points", surface.len()))
}

#[runtime_builtin(name = "scatter3")]
fn scatter3_builtin(_x: Tensor, _y: Tensor, _z: Tensor) -> Result<String, String> {
    Err("scatter3: 3D scatter not yet supported in renderer (point cloud module removed)".to_string())
}

#[runtime_builtin(name = "mesh")]
fn mesh_builtin(x: Tensor, y: Tensor, z: Tensor) -> Result<String, String> { let result = surf_builtin(x, y, z)?; Ok(result.replace("Surface", "Mesh (wireframe)")) }


// ---------- Additional 2D plot types ----------

#[runtime_builtin(name = "errorbar")]
fn errorbar_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty() { return Err("errorbar: expected arguments".to_string()); }
    // Parse data vectors
    let x_opt: Option<Vec<f64>>;
    let y: Vec<f64>;
    let mut idx: usize;
    let mut el: Option<Vec<f64>> = None;
    let mut eu: Option<Vec<f64>> = None;
    // Accept forms: (Y), (X,Y), (X,Y,E), (X,Y,EL,EU), (Y,E), (Y,EL,EU)
    fn to_vec(v: &Value) -> Result<Vec<f64>, String> { match v { Value::Tensor(t) => Ok(t.data.clone()), _ => Err("errorbar: expected numeric vector".to_string()) } }
    if rest.len() >= 2 && matches!(&rest[0], Value::Tensor(_)) && matches!(&rest[1], Value::Tensor(_)) {
        let xv = to_vec(&rest[0])?; let yv = to_vec(&rest[1])?; x_opt = Some(xv); y = yv; idx = 2;
    } else {
        y = to_vec(&rest[0])?; x_opt = None; idx = 1;
    }
    // Optional errors
    if idx < rest.len() && matches!(&rest[idx], Value::Tensor(_)) { el = Some(to_vec(&rest[idx])?); idx += 1; }
    if idx < rest.len() && matches!(&rest[idx], Value::Tensor(_)) { eu = Some(to_vec(&rest[idx])?); idx += 1; }
    let n = y.len(); if n == 0 { return Err("errorbar: empty Y".to_string()); }
    let xv = x_opt.unwrap_or_else(|| (1..=n).map(|k| k as f64).collect()); if xv.len() != n { return Err("errorbar: length(X) must equal length(Y)".to_string()); }
    let elv = match (el, eu.clone()) { (Some(a), _) if a.len() == n => a, _ => vec![0.0; n] };
    let euv = match (eu, elv.is_empty()) { (Some(b), _) if b.len() == n => b, _ => elv.clone() };
    // Name-Value
    let mut color: Option<glam::Vec4> = None; let mut lw: Option<f32> = None; let mut cap: Option<f32> = None; let mut label: Option<String> = None;
    while idx + 1 < rest.len() {
        if let Some(k) = get_string_value(&rest[idx]) {
            match k.to_ascii_lowercase().as_str() {
                "color" => { if let Some(c) = parse_color_value(&rest[idx+1]) { color = Some(c); idx += 2; continue; } }
                "linewidth" => { let w: f64 = std::convert::TryInto::try_into(&rest[idx+1])?; lw = Some(w as f32); idx += 2; continue; }
                "capsize" => { let w: f64 = std::convert::TryInto::try_into(&rest[idx+1])?; cap = Some(w as f32); idx += 2; continue; }
                "displayname" => { if let Some(s) = get_string_value(&rest[idx+1]) { label = Some(s); idx += 2; continue; } }
                _ => {}
            }
        }
        break;
    }
    let mut plot = runmat_plot::plots::ErrorBar::new(xv, y, elv, euv).map_err(|e| format!("errorbar: {e}"))?;
    if let Some(c) = color { plot = plot.with_style(c, lw.unwrap_or(1.0), cap.unwrap_or(0.02)); } else { plot = plot.with_style(glam::Vec4::new(0.0,0.0,0.0,1.0), lw.unwrap_or(1.0), cap.unwrap_or(0.02)); }
    if let Some(s) = label { plot = plot.with_label(s); }
    let mut figure = take_or_new_figure().with_title("Error Bar").with_grid(true);
    figure.add_errorbar(plot);
    execute_plot(figure)
}

#[runtime_builtin(name = "stairs")]
fn stairs_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty() { return Err("stairs: expected Y or X,Y".to_string()); }
    fn to_vec(v: &Value) -> Result<Vec<f64>, String> { match v { Value::Tensor(t) => Ok(t.data.clone()), _ => Err("stairs: expected numeric vector".to_string()) } }
    let (x, y, mut idx) = if rest.len() >= 2 && matches!(&rest[0], Value::Tensor(_)) && matches!(&rest[1], Value::Tensor(_)) { (Some(to_vec(&rest[0])?), to_vec(&rest[1])?, 2usize) } else { (None, to_vec(&rest[0])?, 1usize) };
    let n = y.len(); if n == 0 { return Err("stairs: empty Y".to_string()); }
    let xv = x.unwrap_or_else(|| (1..=n).map(|k| k as f64).collect()); if xv.len() != n { return Err("stairs: length(X) must equal length(Y)".to_string()); }
    let mut color: Option<glam::Vec4> = None; let mut lw: Option<f32> = None; let mut label: Option<String> = None;
    while idx + 1 < rest.len() { if let Some(k) = get_string_value(&rest[idx]) { match k.to_ascii_lowercase().as_str() { "color" => { if let Some(c)=parse_color_value(&rest[idx+1]){color=Some(c); idx+=2; continue;} } "linewidth"=>{ let w:f64=std::convert::TryInto::try_into(&rest[idx+1])?; lw=Some(w as f32); idx+=2; continue;} "displayname"=>{ if let Some(s)=get_string_value(&rest[idx+1]){ label=Some(s); idx+=2; continue;} } _=>{} } } break; }
    let mut plot = runmat_plot::plots::StairsPlot::new(xv, y).map_err(|e| format!("stairs: {e}"))?;
    plot = plot.with_style(color.unwrap_or(glam::Vec4::new(0.0,0.5,1.0,1.0)), lw.unwrap_or(1.0)); if let Some(s)=label { plot = plot.with_label(s); }
    let mut figure = take_or_new_figure().with_title("Stairs").with_grid(true);
    figure.add_stairs_plot(plot);
    execute_plot(figure)
}

#[runtime_builtin(name = "stem")]
fn stem_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty() { return Err("stem: expected Y or X,Y".to_string()); }
    fn to_vec(v: &Value) -> Result<Vec<f64>, String> { match v { Value::Tensor(t) => Ok(t.data.clone()), _ => Err("stem: expected numeric vector".to_string()) } }
    let (x, y, mut idx) = if rest.len() >= 2 && matches!(&rest[0], Value::Tensor(_)) && matches!(&rest[1], Value::Tensor(_)) { (Some(to_vec(&rest[0])?), to_vec(&rest[1])?, 2usize) } else { (None, to_vec(&rest[0])?, 1usize) };
    let n = y.len(); if n == 0 { return Err("stem: empty Y".to_string()); }
    let xv = x.unwrap_or_else(|| (1..=n).map(|k| k as f64).collect()); if xv.len() != n { return Err("stem: length(X) must equal length(Y)".to_string()); }
    let mut color: Option<glam::Vec4> = None; let mut mcolor: Option<glam::Vec4> = None; let mut base: Option<f64> = None; let mut label: Option<String> = None;
    while idx + 1 < rest.len() { if let Some(k) = get_string_value(&rest[idx]) { match k.to_ascii_lowercase().as_str() { "color"=>{ if let Some(c)=parse_color_value(&rest[idx+1]){ color=Some(c); idx+=2; continue; } } "markerfacecolor"=>{ if let Some(c)=parse_color_value(&rest[idx+1]){ mcolor=Some(c); idx+=2; continue; } } "basevalue"=>{ let b:f64=std::convert::TryInto::try_into(&rest[idx+1])?; base=Some(b); idx+=2; continue; } "displayname"=>{ if let Some(s)=get_string_value(&rest[idx+1]){ label=Some(s); idx+=2; continue; } } _=>{} } } break; }
    let mut plot = runmat_plot::plots::StemPlot::new(xv, y).map_err(|e| format!("stem: {e}"))?;
    plot = plot.with_style(color.unwrap_or(glam::Vec4::new(0.0,0.0,0.0,1.0)), mcolor.unwrap_or(glam::Vec4::new(0.0,0.5,1.0,1.0)), base.unwrap_or(0.0)); if let Some(s)=label { plot = plot.with_label(s); }
    let mut figure = take_or_new_figure().with_title("Stem").with_grid(true);
    figure.add_stem_plot(plot);
    execute_plot(figure)
}

#[runtime_builtin(name = "area")]
fn area_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty() { return Err("area: expected Y or X,Y".to_string()); }
    // Accept matrix Y: columns are series. By default, stack (MATLAB area stacks by default).
    let (xv_opt, y_tensor, mut idx) = if rest.len()>=2 && matches!(&rest[0],Value::Tensor(_)) && matches!(&rest[1],Value::Tensor(_)) { (Some(<Tensor as std::convert::TryFrom<&Value>>::try_from(&rest[0])?.data), <Tensor as std::convert::TryFrom<&Value>>::try_from(&rest[1])?, 2usize) } else { (None, <Tensor as std::convert::TryFrom<&Value>>::try_from(&rest[0])?, 1usize) };
    let rows = y_tensor.rows(); let cols = y_tensor.cols(); if rows==0 { return Err("area: empty Y".to_string()); }
    let xv = if let Some(xd) = xv_opt { if xd.len()!=rows { return Err("area: length(X) must equal rows(Y)".to_string()); } xd } else { (1..=rows).map(|k| k as f64).collect() };
    let mut color:Option<glam::Vec4>=None; let mut base:Option<f64>=None; let mut label:Option<String>=None; let mut stacked = true;
    while idx < rest.len(){
        if let Some(k)=get_string_value(&rest[idx]){
            let kl = k.to_ascii_lowercase();
            if kl=="stacked" { stacked = true; idx += 1; continue; }
            if kl=="grouped" { stacked = false; idx += 1; continue; }
            if idx+1 < rest.len() {
                match kl.as_str() {
                    "facecolor"|"color"=>{ if let Some(c)=parse_color_value(&rest[idx+1]){ color=Some(c); idx+=2; continue; } }
                    "basevalue"=>{ let b:f64=std::convert::TryInto::try_into(&rest[idx+1])?; base=Some(b); idx+=2; continue; }
                    "displayname"=>{ if let Some(s)=get_string_value(&rest[idx+1]){ label=Some(s); idx+=2; continue; } }
                    _=>{}
                }
            }
        }
        break;
    }
    let mut figure = take_or_new_figure().with_title("Area").with_grid(true);
    if cols<=1 || !stacked {
        // Single or grouped: one area per column with same baseline
        let yv = if cols<=1 { y_tensor.data.clone() } else { (0..rows).map(|r| y_tensor.data[r]).collect() };
        let mut plot = runmat_plot::plots::AreaPlot::new(xv.clone(), yv).map_err(|e| format!("area: {e}"))?;
        plot = plot.with_style(color.unwrap_or(glam::Vec4::new(0.0,0.5,1.0,0.4)), base.unwrap_or(0.0)); if let Some(s)=label.clone(){ plot = plot.with_label(s); }
        figure.add_area_plot(plot);
        if cols>1 && !stacked {
            for c in 1..cols {
                let series: Vec<f64> = (0..rows).map(|r| y_tensor.data[r + c*rows]).collect();
                let mut p = runmat_plot::plots::AreaPlot::new(xv.clone(), series).map_err(|e| format!("area: {e}"))?;
                p = p.with_style(color.unwrap_or(glam::Vec4::new(0.0,0.5,1.0,0.4)), base.unwrap_or(0.0));
                if label.is_none() { p = p.with_label(format!("Series {}", c+1)); }
                figure.add_area_plot(p);
            }
        }
    } else {
        // Stacked: accumulate baseline
        let mut acc = vec![base.unwrap_or(0.0); rows];
        for c in 0..cols {
            let series: Vec<f64> = (0..rows).map(|r| y_tensor.data[r + c*rows]).collect();
            let y_top: Vec<f64> = (0..rows).map(|r| acc[r] + series[r]).collect();
            let mut p = runmat_plot::plots::AreaPlot::new(xv.clone(), y_top.clone()).map_err(|e| format!("area: {e}"))?;
            p = p.with_style(color.unwrap_or(glam::Vec4::new(0.0,0.5,1.0,0.4)), 0.0);
            if label.is_none() { p = p.with_label(format!("Series {}", c+1)); }
            figure.add_area_plot(p);
            // Update baseline for next layer
            acc = y_top;
        }
    }
    execute_plot(figure)
}

#[runtime_builtin(name = "histogram")]
fn histogram_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty() { return Err("histogram: expected data vector".to_string()); }
    // Parse data
    let data: Vec<f64> = match &rest[0] { Value::Tensor(t) => t.data.clone(), _ => return Err("histogram: expected numeric vector".to_string()) };
    if data.is_empty() { return Err("histogram: empty data".to_string()); }
    // Defaults
    let mut num_bins: Option<usize> = None;
    let mut bin_width: Option<f64> = None;
    let mut bin_edges: Option<Vec<f64>> = None;
    let mut bin_limits: Option<(f64,f64)> = None;
    let mut normalization: String = "count".to_string();
    let mut face_color: Option<glam::Vec4> = None;
    let mut edge_color: Option<glam::Vec4> = None;
    let mut display_name: Option<String> = None;

    let mut i = 1usize;
    while i < rest.len() {
        if let Some(k) = get_string_value(&rest[i]) {
            let kl = k.to_ascii_lowercase();
            match kl.as_str() {
                "numbins" => { if i+1<rest.len() { let n: f64 = std::convert::TryInto::try_into(&rest[i+1])?; num_bins = Some(n.max(1.0) as usize); i+=2; continue; } }
                "binwidth" => { if i+1<rest.len() { let w: f64 = std::convert::TryInto::try_into(&rest[i+1])?; if w>0.0 { bin_width = Some(w); } i+=2; continue; } }
                "binedges" => { if i+1<rest.len() { if let Value::Tensor(t) = &rest[i+1] { if t.data.len()>=2 { bin_edges = Some(t.data.clone()); } } i+=2; continue; } }
                "binlimits" => { if i+1<rest.len() { if let Value::Tensor(t) = &rest[i+1] { if t.data.len()>=2 { bin_limits = Some((t.data[0], t.data[1])); } } i+=2; continue; } }
                "normalization" => { if i+1<rest.len() { if let Some(s) = get_string_value(&rest[i+1]) { normalization = s.to_ascii_lowercase(); } i+=2; continue; } }
                // MATLAB synonyms convenience
                "norm" => { if i+1<rest.len() { if let Some(s) = get_string_value(&rest[i+1]) { normalization = s.to_ascii_lowercase(); } i+=2; continue; } }
                "facecolor"|"color" => { if i+1<rest.len() { if let Some(c) = parse_color_value(&rest[i+1]) { face_color = Some(c); } i+=2; continue; } }
                "edgecolor" => { if i+1<rest.len() { if let Some(c) = parse_color_value(&rest[i+1]) { edge_color = Some(c); } i+=2; continue; } }
                "displayname" => { if i+1<rest.len() { if let Some(s) = get_string_value(&rest[i+1]) { display_name = Some(s); } i+=2; continue; } }
                _ => {}
            }
        }
        break;
    }

    // Determine limits
    let dmin = data.iter().copied().fold(f64::INFINITY, f64::min);
    let dmax = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (mut lo, mut hi) = bin_limits.unwrap_or((dmin, dmax));
    if !lo.is_finite() || !hi.is_finite() || hi <= lo { lo = dmin; hi = dmax; }

    // Compute edges
    let edges: Vec<f64> = if let Some(edges) = bin_edges {
        edges
    } else if let Some(w) = bin_width {
        let nb = (((hi - lo)/w).ceil().max(1.0)) as usize;
        (0..=nb).map(|k| lo + k as f64 * w).collect()
    } else {
        let nb = num_bins.unwrap_or(10);
        let w = (hi - lo) / nb as f64;
        (0..=nb).map(|k| lo + k as f64 * w).collect()
    };
    if edges.len()<2 { return Err("histogram: insufficient bin edges".to_string()); }

    // Count
    let bins = edges.len()-1;
    let mut counts = vec![0f64; bins];
    for &v in &data {
        if v < edges[0] || v > edges[bins] || !v.is_finite() { continue; }
        let mut idx = bins; for b in 0..bins { if v >= edges[b] && v < edges[b+1] { idx = b; break; } }
        if idx==bins && (v-edges[bins]).abs() < f64::EPSILON { idx = bins-1; }
        if idx < bins { counts[idx] += 1.0; }
    }
    // Normalization
    let total: f64 = counts.iter().sum();
    let widths: Vec<f64> = edges.windows(2).map(|w| w[1]-w[0]).collect();
    match normalization.as_str() {
        "probability" => { if total>0.0 { for c in &mut counts { *c /= total; } } }
        "pdf" => { if total>0.0 { for (c,w) in counts.iter_mut().zip(widths.iter()) { if *w>0.0 { *c = *c / (total * *w); } } } }
        "countdensity" => { for (c,w) in counts.iter_mut().zip(widths.iter()) { if *w>0.0 { *c = *c / *w; } } }
        "cumcount" => { for k in 1..counts.len() { counts[k] += counts[k-1]; } }
        "cdf" => { for k in 1..counts.len() { counts[k] += counts[k-1]; } if total>0.0 { for c in &mut counts { *c /= total; } } }
        _ => {}
    }

    // Labels
    let labels: Vec<String> = edges.windows(2).map(|w| format!("[{:.3},{:.3})", w[0], w[1])).collect();
    let mut bar = BarChart::new(labels, counts).map_err(|e| format!("histogram: {e}"))?;
    if let Some(fc) = face_color { bar.set_color(fc); }
    if let Some(ec) = edge_color { bar.set_outline_color(ec); }
    if let Some(s) = display_name { bar = bar.with_label(s); }
    let mut figure = take_or_new_figure().with_title("Histogram").with_labels("Values", "Frequency").with_grid(true);
    figure.add_bar_chart(bar);
    execute_plot(figure)
}

#[runtime_builtin(name = "quiver")]
fn quiver_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.len()<4 { return Err("quiver: expected X,Y,U,V".to_string()); }
    let x: Tensor = std::convert::TryInto::try_into(&rest[0])?; let y: Tensor = std::convert::TryInto::try_into(&rest[1])?; let u: Tensor = std::convert::TryInto::try_into(&rest[2])?; let v: Tensor = std::convert::TryInto::try_into(&rest[3])?;
    if x.data.len()!=y.data.len() || y.data.len()!=u.data.len() || u.data.len()!=v.data.len() { return Err("quiver: X,Y,U,V lengths must match".to_string()); }
    let mut color: Option<glam::Vec4> = None; let mut scale: Option<f32> = None; let mut head: Option<f32> = None; let mut label: Option<String> = None; let mut i=4usize;
    while i+1<rest.len(){ if let Some(k)=get_string_value(&rest[i]){ match k.to_ascii_lowercase().as_str(){ "color"=>{ if let Some(c)=parse_color_value(&rest[i+1]){ color=Some(c); i+=2; continue;} } "scale"=>{ let s:f64=std::convert::TryInto::try_into(&rest[i+1])?; scale=Some(s as f32); i+=2; continue; } "headsize"=>{ let h:f64=std::convert::TryInto::try_into(&rest[i+1])?; head=Some(h as f32); i+=2; continue; } "displayname"=>{ if let Some(s)=get_string_value(&rest[i+1]){ label=Some(s); i+=2; continue; } } _=>{} } } break; }
    let mut plot = runmat_plot::plots::QuiverPlot::new(x.data, y.data, u.data, v.data).map_err(|e| format!("quiver: {e}"))?;
    plot = plot.with_style(color.unwrap_or(glam::Vec4::new(0.0,0.0,0.0,1.0)), 1.0, scale.unwrap_or(1.0), head.unwrap_or(0.1)); if let Some(s)=label{ plot = plot.with_label(s); }
    let mut figure = take_or_new_figure().with_title("Quiver").with_grid(true);
    figure.add_quiver_plot(plot);
    execute_plot(figure)
}

#[runtime_builtin(name = "pie")]
fn pie_builtin(rest: Vec<Value>) -> Result<String, String> {
    if rest.is_empty(){ return Err("pie: expected values".to_string()); }
    let y: Tensor = std::convert::TryInto::try_into(&rest[0])?;
    let mut colors: Option<Vec<glam::Vec4>> = None; let mut label: Option<String> = None; let mut i=1usize;
    while i+1<=rest.len(){ if i<rest.len(){ if let Some(k)=get_string_value(&rest[i]){ let kl=k.to_ascii_lowercase(); if kl=="colors" && i+1<rest.len(){ if let Value::Tensor(t)=&rest[i+1]{ let mut out=Vec::new(); let n=t.data.len()/3; for c in 0..n{ let r=t.data[c]; let g=t.data[c+n]; let b=t.data[c+2*n]; out.push(glam::Vec4::new(r as f32,g as f32,b as f32,1.0)); } colors=Some(out); i+=2; continue; } } if kl=="displayname" && i+1<rest.len(){ if let Some(s)=get_string_value(&rest[i+1]){ label=Some(s); i+=2; continue; } } } }
        break; }
    let mut plot = runmat_plot::plots::PieChart::new(y.data.clone(), colors).map_err(|e| format!("pie: {e}"))?; if let Some(s)=label{ plot = plot.with_label(s); }
    let mut figure = take_or_new_figure().with_title("Pie"); figure.add_pie_chart(plot); execute_plot(figure)
}

#[cfg(test)]
mod scatter_varargs_tests {
    use super::*;
    use runmat_builtins::{Tensor, Value};
    fn approx_eq(a: [f32;4], b: [f32;4], eps: f32) -> bool {
        (a[0]-b[0]).abs() <= eps && (a[1]-b[1]).abs() <= eps && (a[2]-b[2]).abs() <= eps && (a[3]-b[3]).abs() <= eps
    }
    fn distinct_colors(cols: &Vec<[f32;4]>, eps: f32) -> usize {
        let mut reps: Vec<[f32;4]> = Vec::new();
        'outer: for c in cols.iter() {
            for r in reps.iter() { if approx_eq(*c, *r, eps) { continue 'outer; } }
            reps.push(*c);
        }
        reps.len()
    }

    fn tensor(data: &[f64], rows: usize, cols: usize) -> Value {
        let t = Tensor { data: data.to_vec(), shape: vec![rows, cols], rows, cols };
        Value::Tensor(t)
    }

    fn clear_state() {
        let mut st = state().lock().unwrap();
        st.current_figure = None;
        st.hold_on = false;
        st.x_log = false;
        st.y_log = false;
    }

    #[test]
    fn test_scatter_y_only_parses() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let y = tensor(&[1.0, 2.0, 3.0], 3, 1);
        assert!(scatter_builtin(vec![y]).is_ok());
        let fig_opt = state().lock().unwrap().current_figure.clone();
        assert!(fig_opt.is_some());
        let mut fig = fig_opt.unwrap();
        let rds = fig.render_data();
        assert_eq!(rds.len(), 1);
        // points pipeline
        assert!(matches!(rds[0].pipeline_type, runmat_plot::core::renderer::PipelineType::Points));
        assert_eq!(rds[0].vertices.len(), 3);
    }

    #[test]
    fn test_scatter_xy_s_scalar_sets_sizes() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let x = tensor(&[0.0, 1.0, 2.0], 3, 1);
        let y = tensor(&[1.0, 2.0, 3.0], 3, 1);
        let s = tensor(&[25.0], 1, 1); // scalar area
        assert!(scatter_builtin(vec![x, y, s]).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        assert_eq!(rds.len(), 1);
        // vertex normal.z stores size in px
        for v in &rds[0].vertices { assert!((v.normal[2] - 5.0).abs() < 1e-6); }
    }

    #[test]
    fn test_scatter_xy_s_vector_sets_sizes() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let x = tensor(&[0.0, 1.0, 2.0], 3, 1);
        let y = tensor(&[1.0, 2.0, 3.0], 3, 1);
        let s = tensor(&[4.0, 9.0, 16.0], 3, 1);
        assert!(scatter_builtin(vec![x, y, s]).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        let sizes: Vec<f32> = rds[0].vertices.iter().map(|v| v.normal[2]).collect();
        assert_eq!(sizes, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scatter_xy_c_rgb_matrix_sets_colors() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let x = tensor(&[0.0, 1.0, 2.0], 3, 1);
        let y = tensor(&[1.0, 2.0, 3.0], 3, 1);
        let c = tensor(&[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ], 3, 3);
        assert!(scatter_builtin(vec![x, y, c]).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        assert_eq!(rds[0].vertices.len(), 3);
        let cols: Vec<[f32;4]> = rds[0].vertices.iter().map(|v| v.color).collect();
        // Expect at least two distinct vertex colors when RGB per-point provided
        assert!(distinct_colors(&cols, 1e-5) >= 2);
    }

    #[test]
    fn test_scatter_xy_c_scalar_maps_colormap() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let x = tensor(&[0.0, 1.0, 2.0, 3.0], 4, 1);
        let y = tensor(&[1.0, 2.0, 3.0, 4.0], 4, 1);
        let c = tensor(&[0.0, 0.33, 0.66, 1.0], 4, 1);
        assert!(scatter_builtin(vec![x, y, c]).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        assert_eq!(rds[0].vertices.len(), 4);
        let cols: Vec<[f32;4]> = rds[0].vertices.iter().map(|v| v.color).collect();
        // Expect at least two distinct vertex colors when scalar CData mapped by colormap
        assert!(distinct_colors(&cols, 1e-5) >= 2);
    }

    #[test]
    fn test_scatter_filled_facecolor_alpha_applied() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let x = tensor(&[0.0, 1.0], 2, 1);
        let y = tensor(&[1.0, 2.0], 2, 1);
        let args = vec![x, y, Value::String("filled".to_string()), Value::String("MarkerFaceColor".to_string()), Value::String("r".to_string())];
        assert!(scatter_builtin(args).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        // material.albedo.w controls face color mix alpha
        assert_eq!(rds[0].material.albedo.w, 1.0);
    }

    #[test]
    fn test_scatter_marker_and_linewidth_nv() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let x = tensor(&[0.0, 1.0], 2, 1);
        let y = tensor(&[1.0, 2.0], 2, 1);
        let args = vec![x, y, Value::String("Marker".to_string()), Value::String("^".to_string()), Value::String("LineWidth".to_string()), Value::from(2.5f64)];
        assert!(scatter_builtin(args).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        // Triangle => metallic=2.0; linewidth stored in roughness
        assert!((rds[0].material.metallic - 2.0).abs() < 1e-6);
        assert!((rds[0].material.roughness - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_scatter_markerface_edgecolor_synonyms() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let x = tensor(&[0.0, 1.0], 2, 1);
        let y = tensor(&[1.0, 2.0], 2, 1);
        let args = vec![x, y, Value::String("FaceColor".to_string()), Value::String("g".to_string()), Value::String("EdgeColor".to_string()), Value::String("k".to_string())];
        assert!(scatter_builtin(args).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        // Face green, edge black
        assert_eq!(rds[0].material.albedo.truncate(), glam::Vec3::new(0.0,1.0,0.0));
        assert_eq!(rds[0].material.emissive.truncate(), glam::Vec3::new(0.0,0.0,0.0));
    }
}

#[cfg(test)]
mod plot_compat_tests {
    use super::*;
    use runmat_builtins::{Tensor, Value};

    fn tensor(data: &[f64], rows: usize, cols: usize) -> Value {
        let t = Tensor { data: data.to_vec(), shape: vec![rows, cols], rows, cols };
        Value::Tensor(t)
    }

    fn clear_state() {
        let mut st = state().lock().unwrap();
        st.current_figure = None;
        st.hold_on = false;
        st.x_log = false;
        st.y_log = false;
    }

    #[test]
    fn test_plot_y_only_vector_creates_line() {
        std::env::set_var("RUSTMAT_PLOT_MODE", "headless");
        clear_state();
        let y = tensor(&[1.0, 4.0, 9.0], 3, 1);
        assert!(plot_varargs(vec![y]).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        assert_eq!(rds.len(), 1);
        assert!(matches!(rds[0].pipeline_type, runmat_plot::core::renderer::PipelineType::Lines));
    }

    #[test]
    fn test_plot_matrix_y_columns_multiple_series() {
        std::env::set_var("RUSTMAT_PLOT_MODE", "headless");
        clear_state();
        // rows=3, cols=2 → 2 series
        let y = tensor(&[1.0, 2.0, 3.0,   4.0, 5.0, 6.0], 3, 2);
        assert!(plot_varargs(vec![y]).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        // Expect two line series
        assert_eq!(rds.len(), 2);
        assert!(rds.iter().all(|rd| matches!(rd.pipeline_type, runmat_plot::core::renderer::PipelineType::Lines)));
    }

    #[test]
    fn test_plot_style_marker_only_creates_scatter() {
        std::env::set_var("RUSTMAT_PLOT_MODE", "headless");
        clear_state();
        let x = tensor(&[1.0, 2.0, 3.0], 3, 1);
        let y = tensor(&[1.0, 0.0, 1.0], 3, 1);
        let sty = Value::String("o".to_string());
        assert!(plot_varargs(vec![x, y, sty]).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        // Should have at least one points pipeline
        assert!(rds.iter().any(|rd| matches!(rd.pipeline_type, runmat_plot::core::renderer::PipelineType::Points)));
    }

    #[test]
    fn test_hold_on_appends_series() {
        std::env::set_var("RUSTMAT_PLOT_MODE", "headless");
        clear_state();
        assert!(hold_builtin(vec![Value::String("on".to_string())]).is_ok());
        let x = tensor(&[1.0, 2.0, 3.0], 3, 1);
        let y1 = tensor(&[1.0, 2.0, 3.0], 3, 1);
        let y2 = tensor(&[3.0, 2.0, 1.0], 3, 1);
        assert!(plot_varargs(vec![x.clone(), y1]).is_ok());
        assert!(plot_varargs(vec![x, y2]).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        assert!(rds.len() >= 2);
    }

    #[test]
    fn test_semilogx_validation_and_ticks() {
        std::env::set_var("RUSTMAT_PLOT_MODE", "headless");
        clear_state();
        // negative X should error
        let x = tensor(&[-1.0, 1.0], 2, 1);
        let y = tensor(&[1.0, 2.0], 2, 1);
        assert!(semilogx_builtin(vec![x, y]).is_err());
        // valid positive X should succeed
        let x = tensor(&[1.0, 10.0, 100.0], 3, 1);
        let y = tensor(&[1.0, 2.0, 3.0], 3, 1);
        assert!(semilogx_builtin(vec![x, y]).is_ok());
    }

    #[test]
    fn test_semilogy_and_loglog_validation() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        // semilogy: negative Y should error
        let x = tensor(&[1.0, 2.0], 2, 1);
        let y_bad = tensor(&[-1.0, 2.0], 2, 1);
        assert!(semilogy_builtin(vec![x.clone(), y_bad]).is_err());
        // semilogy valid
        let y = tensor(&[1.0, 10.0], 2, 1);
        assert!(semilogy_builtin(vec![x.clone(), y]).is_ok());
        // loglog: both must be positive
        let y2 = tensor(&[1.0, 100.0], 2, 1);
        assert!(loglog_builtin(vec![x, y2]).is_ok());
    }

    #[test]
    fn test_export_parity_mixed_line_scatter() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        // Build a figure with a line and a scatter
        let mut fig = runmat_plot::plots::Figure::new().with_title("Parity").with_labels("X","Y").with_grid(true);
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let mut lp = runmat_plot::plots::LinePlot::new(x.clone(), y.clone()).unwrap();
        lp = lp.with_style(glam::Vec4::new(0.0,0.5,1.0,1.0), 2.0, runmat_plot::plots::line::LineStyle::Dashed).with_label("L");
        fig.add_line_plot(lp);
        let mut sp = runmat_plot::plots::ScatterPlot::new(x, y).unwrap().with_style(glam::Vec4::new(1.0,0.0,0.0,1.0), 8.0, runmat_plot::plots::scatter::MarkerStyle::Circle).with_label("S");
        sp.set_filled(true);
        fig.add_scatter_plot(sp);
        // Export
        let tmp = std::env::var("CARGO_TARGET_TMPDIR").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        let mut path = std::path::PathBuf::from(tmp);
        path.push("mixed_parity.png");
        let r = runmat_plot::show_plot_unified(fig, Some(path.clone().to_str().unwrap()));
        assert!(r.is_ok());
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[test]
    fn test_export_parity_loglog() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        // Create loglog data
        let x = tensor(&[1.0, 10.0, 100.0, 1000.0], 4, 1);
        let y = tensor(&[1.0, 100.0, 10000.0, 1000000.0], 4, 1);
        assert!(loglog_builtin(vec![x, y]).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        let tmp = std::env::var("CARGO_TARGET_TMPDIR").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        let mut path = std::path::PathBuf::from(tmp);
        path.push("loglog_parity.png");
        let r = runmat_plot::show_plot_unified(fig, Some(path.clone().to_str().unwrap()));
        assert!(r.is_ok());
        assert!(path.exists());
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 1000);
    }

    #[test]
    fn test_plot_style_strings_multi_series_line_and_marker() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let x = tensor(&[0.0, 1.0, 2.0, 3.0], 4, 1);
        let y1 = tensor(&[0.0, 1.0, 0.0, 1.0], 4, 1);
        let y2 = tensor(&[1.0, 0.0, 1.0, 0.0], 4, 1);
        let s1 = Value::String("r--o".to_string());
        let s2 = Value::String("g*".to_string());
        assert!(plot_varargs(vec![x.clone(), y1, s1, x, y2, s2]).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        let num_lines = rds.iter().filter(|rd| matches!(rd.pipeline_type, runmat_plot::core::renderer::PipelineType::Lines)).count();
        let num_points = rds.iter().filter(|rd| matches!(rd.pipeline_type, runmat_plot::core::renderer::PipelineType::Points)).count();
        assert!(num_lines >= 1);
        assert!(num_points >= 1);
    }

    #[test]
    fn test_legend_order_with_hold_on() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        assert!(hold_builtin(vec![Value::String("on".to_string())]).is_ok());
        let x = tensor(&[0.0, 1.0, 2.0], 3, 1);
        let y1 = tensor(&[0.0, 1.0, 0.0], 3, 1);
        let y2 = tensor(&[1.0, 0.0, 1.0], 3, 1);
        assert!(plot_varargs(vec![x.clone(), y1.clone(), Value::String("DisplayName".to_string()), Value::String("First".to_string())]).is_ok());
        assert!(plot_varargs(vec![x, y2, Value::String("DisplayName".to_string()), Value::String("Second".to_string())]).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        let legend = fig.legend_entries();
        assert!(legend.len() >= 2);
        assert_eq!(legend[0].label, "First");
        assert_eq!(legend[1].label, "Second");
    }

    #[test]
    fn test_legend_builtin_on_off_and_labels() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        // create a simple plot so a figure exists
        let y = tensor(&[1.0, 2.0, 3.0], 3, 1);
        assert!(plot_varargs(vec![y]).is_ok());
        // legend off
        assert!(legend_builtin(vec![Value::String("off".to_string())]).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        assert!(!fig.legend_enabled);
        // legend on
        assert!(legend_builtin(vec![Value::String("on".to_string())]).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        assert!(fig.legend_enabled);
        // add a second series
        assert!(hold_builtin(vec![Value::String("on".to_string())]).is_ok());
        let y2 = tensor(&[3.0, 2.0, 1.0], 3, 1);
        let x = tensor(&[1.0, 2.0, 3.0], 3, 1);
        assert!(plot_varargs(vec![x, y2]).is_ok());
        // labels
        assert!(legend_builtin(vec![Value::String("L1".to_string()), Value::String("L2".to_string())]).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        let entries = fig.legend_entries();
        assert!(entries.len() >= 2);
        assert_eq!(entries[0].label, "L1");
        assert_eq!(entries[1].label, "L2");
    }

    #[test]
    fn test_axis_equal_and_tight() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        // create a simple line
        let y = tensor(&[1.0, 4.0, 9.0], 3, 1);
        assert!(plot_varargs(vec![y]).is_ok());
        // equal
        assert!(axis_builtin(vec![Value::String("equal".to_string())]).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        assert!(fig.axis_equal);
        // tight -> sets limits
        assert!(axis_builtin(vec![Value::String("tight".to_string())]).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        assert!(fig.x_limits.is_some());
        assert!(fig.y_limits.is_some());
    }

    #[test]
    fn test_imagesc_colormap_caxis_colorbar_semantics() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        // Z 2x2
        let z = tensor(&[0.0, 1.0, 2.0, 3.0], 2, 2);
        assert!(imagesc_builtin(vec![z]).is_ok());
        // Set colormap to jet
        assert!(colormap_builtin("jet".to_string()).is_ok());
        // Enable colorbar
        assert!(colorbar_builtin(vec![]).is_ok());
        // Set caxis limits
        assert!(caxis_builtin(vec![tensor(&[0.0, 3.0], 1, 2)]).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        assert!(fig.colorbar_enabled);
        if let runmat_plot::plots::surface::ColorMap::Jet = fig.colormap { /* ok */ } else { panic!("expected jet colormap"); }
        assert_eq!(fig.color_limits, Some((0.0, 3.0)));
    }
}

#[cfg(test)]
mod runtime_sweep_tests {
    use super::*;
    use runmat_builtins::{Tensor, Value};

    fn tensor(data: &[f64], rows: usize, cols: usize) -> Value {
        let t = Tensor { data: data.to_vec(), shape: vec![rows, cols], rows, cols };
        Value::Tensor(t)
    }

    fn clear_state() {
        let mut st = state().lock().unwrap();
        st.current_figure = None;
        st.hold_on = false;
        st.x_log = false;
        st.y_log = false;
        st.current_axes = 0;
    }

    #[test]
    fn test_plot_multi_series_with_display_names_and_legend() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let x1 = tensor(&[1.0, 2.0, 3.0], 3, 1);
        let y1 = tensor(&[1.0, 4.0, 9.0], 3, 1);
        let x2 = tensor(&[1.0, 2.0, 3.0], 3, 1);
        let y2 = tensor(&[9.0, 4.0, 1.0], 3, 1);
        let args = vec![
            x1, y1, Value::String("DisplayName".to_string()), Value::String("A".to_string()),
            x2, y2, Value::String("DisplayName".to_string()), Value::String("B".to_string()),
        ];
        assert!(plot_varargs(args).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        let entries = fig.legend_entries();
        assert!(entries.len() >= 2);
        assert_eq!(entries[0].label, "A");
        assert_eq!(entries[1].label, "B");
    }

    #[test]
    fn test_subplot_assignment_two_axes() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        assert!(subplot_builtin(vec![Value::from(1.0f64), Value::from(2.0f64), Value::from(1.0f64)]).is_ok());
        let x = tensor(&[1.0, 2.0], 2, 1);
        let y = tensor(&[2.0, 3.0], 2, 1);
        assert!(plot_varargs(vec![x.clone(), y.clone()]).is_ok());
        assert!(subplot_builtin(vec![Value::from(1.0f64), Value::from(2.0f64), Value::from(2.0f64)]).is_ok());
        assert!(plot_varargs(vec![x, y]).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        let map = fig.plot_axes_indices().to_vec();
        assert!(map.len() >= 2);
        // Expect first plot -> axes 0, second plot -> axes 1
        assert_eq!(map[0], 0);
        assert_eq!(map[1], 1);
    }

    #[test]
    fn test_grid_and_title_xlabel_ylabel_mutations() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let y = tensor(&[1.0,2.0], 2, 1);
        assert!(plot_varargs(vec![y]).is_ok());
        assert!(grid_builtin(vec![Value::String("off".to_string())]).is_ok());
        assert!(title_builtin("Hello".to_string()).is_ok());
        assert!(xlabel_builtin("Xlab".to_string()).is_ok());
        assert!(ylabel_builtin("Ylab".to_string()).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        assert!(!fig.grid_enabled);
        assert_eq!(fig.title.as_deref(), Some("Hello"));
        assert_eq!(fig.x_label.as_deref(), Some("Xlab"));
        assert_eq!(fig.y_label.as_deref(), Some("Ylab"));
    }

    #[test]
    fn test_axis_auto_resets_limits() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let y = tensor(&[1.0, 2.0, 3.0], 3, 1);
        assert!(plot_varargs(vec![y]).is_ok());
        assert!(xlim_builtin(vec![tensor(&[0.0, 10.0], 1, 2)]).is_ok());
        assert!(ylim_builtin(vec![tensor(&[0.0, 10.0], 1, 2)]).is_ok());
        assert!(axis_builtin(vec![Value::String("auto".to_string())]).is_ok());
        let fig = state().lock().unwrap().current_figure.clone().unwrap();
        assert!(fig.x_limits.is_none());
        assert!(fig.y_limits.is_none());
    }

    #[test]
    fn test_plot_style_string_r_dash_o_creates_line_and_markers() {
        std::env::set_var("RUNMAT_PLOT_MODE", "headless");
        clear_state();
        let x = tensor(&[0.0, 1.0, 2.0], 3, 1);
        let y = tensor(&[1.0, 0.0, 1.0], 3, 1);
        let sty = Value::String("r--o".to_string());
        assert!(plot_varargs(vec![x, y, sty]).is_ok());
        let mut fig = state().lock().unwrap().current_figure.clone().unwrap();
        let rds = fig.render_data();
        let num_lines = rds.iter().filter(|rd| matches!(rd.pipeline_type, runmat_plot::core::renderer::PipelineType::Lines)).count();
        let num_points = rds.iter().filter(|rd| matches!(rd.pipeline_type, runmat_plot::core::renderer::PipelineType::Points)).count();
        assert!(num_lines >= 1 && num_points >= 1);
    }
}

#[runtime_builtin(name = "subplot")]
fn subplot_builtin(rest: Vec<Value>) -> Result<String, String> {
    // Accept subplot(m,n,p) where m,n,p are scalars (1-based). Also allow two-arg (m,n) -> p=1 and one-arg index shorthand (mnp) not supported here.
    let (m, n, p) = match rest.len() {
        1 => {
            let idx: f64 = std::convert::TryInto::try_into(&rest[0]).map_err(|_| "subplot: expected scalar".to_string())?;
            if idx < 1.0 { return Err("subplot: index must be >= 1".to_string()); }
            (1usize, idx as usize, 1usize)
        }
        2 => {
            let m: f64 = std::convert::TryInto::try_into(&rest[0]).map_err(|_| "subplot: expected m".to_string())?;
            let n: f64 = std::convert::TryInto::try_into(&rest[1]).map_err(|_| "subplot: expected n".to_string())?;
            if m < 1.0 || n < 1.0 { return Err("subplot: m,n must be >= 1".to_string()); }
            (m as usize, n as usize, 1usize)
        }
        3 => {
            let m: f64 = std::convert::TryInto::try_into(&rest[0]).map_err(|_| "subplot: expected m".to_string())?;
            let n: f64 = std::convert::TryInto::try_into(&rest[1]).map_err(|_| "subplot: expected n".to_string())?;
            let p: f64 = std::convert::TryInto::try_into(&rest[2]).map_err(|_| "subplot: expected p".to_string())?;
            if m < 1.0 || n < 1.0 || p < 1.0 { return Err("subplot: m,n,p must be >= 1".to_string()); }
            (m as usize, n as usize, p as usize)
        }
        _ => return Err("subplot: expected (m,n,p)".to_string()),
    };
    let mut st = state().lock().unwrap();
    let mut fig = st.current_figure.take().unwrap_or_else(Figure::new);
    fig.set_subplot_grid(m, n);
    st.current_axes = p.saturating_sub(1);
    // MATLAB semantics: subplot selection becomes current axes; does not clear figure, but subsequent plot() targets selected axes.
    // Ensure existing plots remain; no-op here.
    st.current_figure = Some(fig);
    Ok(format!("subplot({}, {}, {})", m, n, p))
}

#[runtime_builtin(name = "gca")]
fn gca_builtin() -> Result<String, String> { let st = state().lock().unwrap(); Ok(format!("axes {}", st.current_axes + 1)) }
