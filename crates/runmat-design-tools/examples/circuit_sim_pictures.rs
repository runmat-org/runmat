//! Circuit simulation pictures for the site: the daq channel's step response, its AC magnitude
//! and phase, and a Monte Carlo histogram of its −3 dB frequency against the section's spec limits
//! (with the yield), drawn through runmat-plot at 2× in the light and dark themes.
//!
//! `cargo run --release -p runmat-design-tools --features render --example circuit_sim_pictures -- <out dir>`

use glam::Vec4;
use runmat_circuit::fixture::standard_pool;
use runmat_circuit::syntax::parse_circuit;
use runmat_circuit_sim::montecarlo::histogram;
use runmat_circuit_spice::study;
use runmat_plot::export::image::{ImageExportSettings, ImageExporter};
use runmat_plot::plots::line::LineStyle;
use runmat_plot::plots::{AreaPlot, Figure, LinePlot, ReferenceLine, ReferenceLineOrientation};
use runmat_plot::styling::{PlotThemeConfig, ThemeVariant};

const DAQ: &str = include_str!("../../runmat-circuit/tests/projects/daq/daq.circuit");
const SECTION: &str = include_str!("../tests/fixtures/daq_front_end.simulation");

const W: u32 = 2400;
const H: u32 = 1400;

fn theme(dark: bool) -> PlotThemeConfig {
    let mut t = PlotThemeConfig { variant: if dark { ThemeVariant::ModernDark } else { ThemeVariant::ClassicLight }, ..PlotThemeConfig::default() };
    let ty = &mut t.typography;
    for s in [&mut ty.title_font_size, &mut ty.subtitle_font_size, &mut ty.axis_label_font_size, &mut ty.tick_label_font_size, &mut ty.legend_font_size] {
        *s *= 2.0;
    }
    t.layout.data_line_width *= 2.0;
    t.layout.axis_line_width *= 2.0;
    t.layout.grid_line_width *= 2.0;
    t
}

/// Series colors that hold contrast on both backgrounds.
fn color(k: usize, dark: bool) -> Vec4 {
    let light = [[0.0, 0.45, 0.85], [0.85, 0.37, 0.0], [0.0, 0.6, 0.45], [0.75, 0.15, 0.45]];
    let darkp = [[0.35, 0.68, 1.0], [1.0, 0.6, 0.25], [0.3, 0.85, 0.65], [0.95, 0.45, 0.7]];
    let c = if dark { darkp[k % 4] } else { light[k % 4] };
    Vec4::new(c[0], c[1], c[2], 1.0)
}

fn save(fig: &mut Figure, dark: bool, path: &str) {
    let mut ex = pollster::block_on(ImageExporter::with_settings(ImageExportSettings { width: W, height: H, ..Default::default() })).unwrap();
    ex.set_theme_config(theme(dark));
    let png = pollster::block_on(ex.render_png_bytes(fig)).unwrap();
    std::fs::write(path, png).unwrap();
    println!("wrote {path}");
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let end = DAQ.rfind('}').unwrap();
    let text = format!("{}{SECTION}{}", &DAQ[..end], &DAQ[end..]);
    let parsed = parse_circuit("/p/daq.circuit", &text);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let design = parsed.design.unwrap();
    let s = study::run(&design, &standard_pool(), "front_end", &|_| None, None).unwrap();
    let label = |net: &str| match net {
        "IN" => "IN (channel input)",
        "INF" => "INF (after RS·CF, 159 kHz)",
        "AMP" => "AMP (OPA171, gain 2)",
        "AIN" => "AIN (ADC input, after RO·CO)",
        n => n,
    }
    .to_string();
    let f3 = s.measures["f3db"].value.unwrap();
    let rise = s.measures["rise"].value.unwrap();
    for dark in [false, true] {
        let th = if dark { "dark" } else { "light" };
        // Transient.
        let w = s.results["step"].real().unwrap();
        let mut fig = Figure::new();
        fig.set_title(format!("daq channel step response: 10–90 % rise {:.2} µs at AIN", rise * 1e6));
        fig.set_axis_labels("Time (µs)", "Voltage (V)");
        fig.set_grid(true);
        fig.set_legend(true);
        let t: Vec<f64> = w.x.iter().map(|x| x * 1e6).collect();
        for (k, (probe, sig)) in s.probes.iter().enumerate() {
            let net = probe.name();
            let y = w.get(sig).unwrap().to_vec();
            let p = LinePlot::new(t.clone(), y).unwrap().with_style(color(k, dark), 2.5, LineStyle::Solid).with_label(label(&net));
            fig.add_line_plot(p);
        }
        save(&mut fig, dark, &format!("{out}/{th}-circuit-sim-transient.png"));
        // AC magnitude and phase.
        let a = s.results["response"].ac().unwrap();
        let mut fig = Figure::new();
        fig.set_subplot_grid(2, 1);
        fig.set_axes_title(0, format!("daq channel frequency response: −3 dB at {:.1} kHz", f3 / 1e3));
        fig.set_axes_labels(0, "Frequency (Hz)", "Magnitude (dB)");
        fig.set_axes_labels(1, "Frequency (Hz)", "Phase (°)");
        fig.set_axes_log_modes(0, true, false);
        fig.set_axes_log_modes(1, true, false);
        fig.set_grid(true);
        fig.set_legend(true);
        for (k, (probe, sig)) in s.probes.iter().enumerate().filter(|(_, (p, _))| p.name() != "IN") {
            let net = probe.name();
            let db = a.db(sig).unwrap();
            let ph = a.phase_deg(sig).unwrap();
            fig.add_line_plot_on_axes(LinePlot::new(a.freq.clone(), db).unwrap().with_style(color(k, dark), 2.5, LineStyle::Solid).with_label(label(&net)), 0);
            fig.add_line_plot_on_axes(LinePlot::new(a.freq.clone(), ph).unwrap().with_style(color(k, dark), 2.5, LineStyle::Solid).with_label(label(&net)), 1);
        }
        let g0 = a.db(&s.probes[3].1).unwrap()[0];
        let mark = LinePlot::new(vec![a.freq[0], a.freq[a.freq.len() - 1]], vec![g0 - 3.0103; 2]).unwrap().with_style(color(1, dark) * Vec4::new(1.0, 1.0, 1.0, 0.6), 1.5, LineStyle::Dashed).with_label("−3 dB");
        fig.add_line_plot_on_axes(mark, 0);
        save(&mut fig, dark, &format!("{out}/{th}-circuit-sim-ac.png"));
        // Monte Carlo histogram on a kHz axis: in-spec bins in the trace color, out-of-spec bins
        // in red, the spec limits, the mean and ±3σ as reference lines, the yield in the title.
        let m = &s.measures["f3db_spread"];
        let def = &design.simulations["front_end"].measures["f3db_spread"];
        let (lo, hi) = (def.spec_min.unwrap(), def.spec_max.unwrap());
        let st = m.stats.clone().unwrap();
        let within = m.runs.iter().filter(|v| **v >= lo && **v <= hi).count();
        let khz: Vec<f64> = m.runs.iter().map(|v| v / 1e3).collect();
        let (edges, counts) = histogram(&khz, 32);
        let (mut x, mut y_in, mut y_out) = (vec![], vec![], vec![]);
        for (k, c) in counts.iter().enumerate() {
            let out_of_spec = edges[k + 1] <= lo / 1e3 || edges[k] >= hi / 1e3;
            for e in [edges[k], edges[k + 1]] {
                x.push(e);
                y_in.push(if out_of_spec { 0.0 } else { *c as f64 });
                y_out.push(if out_of_spec { *c as f64 } else { 0.0 });
            }
        }
        let red = if dark { Vec4::new(1.0, 0.42, 0.42, 1.0) } else { Vec4::new(0.8, 0.12, 0.12, 1.0) };
        let mut fig = Figure::new();
        fig.set_title(format!(
            "−3 dB frequency over {} Monte Carlo runs (R ±1 %, C ±5 %): yield {:.1} % in {:.0}–{:.0} kHz",
            st.n,
            100.0 * within as f64 / st.n as f64,
            lo / 1e3,
            hi / 1e3
        ));
        fig.set_axis_labels("−3 dB frequency at AIN (kHz)", "Runs");
        fig.set_grid(true);
        fig.set_legend(true);
        fig.add_area_plot(AreaPlot::new(x.clone(), y_in).unwrap().with_style(color(0, dark) * Vec4::new(1.0, 1.0, 1.0, 0.8), 0.0).with_label("runs in spec"));
        fig.add_area_plot(AreaPlot::new(x, y_out).unwrap().with_style(red * Vec4::new(1.0, 1.0, 1.0, 0.8), 0.0).with_label("runs out of spec"));
        let line = |v: f64, c: Vec4, w: f32, style: LineStyle, name: Option<String>| {
            let mut l = ReferenceLine::new(ReferenceLineOrientation::Vertical, v).unwrap().with_style(c, w, style);
            l.display_name = name;
            l.color_auto = false;
            l
        };
        let fg = if dark { Vec4::new(0.9, 0.9, 0.92, 1.0) } else { Vec4::new(0.15, 0.15, 0.18, 1.0) };
        fig.add_reference_line_on_axes(line(lo / 1e3, red, 3.0, LineStyle::Solid, Some(format!("spec min {:.0} kHz", lo / 1e3))), 0);
        fig.add_reference_line_on_axes(line(hi / 1e3, red, 3.0, LineStyle::Solid, Some(format!("spec max {:.0} kHz", hi / 1e3))), 0);
        fig.add_reference_line_on_axes(line(st.mean / 1e3, fg, 2.0, LineStyle::Dashed, Some(format!("mean {:.1} kHz", st.mean / 1e3))), 0);
        fig.add_reference_line_on_axes(line((st.mean - 3.0 * st.std) / 1e3, fg, 1.5, LineStyle::Dotted, Some(format!("±3σ ({:.1} kHz σ)", st.std / 1e3))), 0);
        fig.add_reference_line_on_axes(line((st.mean + 3.0 * st.std) / 1e3, fg, 1.5, LineStyle::Dotted, None), 0);
        save(&mut fig, dark, &format!("{out}/{th}-circuit-sim-montecarlo.png"));
    }
}
