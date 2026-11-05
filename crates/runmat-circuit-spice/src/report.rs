//! The JSON report of a study: what `design.simulate` returns to the agent and what the
//! Desktop's simulation pane draws. One shape for both, so the pane and the agent read the same
//! numbers.
//!
//! Waveforms are the probes, net voltages and pin currents: volts or amperes for operating
//! points, DC sweeps and transients; dB and unwrapped degrees for AC. A Monte Carlo analysis
//! carries its first runs as an overlay family, and a temperature sweep one run per temperature.
//! A noise analysis carries its output and input-referred densities (V/√Hz), the integrated
//! noise and the devices that contribute most.
//! Every measurement names its unit. Usage counts are decimal strings, so JavaScript reads them
//! as `BigInt` without rounding.

use crate::study::Study;
use runmat_circuit::simulation::{SimAnalysis, SimMeasure, SimProbe, SimulationDef};
use runmat_circuit_sim::analysis::noise::NoiseWave;
use runmat_circuit_sim::{montecarlo, SimResult, MEASUREMENT_VERSION};
use serde::Serialize;

#[cfg(feature = "schema")]
use schemars::JsonSchema;

#[derive(Clone, Copy, Debug)]
pub struct ReportOptions {
    /// At most this many samples per waveform (evenly spaced picks of the solver's points).
    pub max_points: usize,
    /// Histogram bins for Monte Carlo measurements.
    pub bins: usize,
    /// Monte Carlo runs drawn as an overlay family (at most the 20 a study keeps).
    pub overlay_runs: usize,
}

impl Default for ReportOptions {
    fn default() -> Self {
        ReportOptions { max_points: 200, bins: 20, overlay_runs: 20 }
    }
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct Signal {
    /// The probed net (`OUT`) or pin (`RO.2`).
    pub probe: String,
    /// `voltage` (a net), `current` (into a pin) or `noise` (a noise analysis's output density).
    pub quantity: String,
    /// Real values (operating point, DC sweep, transient), in volts or amperes; a noise
    /// density in V/√Hz.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f64>>,
    /// AC magnitude in dB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub magnitude_db: Option<Vec<f64>>,
    /// AC phase in degrees (unwrapped).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_deg: Option<Vec<f64>>,
}

/// A noise analysis's figures.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct NoiseOut {
    /// The output net and the input source the noise is referred to.
    pub output: String,
    pub input: String,
    /// Output and input-referred densities on `x`, in V/√Hz.
    pub output_density: Vec<f64>,
    pub input_density: Vec<f64>,
    /// The densities integrated over the solved band, in V rms.
    pub output_rms: f64,
    pub input_rms: f64,
    /// The devices that contribute most to the output noise, largest first.
    pub contributors: Vec<NoiseShare>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct NoiseShare {
    /// The deck element (`rs`, `xu.q1`).
    pub device: String,
    /// Its integrated output noise (V rms) and its share of the output noise power (percent).
    pub rms: f64,
    pub percent: f64,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct AnalysisOut {
    pub name: String,
    /// `op`, `dc`, `ac`, `tran`, `noise`, `monte_carlo` or `temperature`.
    pub kind: String,
    /// The analysis a Monte Carlo or temperature sweep reruns, and its kind.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub of: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub of_kind: Option<String>,
    /// `time`, `frequency`, the swept source, or empty for an operating point.
    pub x_name: String,
    /// `s`, `Hz`, `V` or `A`; empty for an operating point.
    pub x_unit: String,
    pub x: Vec<f64>,
    /// The probed nets (a Monte Carlo analysis: its first run).
    pub signals: Vec<Signal>,
    /// Monte Carlo: the first runs' signals, each on `x` of its own run. A temperature sweep:
    /// one run per temperature.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub runs: Vec<RunOut>,
    /// A noise analysis (a temperature sweep of one: its first temperature).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise: Option<NoiseOut>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct RunOut {
    pub x: Vec<f64>,
    pub signals: Vec<Signal>,
    /// A temperature sweep's run: its temperature in °C.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct MeasureOut {
    pub name: String,
    pub of: String,
    /// What it computes (`bandwidth`, `rise_time`, …), as the section writes it.
    pub kind: String,
    /// The net (`OUT`) or pin (`RO.2`) it reads, and `voltage` or `current`.
    pub probe: String,
    pub quantity: String,
    /// `V`, `A`, `s`, `Hz`, `dB`, `°`, `%`, `V/√Hz`, or empty for Q.
    pub unit: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// Monte Carlo: runs, mean, standard deviation, min, max and a histogram.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub histogram_edges: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub histogram_counts: Option<Vec<usize>>,
    /// A temperature sweep: the temperatures (°C) and the value at each (null where it failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperatures: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<Option<f64>>>,
    /// The specification limits, in `unit`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_max: Option<f64>,
    /// A single value, or every temperature of a sweep: whether it is inside the limits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pass: Option<bool>,
    /// A Monte Carlo: the runs inside the limits, and that count over the runs (in percent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub within: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub yield_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A tolerance the Monte Carlo applied.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct ToleranceOut {
    /// The element (`rs`) or the section statement's pattern (`R*`).
    pub elements: String,
    /// Band edges in percent of nominal (−1, +1 for ±1 %).
    pub minus_percent: f64,
    pub plus_percent: f64,
    /// `component` or `tolerance <name>`.
    pub from: String,
}

/// Measured operations, as decimal strings, in the simulation-usage categories.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct UsageOut {
    pub measurement_version: String,
    pub assembly: String,
    pub factorization: String,
    pub solve: String,
    pub other: String,
    pub total: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct SimulateOutput {
    pub simulation: String,
    pub engine: String,
    /// The probed nets, in the section's order.
    pub probes: Vec<String>,
    pub analyses: Vec<AnalysisOut>,
    pub measures: Vec<MeasureOut>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tolerances: Vec<ToleranceOut>,
    pub usage: UsageOut,
    pub warnings: Vec<String>,
    /// Set when the run stopped at its granted operations: what stopped and what didn't run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopped: Option<String>,
}

fn pick<T: Clone>(v: &[T], max: usize) -> Vec<T> {
    if v.len() <= max || max < 2 {
        return v.to_vec();
    }
    (0..max).map(|k| v[k * (v.len() - 1) / (max - 1)].clone()).collect()
}

/// Against a measure's specification limits: whether its value passes, and for a Monte Carlo the
/// runs inside them and the yield (a failed run, NaN, counts against it).
fn spec_result(d: Option<&SimMeasure>, value: Option<f64>, runs: &[f64], sweep: bool) -> (Option<bool>, Option<usize>, Option<f64>) {
    let Some(d) = d.filter(|d| d.spec_min.is_some() || d.spec_max.is_some()) else { return (None, None, None) };
    let ok = |v: f64| v.is_finite() && d.spec_min.is_none_or(|lo| v >= lo) && d.spec_max.is_none_or(|hi| v <= hi);
    if runs.is_empty() {
        return (value.map(ok), None, None);
    }
    if sweep {
        return (Some(runs.iter().all(|v| ok(*v))), Some(runs.iter().filter(|v| ok(**v)).count()), None);
    }
    let within = runs.iter().filter(|v| ok(**v)).count();
    (None, Some(within), Some(100.0 * within as f64 / runs.len() as f64))
}

/// The noise figures of `w`, the densities picked down to `max` points.
fn noise_out(w: &NoiseWave, output: &str, input: &str, max: usize) -> NoiseOut {
    let total = w.total_onoise();
    let band = |d: &[f64]| (1..w.freq.len().min(d.len())).map(|k| 0.5 * (d[k] + d[k - 1]) * (w.freq[k] - w.freq[k - 1])).sum::<f64>();
    let mut contributors: Vec<NoiseShare> = w
        .contributions
        .iter()
        .map(|(device, power)| {
            let p = band(power).max(0.0);
            NoiseShare { device: device.clone(), rms: p.sqrt(), percent: if total > 0.0 { 100.0 * p / (total * total) } else { 0.0 } }
        })
        .collect();
    contributors.sort_by(|a, b| b.rms.total_cmp(&a.rms));
    contributors.truncate(8);
    NoiseOut { output: output.into(), input: input.into(), output_density: pick(&w.onoise, max), input_density: pick(&w.inoise, max), output_rms: total, input_rms: w.total_inoise(), contributors }
}

/// A noise analysis drawn as one trace: the output density at its net.
fn noise_signal(n: &NoiseOut) -> Signal {
    Signal { probe: n.output.clone(), quantity: "noise".into(), values: Some(n.output_density.clone()), magnitude_db: None, phase_deg: None }
}

fn kind_of(r: &SimResult) -> &'static str {
    match r {
        SimResult::Op(_) => "op",
        SimResult::Dc(_) => "dc",
        SimResult::Ac(_) => "ac",
        SimResult::Tran(_) => "tran",
    }
}

fn signal(p: &SimProbe, values: Option<Vec<f64>>, magnitude_db: Option<Vec<f64>>, phase_deg: Option<Vec<f64>>) -> Signal {
    Signal { probe: p.name(), quantity: p.quantity().into(), values, magnitude_db, phase_deg }
}

/// The x axis and probed signals of one result.
fn signals(s: &Study, r: &SimResult, max: usize) -> (String, String, Vec<f64>, Vec<Signal>) {
    match r {
        SimResult::Op(m) => (
            String::new(),
            String::new(),
            vec![],
            s.probes.iter().map(|(p, sig)| signal(p, m.get(sig).map(|v| vec![*v]), None, None)).collect(),
        ),
        SimResult::Dc(w) | SimResult::Tran(w) => {
            let unit = if matches!(r, SimResult::Tran(_)) { "s" } else if w.x_name.starts_with('i') { "A" } else { "V" };
            (
                w.x_name.clone(),
                unit.into(),
                pick(&w.x, max),
                s.probes.iter().map(|(p, sig)| signal(p, w.get(sig).map(|v| pick(v, max)), None, None)).collect(),
            )
        }
        SimResult::Ac(w) => (
            "frequency".into(),
            "Hz".into(),
            pick(&w.freq, max),
            s.probes
                .iter()
                .map(|(p, sig)| signal(p, None, w.db(sig).map(|v| pick(&v, max)), w.phase_deg(sig).map(|v| pick(&v, max))))
                .collect(),
        ),
    }
}

/// The report of study `s`, run from section `def`.
pub fn report(s: &Study, def: &SimulationDef, o: &ReportOptions) -> SimulateOutput {
    let max = o.max_points.max(2);
    let mut analyses = vec![];
    let mut names: Vec<&String> = s.results.keys().chain(s.noise.keys()).chain(s.temperatures.keys()).collect();
    names.sort();
    names.dedup();
    for k in names {
        let a = def.analyses.get(k);
        let base = a.and_then(|a| a.reruns()).and_then(|of| def.analyses.get(of).map(|b| (of.to_string(), b)));
        let (of, of_kind) = match &base {
            Some((of, b)) => (Some(of.clone()), Some(b.word().to_string())),
            None => (None, None),
        };
        let noise_of = |w: &NoiseWave| match base.as_ref().map(|(_, b)| *b).or(a) {
            Some(SimAnalysis::Noise { output, input, .. }) => noise_out(w, output, input, max),
            _ => noise_out(w, "", "", max),
        };
        let mut out = match (s.results.get(k), s.noise.get(k)) {
            (_, Some(w)) => {
                let n = noise_of(w);
                AnalysisOut { name: k.clone(), kind: "noise".into(), of, of_kind, x_name: "frequency".into(), x_unit: "Hz".into(), x: pick(&w.freq, max), signals: vec![noise_signal(&n)], runs: vec![], noise: Some(n) }
            }
            (Some(r), None) => {
                let (x_name, x_unit, x, sigs) = signals(s, r, max);
                AnalysisOut { name: k.clone(), kind: kind_of(r).into(), of, of_kind, x_name, x_unit, x, signals: sigs, runs: vec![], noise: None }
            }
            (None, None) => AnalysisOut { name: k.clone(), kind: String::new(), of, of_kind, x_name: "frequency".into(), x_unit: "Hz".into(), x: vec![], signals: vec![], runs: vec![], noise: None },
        };
        match a {
            Some(SimAnalysis::MonteCarlo { .. }) => {
                out.kind = "monte_carlo".into();
                if let Some(mc) = s.monte_carlo.get(k) {
                    out.runs = mc.kept.iter().take(o.overlay_runs).map(|r| {
                        let (_, _, x, signals) = signals(s, r, max);
                        RunOut { x, signals, temperature: None }
                    }).collect();
                }
            }
            Some(SimAnalysis::Temperature { .. }) => {
                out.kind = "temperature".into();
                for p in s.temperatures.get(k).into_iter().flatten() {
                    let run = match (&p.result, &p.noise) {
                        (Some(r), _) => {
                            let (_, _, x, signals) = signals(s, r, max);
                            RunOut { x, signals, temperature: Some(p.temp_c) }
                        }
                        (None, Some(w)) => {
                            let n = noise_of(w);
                            if out.noise.is_none() {
                                (out.x_name, out.x_unit, out.x, out.signals) = ("frequency".into(), "Hz".into(), pick(&w.freq, max), vec![noise_signal(&n)]);
                                out.noise = Some(n.clone());
                            }
                            RunOut { x: pick(&w.freq, max), signals: vec![noise_signal(&n)], temperature: Some(p.temp_c) }
                        }
                        (None, None) => continue,
                    };
                    out.runs.push(run);
                }
            }
            _ => {}
        }
        analyses.push(out);
    }
    let bins = o.bins.max(1);
    let measures = s
        .measures
        .iter()
        .map(|(n, m)| {
            let (edges, counts) = if m.runs.is_empty() {
                (None, None)
            } else {
                let (e, c) = montecarlo::histogram(&m.runs, bins);
                (Some(e), Some(c))
            };
            let d = def.measures.get(n);
            let sweep = !m.temperatures.is_empty();
            let (pass, within, yield_percent) = spec_result(d, m.value, &m.runs, sweep);
            let (edges, counts) = if sweep { (None, None) } else { (edges, counts) };
            MeasureOut {
                name: n.clone(),
                of: m.of.clone(),
                kind: d.map(|d| d.kind.word().to_string()).unwrap_or_default(),
                probe: d.map(|d| d.probe.name()).unwrap_or_default(),
                quantity: d.map(|d| d.probe.quantity()).unwrap_or("voltage").into(),
                unit: d.map(|d| d.unit().to_string()).unwrap_or_default(),
                temperatures: sweep.then(|| m.temperatures.clone()),
                values: sweep.then(|| m.runs.iter().map(|v| v.is_finite().then_some(*v)).collect()),
                value: m.value,
                runs: m.stats.as_ref().map(|st| st.n),
                mean: m.stats.as_ref().map(|st| st.mean),
                std: m.stats.as_ref().map(|st| st.std),
                min: m.stats.as_ref().map(|st| st.min),
                max: m.stats.as_ref().map(|st| st.max),
                histogram_edges: edges,
                histogram_counts: counts,
                spec_min: d.and_then(|d| d.spec_min),
                spec_max: d.and_then(|d| d.spec_max),
                pass,
                within,
                yield_percent,
                error: m.error.clone(),
            }
        })
        .collect();
    let u = &s.usage;
    let pct = |f: f64| ((f - 1.0) * 1e6).round() / 1e4;
    SimulateOutput {
        simulation: s.simulation.clone(),
        engine: format!("runmat-circuit-sim {}", env!("CARGO_PKG_VERSION")),
        probes: s.probes.iter().map(|(p, _)| p.name()).collect(),
        analyses,
        measures,
        tolerances: s.tolerances.iter().map(|t| ToleranceOut { elements: t.elements.clone(), minus_percent: pct(t.band.0), plus_percent: pct(t.band.1), from: t.from.clone() }).collect(),
        usage: UsageOut {
            measurement_version: MEASUREMENT_VERSION.into(),
            assembly: u.assembly.operations().to_string(),
            factorization: u.factorization.operations().to_string(),
            solve: u.solve.operations().to_string(),
            other: u.other.operations().to_string(),
            total: u.total().map(|t| t.operations().to_string()).unwrap_or_else(|_| "overflow".into()),
        },
        warnings: s.warnings.clone(),
        stopped: s.stopped.clone(),
    }
}
