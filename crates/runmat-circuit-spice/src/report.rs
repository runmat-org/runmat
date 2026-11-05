//! The JSON report of a study: what `design.simulate` returns to the agent and what the
//! Desktop's simulation pane draws. One shape for both, so the pane and the agent read the same
//! numbers.
//!
//! Waveforms are the probes, net voltages and pin currents: volts or amperes for operating
//! points, DC sweeps and transients; dB and unwrapped degrees for AC. A Monte Carlo analysis carries its first runs as an overlay family.
//! Every measurement names its unit. Usage counts are decimal strings, so JavaScript reads them
//! as `BigInt` without rounding.

use crate::study::Study;
use runmat_circuit::simulation::{SimAnalysis, SimMeasure, SimProbe, SimulationDef};
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
    /// `voltage` (a net) or `current` (into a pin).
    pub quantity: String,
    /// Real values (operating point, DC sweep, transient), in volts or amperes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f64>>,
    /// AC magnitude in dB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub magnitude_db: Option<Vec<f64>>,
    /// AC phase in degrees (unwrapped).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_deg: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct AnalysisOut {
    pub name: String,
    /// `op`, `dc`, `ac`, `tran` or `monte_carlo`.
    pub kind: String,
    /// The analysis a Monte Carlo reruns, and its kind.
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
    /// Monte Carlo: the first runs' signals, each on `x` of its own run.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub runs: Vec<RunOut>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct RunOut {
    pub x: Vec<f64>,
    pub signals: Vec<Signal>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(JsonSchema))]
#[serde(rename_all = "camelCase")]
pub struct MeasureOut {
    pub name: String,
    pub of: String,
    /// What it computes (`bandwidth`, `rise_time`, …), as the section writes it.
    pub kind: String,
    /// The net it reads (`OUT`), and `voltage`.
    pub probe: String,
    pub quantity: String,
    /// `V`, `s`, `Hz`, `dB`, `°`, `%`, or empty for Q.
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
    /// The specification limits, in `unit`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_max: Option<f64>,
    /// A single value: whether it is inside the limits.
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
fn spec_result(d: Option<&SimMeasure>, value: Option<f64>, runs: &[f64]) -> (Option<bool>, Option<usize>, Option<f64>) {
    let Some(d) = d.filter(|d| d.spec_min.is_some() || d.spec_max.is_some()) else { return (None, None, None) };
    let ok = |v: f64| v.is_finite() && d.spec_min.is_none_or(|lo| v >= lo) && d.spec_max.is_none_or(|hi| v <= hi);
    if runs.is_empty() {
        return (value.map(ok), None, None);
    }
    let within = runs.iter().filter(|v| ok(**v)).count();
    (None, Some(within), Some(100.0 * within as f64 / runs.len() as f64))
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
    for (k, r) in &s.results {
        let (x_name, x_unit, x, sigs) = signals(s, r, max);
        let (kind, of, of_kind, runs) = match def.analyses.get(k) {
            Some(SimAnalysis::MonteCarlo { of, .. }) => {
                let base = def.analyses.get(of).map(|a| a.word().to_string());
                let runs = s.monte_carlo.get(k).map(|mc| {
                    mc.kept.iter().take(o.overlay_runs).map(|r| {
                        let (_, _, x, signals) = signals(s, r, max);
                        RunOut { x, signals }
                    }).collect()
                });
                ("monte_carlo".to_string(), Some(of.clone()), base, runs.unwrap_or_default())
            }
            _ => (kind_of(r).to_string(), None, None, vec![]),
        };
        analyses.push(AnalysisOut { name: k.clone(), kind, of, of_kind, x_name, x_unit, x, signals: sigs, runs });
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
            let (pass, within, yield_percent) = spec_result(d, m.value, &m.runs);
            MeasureOut {
                name: n.clone(),
                of: m.of.clone(),
                kind: d.map(|d| d.kind.word().to_string()).unwrap_or_default(),
                probe: d.map(|d| d.net.clone()).unwrap_or_default(),
                quantity: "voltage".into(),
                unit: d.map(|d| d.kind.unit().to_string()).unwrap_or_default(),
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
