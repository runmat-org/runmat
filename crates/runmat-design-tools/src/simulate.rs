//! `design.simulate`: runs a `.circuit` file's `simulation` section on the native engine and
//! returns the probed waveforms, the measurements and the measured operations.

use crate::{kind_of, text_of, Files, ToolError};
use runmat_circuit::runtime::{design_and_pool, CompileOptions};
use runmat_circuit_sim::{montecarlo, SimResult, MEASUREMENT_VERSION};
use runmat_circuit_spice::study;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SimulateInput {
    /// Workspace path of the `.circuit`.
    pub path: String,
    /// The `simulation` section to run; required when the file has more than one.
    #[serde(default)]
    pub simulation: Option<String>,
    /// At most this many samples per waveform (default 200; evenly spaced picks of the solver's points).
    #[serde(default)]
    pub max_points: Option<usize>,
    /// Histogram bins for Monte Carlo measurements (default 20).
    #[serde(default)]
    pub bins: Option<usize>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Signal {
    /// The probed net.
    pub net: String,
    /// Real values (operating point, DC sweep, transient), in volts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f64>>,
    /// AC magnitude in dB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub magnitude_db: Option<Vec<f64>>,
    /// AC phase in degrees (unwrapped).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_deg: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisOut {
    pub name: String,
    /// `op`, `dc`, `ac`, `tran` (a Monte Carlo analysis shows its first run).
    pub kind: String,
    /// `time` (s), `frequency` (Hz), the swept source (V or A), or empty for an operating point.
    pub x_name: String,
    pub x: Vec<f64>,
    pub signals: Vec<Signal>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MeasureOut {
    pub name: String,
    pub of: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Measured operations, as decimal strings, in the simulation-usage categories.
#[derive(Clone, Debug, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct UsageOut {
    pub measurement_version: String,
    pub assembly: String,
    pub factorization: String,
    pub solve: String,
    pub other: String,
    pub total: String,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SimulateOutput {
    pub simulation: String,
    pub engine: String,
    pub analyses: Vec<AnalysisOut>,
    pub measures: Vec<MeasureOut>,
    pub usage: UsageOut,
    pub warnings: Vec<String>,
}

fn pick<T: Clone>(v: &[T], max: usize) -> Vec<T> {
    if v.len() <= max || max < 2 {
        return v.to_vec();
    }
    (0..max).map(|k| v[k * (v.len() - 1) / (max - 1)].clone()).collect()
}

pub fn simulate(input: &SimulateInput, files: &Files) -> Result<SimulateOutput, ToolError> {
    let kind = kind_of(&input.path)?;
    if !kind.is_circuit() {
        return Err(ToolError::invalid(format!("design.simulate runs on a .circuit; {} is a {kind:?} file", input.path)));
    }
    let text = text_of(files, &input.path)?;
    let (design, pool) = design_and_pool(&input.path, text, files, &CompileOptions::default()).map_err(|d| {
        ToolError::new("design_invalid", d.iter().map(|x| x.message.clone()).collect::<Vec<_>>().join("; "))
    })?;
    let name = match &input.simulation {
        Some(n) => n.clone(),
        None => match design.simulations.keys().collect::<Vec<_>>().as_slice() {
            [one] => (*one).clone(),
            [] => return Err(ToolError::new("no_simulation", format!("{} has no `simulation` section; add one to the circuit block", input.path))),
            many => return Err(ToolError::invalid(format!("{} has several simulations ({}); name one", input.path, many.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")))),
        },
    };
    let dir = input.path.rsplit_once('/').map(|(d, _)| d.to_string()).unwrap_or_default();
    let resolve = |lib: &str| files.get(lib).or_else(|| files.get(&format!("{dir}/{lib}"))).cloned();
    let s = study::run(&design, &pool, &name, &resolve, None).map_err(|e| ToolError::new("simulation_failed", e))?;
    let max = input.max_points.unwrap_or(200).max(2);
    let mut analyses = vec![];
    for (k, r) in &s.results {
        let (kind, x_name, x, signals) = match r {
            SimResult::Op(m) => (
                "op",
                String::new(),
                vec![],
                s.probes.iter().map(|(net, sig)| Signal { net: net.clone(), values: m.get(sig).map(|v| vec![*v]), magnitude_db: None, phase_deg: None }).collect(),
            ),
            SimResult::Dc(w) | SimResult::Tran(w) => (
                if matches!(r, SimResult::Tran(_)) { "tran" } else { "dc" },
                w.x_name.clone(),
                pick(&w.x, max),
                s.probes.iter().map(|(net, sig)| Signal { net: net.clone(), values: w.get(sig).map(|v| pick(v, max)), magnitude_db: None, phase_deg: None }).collect(),
            ),
            SimResult::Ac(w) => (
                "ac",
                "frequency".into(),
                pick(&w.freq, max),
                s.probes
                    .iter()
                    .map(|(net, sig)| Signal { net: net.clone(), values: None, magnitude_db: w.db(sig).map(|v| pick(&v, max)), phase_deg: w.phase_deg(sig).map(|v| pick(&v, max)) })
                    .collect(),
            ),
        };
        analyses.push(AnalysisOut { name: k.clone(), kind: kind.into(), x_name, x, signals });
    }
    let bins = input.bins.unwrap_or(20).max(1);
    let measures = s
        .measures
        .iter()
        .map(|(n, m)| {
            let (edges, counts) = if m.runs.is_empty() { (None, None) } else {
                let (e, c) = montecarlo::histogram(&m.runs, bins);
                (Some(e), Some(c))
            };
            MeasureOut {
                name: n.clone(),
                of: m.of.clone(),
                value: m.value,
                runs: m.stats.as_ref().map(|st| st.n),
                mean: m.stats.as_ref().map(|st| st.mean),
                std: m.stats.as_ref().map(|st| st.std),
                min: m.stats.as_ref().map(|st| st.min),
                max: m.stats.as_ref().map(|st| st.max),
                histogram_edges: edges,
                histogram_counts: counts,
                error: m.error.clone(),
            }
        })
        .collect();
    let u = &s.usage;
    Ok(SimulateOutput {
        simulation: name,
        engine: format!("runmat-circuit-sim {}", env!("CARGO_PKG_VERSION")),
        analyses,
        measures,
        usage: UsageOut {
            measurement_version: MEASUREMENT_VERSION.into(),
            assembly: u.assembly.to_string(),
            factorization: u.factorization.to_string(),
            solve: u.solve.to_string(),
            other: u.other.to_string(),
            total: u.total().map(|t| t.to_string()).unwrap_or_else(|_| "overflow".into()),
        },
        warnings: s.warnings,
    })
}
