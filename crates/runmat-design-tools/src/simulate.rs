//! `design.simulate`: runs a `.circuit` file's `simulation` section on the native engine and
//! returns the probed waveforms, the measurements and the measured operations.

use crate::{kind_of, text_of, Files, ToolError};
use runmat_circuit::runtime::{design_and_pool, CompileOptions};
use runmat_circuit_spice::{report, study};
use schemars::JsonSchema;
use serde::Deserialize;

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
    /// The operations the run may spend, as a decimal string: the host's granted operations
    /// on its simulation lease. The run stops before it would pass them and returns what it
    /// solved, with `stopped` saying where. Omitted: no limit.
    #[serde(default)]
    pub limit_ops: Option<String>,
}

pub use runmat_circuit_spice::report::{AnalysisOut, MeasureOut, RunOut, Signal, SimulateOutput, ToleranceOut, UsageOut};

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
    let limit = match &input.limit_ops {
        Some(t) => Some(t.parse::<u64>().map_err(|_| ToolError::invalid(format!("limitOps is a decimal count of operations, not `{t}`")))?),
        None => None,
    };
    let s = study::run_with(&design, &pool, &name, &resolve, study::RunOptions { threads: None, limit }).map_err(|e| ToolError::new("simulation_failed", e))?;
    let def = &design.simulations[&name];
    let options = report::ReportOptions { max_points: input.max_points.unwrap_or(200), bins: input.bins.unwrap_or(20), ..Default::default() };
    Ok(report::report(&s, def, &options))
}
