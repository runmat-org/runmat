//! `runmat-circuit-sim`: RunMat's circuit simulator, written from the published algorithms.
//!
//! - [`netlist`]: SPICE netlists with `.model`, `.subckt` (nested, with parameters), `.param`
//!   expressions, `.include` and `.lib`, `.options`, `.ic` and `.nodeset`.
//! - [`devices`]: R, C, L and mutual inductance `K`; independent sources with `PULSE`, `SIN`,
//!   `EXP` and `PWL`; `E`, `F`, `G`, `H` (linear and `POLY`); diodes; Gummel–Poon bipolar
//!   transistors; level-1 MOSFETs; voltage- and current-controlled switches; ideal and
//!   macromodel op-amps.
//! - [`analysis`]: modified nodal analysis on a sparse LU; operating points by damped Newton
//!   with gmin and source stepping; DC sweeps; AC small-signal sweeps; transient with
//!   trapezoidal or Gear-2 integration and truncation-error step control.
//! - [`measure`]: `.meas`-style measurements on waveforms and responses.
//! - [`montecarlo`]: tolerance runs on every core with deterministic per-run seeds.
//! - [`meter`]: the measured operations of every run, in simulation-usage categories.

pub mod analysis;
pub mod circuit;
pub mod devices;
pub mod measure;
pub mod meter;
pub mod montecarlo;
pub mod netlist;
pub mod num;
pub mod results;
pub mod sparse;

pub use analysis::ac::AcSweep;
pub use analysis::tran::{TranSpec, TranStats};
pub use circuit::Circuit;
pub use devices::Method;
pub use meter::{Usage, MEASUREMENT_VERSION};
pub use results::{AcWave, RealWave, SimResult};

use analysis::engine::{Engine, LoadParams};
use netlist::{parse_number, Flat, Parsed, Tok};
use std::collections::BTreeMap;

/// Simulator options (`.options`), with SPICE's defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub reltol: f64,
    pub abstol: f64,
    pub vntol: f64,
    pub chgtol: f64,
    pub trtol: f64,
    pub gmin: f64,
    /// Newton iterations: operating point, DC sweep point, transient step.
    pub itl1: usize,
    pub itl2: usize,
    pub itl4: usize,
    pub method: Method,
    pub temp_c: f64,
    /// A resistor from every node to ground (`rshunt`), so nodes left without a DC path solve.
    pub rshunt: Option<f64>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            reltol: 1e-3,
            abstol: 1e-12,
            vntol: 1e-6,
            chgtol: 1e-14,
            trtol: 7.0,
            gmin: 1e-12,
            itl1: 100,
            itl2: 50,
            itl4: 20,
            method: Method::Trapezoidal,
            temp_c: 27.0,
            rshunt: None,
        }
    }
}

impl Options {
    pub fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        let num = || parse_number(value).ok_or_else(|| format!("option `{key}` needs a number"));
        match key {
            "reltol" => self.reltol = num()?,
            "abstol" => self.abstol = num()?,
            "vntol" => self.vntol = num()?,
            "chgtol" => self.chgtol = num()?,
            "trtol" => self.trtol = num()?,
            "gmin" => self.gmin = num()?,
            "itl1" => self.itl1 = num()? as usize,
            "itl2" => self.itl2 = num()? as usize,
            "itl4" => self.itl4 = num()? as usize,
            "temp" => self.temp_c = num()?,
            "rshunt" => self.rshunt = Some(num()?).filter(|r| *r > 0.0),
            "method" => {
                self.method = match value {
                    "trap" | "trapezoidal" => Method::Trapezoidal,
                    "gear" | "gear2" => Method::Gear2,
                    "euler" | "be" => Method::Euler,
                    v => return Err(format!("unknown integration method `{v}` (trap, gear)")),
                }
            }
            // Accepted for compatibility; no effect on this engine.
            "tnom" | "itl3" | "itl5" | "itl6" | "maxord" | "noacct" | "nopage" | "nomod" | "acct" | "list" | "node" | "opts"
            | "gminsteps" | "srcsteps" | "numdgt" | "pivtol" | "pivrel" | "cshunt" | "post" | "probe" | "ingold" => {}
            k => return Err(format!("unknown option `{k}`")),
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimError {
    Parse(String),
    Build(String),
    BadRequest(String),
    Singular { unknown: String },
    NoConvergence { analysis: String, detail: String },
    TimestepTooSmall { time: f64 },
    CountOverflow,
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimError::Parse(m) => write!(f, "netlist: {m}"),
            SimError::Build(m) => write!(f, "circuit: {m}"),
            SimError::BadRequest(m) => write!(f, "{m}"),
            SimError::Singular { unknown } => write!(
                f,
                "the circuit matrix is singular at `{unknown}`: a node has no DC path to ground, or voltage sources or inductors form a loop"
            ),
            SimError::NoConvergence { analysis, detail } => write!(f, "{analysis} did not converge: {detail}"),
            SimError::TimestepTooSmall { time } => write!(f, "transient time step too small at t = {time:e} s"),
            SimError::CountOverflow => write!(f, "operation count exceeds u64"),
        }
    }
}

impl std::error::Error for SimError {}

/// One analysis.
#[derive(Clone, Debug, PartialEq)]
pub enum Analysis {
    Op,
    Dc { source: String, start: f64, stop: f64, step: f64 },
    Ac { sweep: AcSweep, points: usize, fstart: f64, fstop: f64 },
    Tran(TranSpec),
}

impl Analysis {
    pub fn kind(&self) -> &'static str {
        match self {
            Analysis::Op => "op",
            Analysis::Dc { .. } => "dc",
            Analysis::Ac { .. } => "ac",
            Analysis::Tran(_) => "tran",
        }
    }
}

/// A finished analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub result: SimResult,
    pub usage: Usage,
    pub newton_iterations: u64,
    pub tran: Option<TranStats>,
    pub warnings: Vec<String>,
}

/// A parsed and flattened netlist.
#[derive(Clone, Debug)]
pub struct Design {
    pub parsed: Parsed,
    pub flat: Flat,
    pub options: Options,
    pub ics: Vec<(String, f64)>,
    pub nodesets: Vec<(String, f64)>,
}

impl Design {
    pub fn from_netlist(text: &str, resolve: &dyn Fn(&str) -> Option<String>) -> Result<Design, SimError> {
        let parsed = netlist::parse(text, resolve).map_err(|e| SimError::Parse(e.to_string()))?;
        let flat = netlist::flatten(&parsed).map_err(|e| SimError::Parse(format!("line {}: {}", e.line, e.message)))?;
        let mut options = Options::default();
        for (k, v) in &parsed.options {
            options.set(k, v).map_err(SimError::Parse)?;
        }
        if let Some(t) = &parsed.temp {
            options.temp_c = parse_number(t).ok_or_else(|| SimError::Parse(".temp needs a number".into()))?;
        }
        let eval = |list: &[(String, String)]| -> Result<Vec<(String, f64)>, SimError> {
            list.iter()
                .map(|(n, v)| netlist::expr::eval(v, &flat.params).map(|x| (n.clone(), x)).map_err(SimError::Parse))
                .collect()
        };
        let ics = eval(&parsed.ics)?;
        let nodesets = eval(&parsed.nodesets)?;
        Ok(Design { parsed, flat, options, ics, nodesets })
    }

    pub fn circuit(&self) -> Result<Circuit, SimError> {
        Circuit::build(&self.flat, self.options.clone(), &self.ics, &self.nodesets).map_err(|e| SimError::Build(e.to_string()))
    }

    /// The analyses the netlist's own control lines ask for.
    pub fn analyses(&self) -> Result<Vec<Analysis>, SimError> {
        self.parsed.controls.iter().filter(|c| c.name != ".meas" && c.name != ".measure").map(control_analysis).collect()
    }
}

fn control_analysis(c: &netlist::Card) -> Result<Analysis, SimError> {
    let w: Vec<&str> = c.toks.iter().filter_map(Tok::word).collect();
    let n = |i: usize| w.get(i).and_then(|s| parse_number(s)).ok_or_else(|| SimError::Parse(format!("line {}: {} needs a number in position {}", c.line, c.name, i + 1)));
    Ok(match c.name.as_str() {
        ".op" => Analysis::Op,
        ".dc" => Analysis::Dc { source: w.first().copied().unwrap_or_default().into(), start: n(1)?, stop: n(2)?, step: n(3)? },
        ".ac" => {
            let sweep = match w.first().copied() {
                Some("dec") => AcSweep::Dec,
                Some("oct") => AcSweep::Oct,
                Some("lin") => AcSweep::Lin,
                _ => return Err(SimError::Parse(format!("line {}: .ac needs dec, oct or lin", c.line))),
            };
            Analysis::Ac { sweep, points: n(1)? as usize, fstart: n(2)?, fstop: n(3)? }
        }
        ".tran" => {
            let uic = w.contains(&"uic");
            let nums: Vec<f64> = w.iter().filter_map(|s| parse_number(s)).collect();
            if nums.len() < 2 {
                return Err(SimError::Parse(format!("line {}: .tran needs tstep and tstop", c.line)));
            }
            Analysis::Tran(TranSpec { tstep: nums[0], tstop: nums[1], tstart: nums.get(2).copied().unwrap_or(0.0), tmax: nums.get(3).copied(), uic })
        }
        other => return Err(SimError::Parse(format!("unsupported analysis {other}"))),
    })
}

/// Runs one analysis on a design.
pub fn simulate(design: &Design, analysis: &Analysis) -> Result<Outcome, SimError> {
    let c = design.circuit()?;
    simulate_circuit(&c, analysis)
}

/// Runs one analysis on a built circuit.
pub fn simulate_circuit(c: &Circuit, analysis: &Analysis) -> Result<Outcome, SimError> {
    let _ = num::dual::take_ops();
    let mut e = Engine::new(c);
    let mut tran = None;
    let result = match analysis {
        Analysis::Op => {
            let mut x = vec![0.0; c.size()];
            analysis::op::solve(&mut e, &mut x, LoadParams::dc())?;
            e.assemble(&x, &LoadParams::dc());
            let probes = analysis::Probes::new(c);
            let vals = probes.values(c, &x, &e.states);
            SimResult::Op(probes.names.iter().cloned().zip(vals).collect::<BTreeMap<_, _>>())
        }
        Analysis::Dc { source, start, stop, step } => {
            if *step == 0.0 || (stop - start) / step < 0.0 {
                return Err(SimError::BadRequest("DC sweep step must move from start toward stop".into()));
            }
            let n = ((stop - start) / step + 1e-9).floor() as usize;
            let vals: Vec<f64> = (0..=n).map(|k| start + step * k as f64).collect();
            SimResult::Dc(analysis::dc::sweep(&mut e, source, &vals)?)
        }
        Analysis::Ac { sweep, points, fstart, fstop } => {
            let f = analysis::ac::frequencies(*sweep, *points, *fstart, *fstop)?;
            SimResult::Ac(analysis::ac::sweep(&mut e, &f)?)
        }
        Analysis::Tran(spec) => {
            let (w, s) = analysis::tran::run(&mut e, spec)?;
            tran = Some(s);
            SimResult::Tran(w)
        }
    };
    Ok(Outcome {
        result,
        usage: e.meter.usage().map_err(|_| SimError::CountOverflow)?,
        newton_iterations: e.newton_iterations,
        tran,
        warnings: c.warnings.clone(),
    })
}
