//! `runmat-circuit-spice`: circuit simulation for RunMat PCB.
//!
//! - [`netlist`]: a SPICE deck from an elaborated design ([`runmat_circuit::elab::Netlist`])
//!   and its pool: each component becomes its part's subcircuit (`X…`, pins in the model's
//!   `pin_order`) or a primitive (`R`, `C`, `L`, `D`, …) from its parameters; stimuli are
//!   sources between nets; every component pin can be fed through a zero-volt sense source so
//!   pin currents, and therefore device power, are exact for any model.
//! - [`Analysis`] and the result types: operating point, DC sweep, AC (complex) and
//!   transient.
//! - [`SpiceEngine`]: the engine seam. [`ngspice::Ngspice`] drives the ngspice shared
//!   library (BSD), loaded at run time; when it is absent the error says where it looked and
//!   how to install or point at it. [`fake::FakeEngine`] is a small exact solver for
//!   resistive circuits with DC sources, for tests.
//! - [`power_map`]: per-designator dissipation from an operating point, the input of
//!   [`runmat_circuit::fea::pcb_source`].
//! - [`UsageSeam`]: where an adapter reports simulation work to the operation meter
//!   (`runmat-simulation-usage`). This crate meters nothing itself.

pub mod fake;
pub mod netlist;
pub mod ngspice;

use std::collections::BTreeMap;
use std::fmt;

pub use netlist::{emit, Deck, EmitError, EmitOptions, Stimulus, Waveform};

/// One analysis.
#[derive(Clone, Debug, PartialEq)]
pub enum Analysis {
    /// DC operating point.
    Op,
    /// DC sweep of a source (by its deck name) from `start` to `stop` in `step`.
    Dc {
        source: String,
        start: f64,
        stop: f64,
        step: f64,
    },
    /// Small-signal AC from `fstart` to `fstop` (Hz).
    Ac {
        sweep: AcSweep,
        points: u32,
        fstart: f64,
        fstop: f64,
    },
    /// Transient to `stop` (s) with print step `step`.
    Tran {
        step: f64,
        stop: f64,
        start: f64,
        max_step: Option<f64>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcSweep {
    Dec,
    Oct,
    Lin,
}

impl Analysis {
    /// The SPICE control line.
    pub fn card(&self) -> String {
        match self {
            Analysis::Op => "op".into(),
            Analysis::Dc {
                source,
                start,
                stop,
                step,
            } => format!("dc {source} {start:e} {stop:e} {step:e}"),
            Analysis::Ac {
                sweep,
                points,
                fstart,
                fstop,
            } => format!(
                "ac {} {points} {fstart:e} {fstop:e}",
                match sweep {
                    AcSweep::Dec => "dec",
                    AcSweep::Oct => "oct",
                    AcSweep::Lin => "lin",
                }
            ),
            Analysis::Tran {
                step,
                stop,
                start,
                max_step,
            } => match max_step {
                Some(m) => format!("tran {step:e} {stop:e} {start:e} {m:e}"),
                None => format!("tran {step:e} {stop:e} {start:e}"),
            },
        }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Analysis::Op => "op",
            Analysis::Dc { .. } => "dc",
            Analysis::Ac { .. } => "ac",
            Analysis::Tran { .. } => "tran",
        }
    }
}

/// Operating point: node voltages (V, by net name) and source branch currents (A, by deck
/// source name, flowing into the source's positive terminal).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpPoint {
    pub voltages: BTreeMap<String, f64>,
    pub currents: BTreeMap<String, f64>,
}

/// A swept real result: the sweep variable and every signal over it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sweep {
    pub x: Vec<f64>,
    pub voltages: BTreeMap<String, Vec<f64>>,
    pub currents: BTreeMap<String, Vec<f64>>,
}

/// AC result: frequency and complex signals `(re, im)`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AcResult {
    pub freq: Vec<f64>,
    pub voltages: BTreeMap<String, Vec<(f64, f64)>>,
    pub currents: BTreeMap<String, Vec<(f64, f64)>>,
}

impl AcResult {
    /// Magnitude in dB of a node voltage.
    pub fn db(&self, net: &str) -> Option<Vec<f64>> {
        self.voltages
            .get(net)
            .map(|v| v.iter().map(|(r, i)| 20.0 * r.hypot(*i).log10()).collect())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimResult {
    Op(OpPoint),
    Dc(Sweep),
    Ac(AcResult),
    /// Transient: `x` is time (s).
    Tran(Sweep),
}

impl SimResult {
    pub fn op(&self) -> Option<&OpPoint> {
        match self {
            SimResult::Op(o) => Some(o),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpiceError {
    /// The engine is not installed or could not be loaded.
    EngineUnavailable { tried: Vec<String>, hint: String },
    /// The engine rejected the deck or failed during the analysis (with its output).
    Simulation { message: String, log: Vec<String> },
    /// The engine cannot run this analysis or device.
    Unsupported(String),
}

impl fmt::Display for SpiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpiceError::EngineUnavailable { tried, hint } => {
                write!(
                    f,
                    "ngspice is not available (tried {}). {hint}",
                    tried.join(", ")
                )
            }
            SpiceError::Simulation { message, log } => {
                write!(f, "simulation failed: {message}")?;
                for l in log.iter().rev().take(5).rev() {
                    write!(f, "\n  {l}")?;
                }
                Ok(())
            }
            SpiceError::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}

impl std::error::Error for SpiceError {}

/// A SPICE engine.
pub trait SpiceEngine {
    /// Runs one analysis on the deck.
    fn run(&mut self, deck: &Deck, analysis: &Analysis) -> Result<SimResult, SpiceError>;
    /// Engine name and version, for reports.
    fn describe(&self) -> String;
}

/// What a finished analysis did, for usage reporting.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisReport {
    pub engine: String,
    pub analysis: &'static str,
    /// Circuit size: nodes and elements in the deck.
    pub nodes: usize,
    pub elements: usize,
    /// Result points (1 for an operating point).
    pub points: usize,
}

/// The usage seam. An adapter implements it to turn finished analyses into operation-meter
/// records (`runmat-simulation-usage`); the simulation crate itself keeps no meter.
pub trait UsageSeam {
    fn analysis_finished(&mut self, report: &AnalysisReport);
}

/// Runs an analysis and reports it to the usage seam.
pub fn simulate(
    engine: &mut dyn SpiceEngine,
    deck: &Deck,
    analysis: &Analysis,
    usage: Option<&mut dyn UsageSeam>,
) -> Result<SimResult, SpiceError> {
    let r = engine.run(deck, analysis)?;
    if let Some(u) = usage {
        let points = match &r {
            SimResult::Op(_) => 1,
            SimResult::Dc(s) | SimResult::Tran(s) => s.x.len(),
            SimResult::Ac(a) => a.freq.len(),
        };
        u.analysis_finished(&AnalysisReport {
            engine: engine.describe(),
            analysis: analysis.kind(),
            nodes: deck.nodes().len(),
            elements: deck.elements.len(),
            points,
        });
    }
    Ok(r)
}

/// Dissipation per designator (W) at an operating point: Σ V(pin) · I(into pin) over each
/// component's pins, from the deck's sense sources. Exact for any model (KCL makes it
/// independent of the reference node). Components without sense sources are absent.
pub fn power_map(deck: &Deck, op: &OpPoint) -> BTreeMap<String, f64> {
    let mut out: BTreeMap<String, f64> = BTreeMap::new();
    for s in &deck.senses {
        let v = op.voltages.get(&s.net).copied().unwrap_or(0.0);
        let i = op.currents.get(&s.source).copied().unwrap_or(0.0);
        *out.entry(s.designator.clone()).or_default() += v * i;
    }
    out
}
