//! The native engine: [`runmat_circuit_sim`] behind the [`SpiceEngine`] seam, so every deck the
//! circuit IR emits runs without ngspice.

use crate::{AcResult, Analysis, Deck, OpPoint, SimResult, SpiceEngine, SpiceError, Sweep};
use runmat_circuit_sim as sim;
use std::collections::BTreeMap;

/// Reads library files a deck includes.
pub type LibraryResolver = Box<dyn Fn(&str) -> Option<String> + Send + Sync>;

pub struct Native {
    pub resolve: LibraryResolver,
    /// Measured operations of the last run.
    pub last_usage: Option<sim::Usage>,
}

impl Default for Native {
    fn default() -> Self {
        Native { resolve: Box::new(|_| None), last_usage: None }
    }
}

pub fn to_sim_analysis(a: &Analysis) -> sim::Analysis {
    match a {
        Analysis::Op => sim::Analysis::Op,
        Analysis::Dc { source, start, stop, step } => sim::Analysis::Dc { source: source.clone(), start: *start, stop: *stop, step: *step },
        Analysis::Ac { sweep, points, fstart, fstop } => sim::Analysis::Ac {
            sweep: match sweep {
                crate::AcSweep::Dec => sim::AcSweep::Dec,
                crate::AcSweep::Oct => sim::AcSweep::Oct,
                crate::AcSweep::Lin => sim::AcSweep::Lin,
            },
            points: *points as usize,
            fstart: *fstart,
            fstop: *fstop,
        },
        Analysis::Tran { step, stop, start, max_step } => sim::Analysis::Tran(sim::TranSpec { tstep: *step, tstop: *stop, tstart: *start, tmax: *max_step, uic: false }),
    }
}

fn err(e: sim::SimError) -> SpiceError {
    SpiceError::Simulation { message: e.to_string(), log: vec![] }
}

impl SpiceEngine for Native {
    fn run(&mut self, deck: &Deck, analysis: &Analysis) -> Result<SimResult, SpiceError> {
        let design = sim::Design::from_netlist(&deck.text(), &*self.resolve).map_err(err)?;
        let out = sim::simulate(&design, &to_sim_analysis(analysis)).map_err(err)?;
        self.last_usage = Some(out.usage);
        // Results by net name and by deck source name, as the seam reports them.
        let nets: Vec<(String, String)> = deck.node_of.iter().filter(|(_, n)| *n != "0").map(|(net, node)| (net.clone(), format!("v({})", node.to_ascii_lowercase()))).collect();
        let sources: Vec<(String, String)> = deck.elements.iter().filter(|e| e.name.starts_with('V')).map(|e| (e.name.clone(), format!("i({})", e.name.to_ascii_lowercase()))).collect();
        Ok(match out.result {
            sim::SimResult::Op(m) => {
                let pick = |list: &[(String, String)]| list.iter().filter_map(|(k, s)| m.get(s).map(|v| (k.clone(), *v))).collect::<BTreeMap<_, _>>();
                SimResult::Op(OpPoint { voltages: pick(&nets), currents: pick(&sources) })
            }
            sim::SimResult::Dc(w) | sim::SimResult::Tran(w) => {
                let pick = |list: &[(String, String)]| list.iter().filter_map(|(k, s)| w.get(s).map(|v| (k.clone(), v.to_vec()))).collect::<BTreeMap<_, _>>();
                let s = Sweep { x: w.x.clone(), voltages: pick(&nets), currents: pick(&sources) };
                if matches!(analysis, Analysis::Tran { .. }) {
                    SimResult::Tran(s)
                } else {
                    SimResult::Dc(s)
                }
            }
            sim::SimResult::Ac(w) => {
                let pick = |list: &[(String, String)]| {
                    list.iter().filter_map(|(k, s)| w.get(s).map(|v| (k.clone(), v.iter().map(|c| (c.re, c.im)).collect()))).collect::<BTreeMap<_, _>>()
                };
                SimResult::Ac(AcResult { freq: w.freq.clone(), voltages: pick(&nets), currents: pick(&sources) })
            }
        })
    }
    fn describe(&self) -> String {
        format!("runmat-circuit-sim {}", env!("CARGO_PKG_VERSION"))
    }
}
