//! Analyses: operating point, DC sweep, AC small-signal and transient.

pub mod ac;
pub mod dc;
pub mod engine;
pub mod noise;
pub mod op;
pub mod tran;

use crate::circuit::Circuit;
use crate::devices::States;
use crate::sparse::GND;

/// The signals every analysis records: node voltages `v(<node>)`, branch currents
/// `i(<device>)`, and the currents of devices without a branch unknown (`i(r1)`, the collector
/// current `i(q1)`, the drain current `i(m1)`, a capacitor's `i(c1)`).
pub struct Probes {
    pub names: Vec<String>,
    unknowns: Vec<usize>,
    devices: Vec<usize>,
}

impl Probes {
    pub fn new(c: &Circuit) -> Probes {
        let mut names = vec![];
        let mut unknowns = vec![];
        for (k, n) in c.names.iter().enumerate() {
            if k < c.n_nodes {
                names.push(format!("v({n})"));
                unknowns.push(k);
            } else if n.starts_with("i(") {
                names.push(n.clone());
                unknowns.push(k);
            }
        }
        let mut devices = vec![];
        let probe_x = vec![0.0; c.size()];
        let probe_s = States::new(c.state_is_flux.clone());
        for (k, d) in c.devices.iter().enumerate() {
            // Asking a model whether it reports a current evaluates it at zero; that is not
            // simulation work and is not counted.
            if d.branch().is_none()
                && crate::num::dual::uncounted(|| d.probe_current(&probe_x, &probe_s)).is_some()
            {
                names.push(format!("i({})", d.name()));
                devices.push(k);
            }
        }
        Probes {
            names,
            unknowns,
            devices,
        }
    }
    pub fn unknown_count(&self) -> usize {
        self.unknowns.len()
    }
    pub fn unknown(&self, k: usize) -> usize {
        self.unknowns[k]
    }
    /// Real values at a solution.
    pub fn values(&self, c: &Circuit, x: &[f64], s: &States) -> Vec<f64> {
        let mut v: Vec<f64> = self
            .unknowns
            .iter()
            .map(|&k| if k == GND { 0.0 } else { x[k] })
            .collect();
        for &d in &self.devices {
            v.push(c.devices[d].probe_current(x, s).unwrap_or(0.0));
        }
        v
    }
}
