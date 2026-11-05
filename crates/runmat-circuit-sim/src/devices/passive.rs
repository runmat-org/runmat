//! Resistors, capacitors and inductors (with mutual coupling).

use super::{node_v, Device, Load, States};
use crate::num::Dual;

pub struct Resistor {
    pub name: String,
    pub a: usize,
    pub b: usize,
    pub ohms: f64,
}

impl Device for Resistor {
    fn name(&self) -> &str {
        &self.name
    }
    fn load(&self, ld: &mut Load) {
        ld.conductance(self.a, self.b, 1.0 / self.ohms);
        ld.ops += 1;
    }
    fn probe_current(&self, x: &[f64], _s: &States) -> Option<f64> {
        Some((node_v(x, self.a) - node_v(x, self.b)) / self.ohms)
    }
    fn noise(&self, sink: &mut crate::analysis::noise::NoiseSink) {
        sink.thermal(self.a, self.b, 1.0 / self.ohms);
    }
}

pub struct Capacitor {
    pub name: String,
    pub a: usize,
    pub b: usize,
    pub farads: f64,
    pub state: usize,
    pub ic: Option<f64>,
}

impl Device for Capacitor {
    fn name(&self) -> &str {
        &self.name
    }
    fn load(&self, ld: &mut Load) {
        let v = Dual::<1>::var(ld.v(self.a) - ld.v(self.b), 0);
        ld.charge(self.state, self.a, self.b, v * self.farads, [(self.a, self.b)]);
    }
    fn probe_current(&self, _x: &[f64], s: &States) -> Option<f64> {
        Some(s.i[self.state][0])
    }
    fn apply_ic(&self, states: &mut States) {
        if let Some(v) = self.ic {
            states.q[self.state] = [self.farads * v; 4];
        }
    }
}

pub struct Inductor {
    pub name: String,
    pub a: usize,
    pub b: usize,
    pub henries: f64,
    pub branch: usize,
    pub state: usize,
    pub ic: Option<f64>,
    /// Mutual inductances to other inductors' branches: `(branch, M)`.
    pub couplings: Vec<(usize, f64)>,
}

impl Device for Inductor {
    fn name(&self) -> &str {
        &self.name
    }
    fn load(&self, ld: &mut Load) {
        let (a, b, j) = (self.a, self.b, self.branch);
        ld.mat(a, j, 1.0);
        ld.mat(b, j, -1.0);
        ld.mat(j, a, 1.0);
        ld.mat(j, b, -1.0);
        let mut terms = vec![(j, self.henries)];
        terms.extend(self.couplings.iter().copied());
        ld.flux(self.state, j, &terms);
    }
    fn branch(&self) -> Option<usize> {
        Some(self.branch)
    }
    fn apply_ic(&self, states: &mut States) {
        if let Some(i) = self.ic {
            states.q[self.state] = [self.henries * i; 4];
        }
    }
}
