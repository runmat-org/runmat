//! Switches: `S` (voltage-controlled, model `SW`: `vt vh ron roff`) and `W` (current-controlled
//! through a voltage source's branch, model `CSW`: `it ih ron roff`).
//!
//! The switch is on above `vt + vh`, off below `vt − vh`, and keeps its last accepted state in
//! between (hysteresis). A state change during Newton iterations marks the iteration
//! unconverged, so the solution is consistent with the final state.

use super::{Device, Load};

pub struct Switch {
    pub name: String,
    pub a: usize,
    pub b: usize,
    /// The control: a node pair, or a branch unknown paired with ground.
    pub ctrl: (usize, usize),
    pub threshold: f64,
    pub hysteresis: f64,
    pub g_on: f64,
    pub g_off: f64,
    /// `mem[k]`: accepted state, `mem[k+1]`: present state (1 on, 0 off).
    pub mem: usize,
}

impl Device for Switch {
    fn name(&self) -> &str {
        &self.name
    }
    fn load(&self, ld: &mut Load) {
        let c = ld.v(self.ctrl.0) - ld.v(self.ctrl.1);
        let accepted = ld.mem[self.mem];
        let state = if ld.is_ac() {
            ld.mem[self.mem + 1]
        } else if c > self.threshold + self.hysteresis {
            1.0
        } else if c < self.threshold - self.hysteresis {
            0.0
        } else {
            accepted
        };
        if !ld.is_ac() {
            if state != ld.mem[self.mem + 1] {
                ld.noncon = true;
            }
            ld.mem[self.mem + 1] = state;
        }
        ld.conductance(
            self.a,
            self.b,
            if state > 0.5 { self.g_on } else { self.g_off },
        );
    }
    fn accept(&self, _x: &[f64], mem: &mut [f64]) {
        mem[self.mem] = mem[self.mem + 1];
    }
}
