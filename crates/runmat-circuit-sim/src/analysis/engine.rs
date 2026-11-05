//! Assembly, linear solves and Newton–Raphson iteration over one circuit.

use crate::circuit::Circuit;
use crate::devices::{Load, Mode, States, Target};
use crate::meter::Meter;
use crate::num::dual;
use crate::sparse::lu::{Lu, LuError};
use crate::sparse::{order, Matrix, Pattern};
use crate::SimError;

/// Conductance that holds `.ic` nodes at their values during the initial operating point.
const IC_CONDUCTANCE: f64 = 1e4;

#[derive(Clone, Copy, Debug)]
pub struct LoadParams {
    pub mode: Mode,
    pub time: f64,
    pub src_scale: f64,
    /// Conductance from every node to ground (gmin stepping).
    pub gdiag: f64,
    pub init_junctions: bool,
    /// Hold `.ic` (and `.nodeset`) nodes.
    pub force_ic: bool,
    pub force_nodesets: bool,
}

impl LoadParams {
    pub fn dc() -> Self {
        LoadParams { mode: Mode::Dc, time: 0.0, src_scale: 1.0, gdiag: 0.0, init_junctions: false, force_ic: false, force_nodesets: false }
    }
}

pub struct Engine<'c> {
    pub c: &'c Circuit,
    pub mat: Matrix<f64>,
    order: Vec<usize>,
    lu: Option<Lu<f64>>,
    pub meter: Meter,
    pub mem: Vec<f64>,
    pub states: States,
    pub vt: f64,
    pub newton_iterations: u64,
}

impl<'c> Engine<'c> {
    pub fn new(c: &'c Circuit) -> Self {
        let pattern = Pattern::build(c.size(), []);
        let order = order::minimum_degree(&pattern);
        Engine {
            c,
            mat: Matrix::new(pattern),
            order,
            lu: None,
            meter: Meter::default(),
            mem: c.mem_init.clone(),
            states: States::new(c.state_is_flux.clone()),
            vt: crate::devices::K_OVER_Q * (c.options.temp_c + 273.15),
            newton_iterations: 0,
        }
    }

    /// Loads every device at `x`; returns the right-hand side and whether a device limited.
    pub fn assemble(&mut self, x: &[f64], p: &LoadParams) -> (Vec<f64>, bool) {
        let n = self.c.size();
        loop {
            self.mat.clear();
            self.mat.ops = 0;
            let mut rhs = vec![0.0; n];
            let mut ld = Load {
                mode: p.mode,
                target: Target::Real { mat: &mut self.mat, rhs: &mut rhs },
                x,
                mem: &mut self.mem,
                states: &mut self.states,
                time: p.time,
                src_scale: p.src_scale,
                gmin: self.c.options.gmin,
                noncon: false,
                init_junctions: p.init_junctions,
                vt: self.vt,
                temp_k: self.c.options.temp_c + 273.15,
                ops: 0,
            };
            for d in &self.c.devices {
                d.load(&mut ld);
            }
            if p.gdiag > 0.0 {
                for k in 0..self.c.n_nodes {
                    ld.mat(k, k, p.gdiag);
                }
            }
            let force = |list: &[(usize, f64)], ld: &mut Load| {
                for &(k, v) in list {
                    ld.mat(k, k, IC_CONDUCTANCE);
                    ld.rhs(k, IC_CONDUCTANCE * v);
                }
            };
            if p.force_ic {
                force(&self.c.ics, &mut ld);
            }
            if p.force_nodesets {
                force(&self.c.nodesets, &mut ld);
            }
            let (noncon, ops) = (ld.noncon, ld.ops);
            self.meter.assembly += dual::take_ops() + ops + self.mat.ops;
            if self.mat.needs_rebuild() {
                self.mat.rebuild();
                self.order = order::minimum_degree(&self.mat.pattern);
                self.lu = None;
                continue;
            }
            return (rhs, noncon);
        }
    }

    /// Solves the assembled matrix for `rhs` (refactoring with the previous pivots when they
    /// remain acceptable).
    pub fn solve(&mut self, rhs: &mut [f64]) -> Result<(), SimError> {
        let fresh = match &mut self.lu {
            Some(lu) => match lu.refactor(&self.mat.pattern, &self.mat.vals) {
                Some(Ok(())) => {
                    self.meter.factorization += lu.ops;
                    false
                }
                Some(Err(e)) => return Err(self.lu_error(e)),
                None => true,
            },
            None => true,
        };
        if fresh {
            let lu = Lu::factor(&self.mat.pattern, &self.mat.vals, &self.order).map_err(|e| self.lu_error(e))?;
            self.meter.factorization += lu.ops;
            self.lu = Some(lu);
        }
        self.meter.solve += self.lu.as_ref().expect("factored").solve(rhs);
        Ok(())
    }

    fn lu_error(&self, e: LuError) -> SimError {
        match e {
            LuError::Singular { column } => SimError::Singular { unknown: self.c.names[column].clone() },
            LuError::NonFinite => SimError::NoConvergence { analysis: "matrix".into(), detail: "a matrix entry is not finite".into() },
        }
    }

    /// Newton–Raphson from `x` until the update meets the tolerances and no device limited.
    /// Returns the iterations used, or `None` when `max_iter` passed without convergence.
    pub fn newton(&mut self, x: &mut Vec<f64>, p: &LoadParams, max_iter: usize) -> Result<Option<usize>, SimError> {
        let o = &self.c.options;
        let (reltol, vntol, abstol) = (o.reltol, o.vntol, o.abstol);
        let n_nodes = self.c.n_nodes;
        for it in 0..max_iter {
            let mut lp = *p;
            lp.init_junctions = p.init_junctions && it == 0;
            let (mut rhs, noncon) = self.assemble(x, &lp);
            match self.solve(&mut rhs) {
                Ok(()) => {}
                Err(SimError::Singular { .. }) if it > 0 => return Ok(None),
                Err(e) => return Err(e),
            }
            self.newton_iterations += 1;
            if rhs.iter().any(|v| !v.is_finite()) {
                return Ok(None);
            }
            let mut converged = !noncon && it > 0 && !lp.init_junctions;
            if converged {
                for (k, (new, old)) in rhs.iter().zip(x.iter()).enumerate() {
                    let tol = reltol * new.abs().max(old.abs()) + if k < n_nodes { vntol } else { abstol };
                    if (new - old).abs() > tol {
                        converged = false;
                        break;
                    }
                }
                self.meter.other += 5 * rhs.len() as u64;
            }
            *x = rhs;
            if converged {
                return Ok(Some(it + 1));
            }
        }
        Ok(None)
    }

    /// Lets devices record an accepted solution (switch states).
    pub fn accept(&mut self, x: &[f64]) {
        for d in &self.c.devices {
            d.accept(x, &mut self.mem);
        }
    }
}
