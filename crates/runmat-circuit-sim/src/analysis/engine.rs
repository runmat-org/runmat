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
    /// The operations this run may spend (the host's granted operations); `None`: no limit.
    pub limit: Option<u64>,
    /// Set when the run stopped before the limit; its result holds the points solved so far.
    pub stopped: bool,
    /// The most one assembly has cost so far: the guard stops before an assembly that could not
    /// fit (an assembly that would pass the limit anyway is discarded, see [`Engine::assemble`]).
    max_assembly: u64,
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
            limit: None,
            stopped: false,
            max_assembly: 0,
        }
    }

    /// The operations counted so far, with the device-model operations not yet absorbed.
    pub fn spent(&self) -> u64 {
        let m = &self.meter;
        m.assembly.saturating_add(m.factorization).saturating_add(m.solve).saturating_add(m.other).saturating_add(dual::peek_ops())
    }

    /// The operations left under the limit (`u64::MAX` without one).
    pub fn left(&self) -> u64 {
        self.limit.map_or(u64::MAX, |l| l.saturating_sub(self.spent()))
    }

    /// Marks the run stopped at its limit.
    pub fn stop(&mut self) -> SimError {
        self.stopped = true;
        SimError::Stopped
    }

    /// Whether `cost` more operations fit under the limit; when they don't, marks the run
    /// stopped.
    pub fn room(&mut self, cost: u64) -> bool {
        if cost > self.left() {
            self.stopped = true;
            return false;
        }
        true
    }

    /// Spends `ops` bookkeeping operations (convergence tests, predictors, error estimates) if
    /// they fit; their count is fixed by the vector lengths, so the check comes first.
    pub fn charge_other(&mut self, ops: u64) -> Result<(), SimError> {
        if !self.room(ops) {
            return Err(SimError::Stopped);
        }
        self.meter.other += ops;
        Ok(())
    }

    /// Before an assembly: stops when the costliest assembly so far would not fit.
    pub fn room_to_assemble(&mut self) -> Result<(), SimError> {
        if self.room(self.max_assembly) {
            Ok(())
        } else {
            Err(SimError::Stopped)
        }
    }

    /// Commits one assembly's operations. A device's cost is known only once it has run, so an
    /// assembly that would pass the limit is discarded instead: neither its result nor its
    /// count is kept, and the run stops. What the run reports never passes its limit.
    pub fn commit_assembly(&mut self, ops: u64) -> Result<(), SimError> {
        self.max_assembly = self.max_assembly.max(ops);
        if ops > self.left() {
            return Err(self.stop());
        }
        self.meter.assembly += ops;
        Ok(())
    }

    /// The probe values at a solution. Device currents evaluate the models, so their operations
    /// count as assembly; past the limit the values are discarded uncounted and the run stops.
    pub fn probe_values(&mut self, probes: &super::Probes, x: &[f64]) -> Result<Vec<f64>, SimError> {
        let v = probes.values(self.c, x, &self.states);
        let ops = dual::take_ops();
        if ops > self.left() {
            return Err(self.stop());
        }
        self.meter.assembly += ops;
        Ok(v)
    }

    /// Loads every device at `x`; returns the right-hand side and whether a device limited, or
    /// [`SimError::Stopped`] at the operation limit.
    pub fn assemble(&mut self, x: &[f64], p: &LoadParams) -> Result<(Vec<f64>, bool), SimError> {
        let n = self.c.size();
        loop {
            self.room_to_assemble()?;
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
            let gdiag = p.gdiag + self.c.options.rshunt.map(|r| 1.0 / r).unwrap_or(0.0);
            if gdiag > 0.0 {
                for k in 0..self.c.n_nodes {
                    ld.mat(k, k, gdiag);
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
            self.commit_assembly(dual::take_ops() + ops + self.mat.ops)?;
            if self.mat.needs_rebuild() {
                self.mat.rebuild();
                self.order = order::minimum_degree(&self.mat.pattern);
                self.lu = None;
                continue;
            }
            return Ok((rhs, noncon));
        }
    }

    /// Solves the assembled matrix for `rhs` (refactoring with the previous pivots when they
    /// remain acceptable). The factorization stops before a column that would pass the limit,
    /// and the solve runs only when its fixed count fits.
    pub fn solve(&mut self, rhs: &mut [f64]) -> Result<(), SimError> {
        let budget = self.left();
        let fresh = match &mut self.lu {
            Some(lu) => {
                let mut spent = 0;
                let r = lu.refactor(&self.mat.pattern, &self.mat.vals, budget, &mut spent);
                self.meter.factorization += spent;
                match r {
                    Some(Ok(())) => false,
                    Some(Err(e)) => return Err(self.lu_error(e)),
                    None => true,
                }
            }
            None => true,
        };
        if fresh {
            self.lu = None;
            let mut spent = 0;
            let r = Lu::factor(&self.mat.pattern, &self.mat.vals, &self.order, self.left(), &mut spent);
            self.meter.factorization += spent;
            let lu = r.map_err(|e| self.lu_error(e))?;
            self.lu = Some(lu);
        }
        let ops = self.lu.as_ref().expect("factored").solve_ops();
        if !self.room(ops) {
            return Err(SimError::Stopped);
        }
        self.meter.solve += self.lu.as_ref().expect("factored").solve(rhs);
        Ok(())
    }

    fn lu_error(&mut self, e: LuError) -> SimError {
        match e {
            LuError::Budget => self.stop(),
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
            let (mut rhs, noncon) = self.assemble(x, &lp)?;
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
                self.charge_other(5 * rhs.len() as u64)?;
                for (k, (new, old)) in rhs.iter().zip(x.iter()).enumerate() {
                    let tol = reltol * new.abs().max(old.abs()) + if k < n_nodes { vntol } else { abstol };
                    if (new - old).abs() > tol {
                        converged = false;
                        break;
                    }
                }
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
