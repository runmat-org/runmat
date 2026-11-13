//! Devices and the load context they stamp through.
//!
//! A device writes its contribution to the modified nodal analysis system once, through
//! [`Load`]: nonlinear branch currents and charges as [`Dual`] numbers of their controlling
//! voltages, branch equations as plain coefficients. The same code serves every analysis:
//!
//! - DC: currents are linearized (Newton companion: conductances plus an equivalent source),
//!   charges carry no current;
//! - transient: each charge is integrated by the current method ([`States`]) into a current;
//! - AC: currents contribute their conductances and charges `jω` times their capacitances, at
//!   the operating point.

pub mod bjt;
pub mod controlled;
pub mod diode;
pub mod junction;
pub mod mos_levels;
pub mod mosfet;
pub mod opamp;
pub mod passive;
pub mod source;
pub mod switch;
pub mod waveform;

use crate::num::{Complex, Dual};
use crate::sparse::{Matrix, GND};

/// Boltzmann constant over electron charge (V/K).
pub const K_OVER_Q: f64 = 1.380_649e-23 / 1.602_176_634e-19;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Dc,
    Tran,
    Ac { omega: f64 },
}

/// Integration method for charges and fluxes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Method {
    #[default]
    /// Backward Euler (order 1), used for the first step and after breakpoints.
    Euler,
    Trapezoidal,
    Gear2,
}

/// Charge and flux history, one slot per state.
#[derive(Clone, Debug, Default)]
pub struct States {
    /// `q[s] = [q(n+1), q(n), q(n-1), q(n-2)]`.
    pub q: Vec<[f64; 4]>,
    /// `i[s] = [dq/dt(n+1), dq/dt(n)]`.
    pub i: Vec<[f64; 2]>,
    /// The derivative's tolerance class: current (charges) or voltage (fluxes).
    pub is_flux: Vec<bool>,
    pub method: Method,
    /// Step sizes `h(n+1) = t(n+1) - t(n)`, `h(n)`.
    pub h: [f64; 2],
}

impl States {
    pub fn new(is_flux: Vec<bool>) -> Self {
        let n = is_flux.len();
        States {
            q: vec![[0.0; 4]; n],
            i: vec![[0.0; 2]; n],
            is_flux,
            method: Method::Euler,
            h: [0.0; 2],
        }
    }
    /// `(a0, hist)` with `dq/dt(n+1) = a0·q(n+1) + hist`.
    pub fn coefficients(&self, s: usize) -> (f64, f64) {
        let [_, q1, q2, _] = self.q[s];
        let h = self.h[0];
        match self.method {
            Method::Euler => (1.0 / h, -q1 / h),
            Method::Trapezoidal => (2.0 / h, -2.0 / h * q1 - self.i[s][1]),
            Method::Gear2 => {
                let w = h / self.h[1];
                let a0 = (1.0 + 2.0 * w) / ((1.0 + w) * h);
                let a1 = -(1.0 + w) / h;
                let a2 = w * w / ((1.0 + w) * h);
                (a0, a1 * q1 + a2 * q2)
            }
        }
    }
    /// Evaluate the derivative from charge differences. Expanding the history
    /// into `a0*q + hist` loses small currents when charge is large relative to
    /// the timestep, including a nonzero current for an unchanged charge.
    pub fn derivative<const N: usize>(&self, s: usize, q: Dual<N>) -> Dual<N> {
        let [_, q1, q2, _] = self.q[s];
        let h = self.h[0];
        match self.method {
            Method::Euler => (q - q1) / h,
            Method::Trapezoidal => (q - q1) * (2.0 / h) - self.i[s][1],
            Method::Gear2 => {
                let w = h / self.h[1];
                let a0 = (1.0 + 2.0 * w) / ((1.0 + w) * h);
                let a2 = w * w / ((1.0 + w) * h);
                (q - q1) * a0 + a2 * (q2 - q1)
            }
        }
    }
    /// After an accepted step: shift the history.
    pub fn accept(&mut self) {
        for (q, i) in self.q.iter_mut().zip(self.i.iter_mut()) {
            *q = [q[0], q[0], q[1], q[2]];
            *i = [i[0], i[0]];
        }
    }
    /// Starts the history from the operating point (all past charges equal, no current).
    pub fn settle(&mut self) {
        for (q, i) in self.q.iter_mut().zip(self.i.iter_mut()) {
            *q = [q[0]; 4];
            *i = [0.0; 2];
        }
    }
}

pub enum Target<'a> {
    Real {
        mat: &'a mut Matrix<f64>,
        rhs: &'a mut [f64],
    },
    Complex {
        mat: &'a mut Matrix<Complex>,
        rhs: &'a mut [Complex],
    },
}

/// Everything a device sees while it loads.
pub struct Load<'a> {
    pub mode: Mode,
    pub target: Target<'a>,
    /// The present iterate (DC and transient) or the operating point (AC).
    pub x: &'a [f64],
    /// Device memory: limiting history and switch states.
    pub mem: &'a mut [f64],
    pub states: &'a mut States,
    pub time: f64,
    /// Source-stepping factor on independent sources (1 normally).
    pub src_scale: f64,
    /// Conductance placed across every junction.
    pub gmin: f64,
    /// Set by a device that limited a voltage this iteration (not converged yet).
    pub noncon: bool,
    /// First Newton iteration of an operating point: junctions start from their critical voltage.
    pub init_junctions: bool,
    /// Thermal voltage kT/q at the circuit temperature.
    pub vt: f64,
    pub temp_k: f64,
    /// Real operations of the stamping arithmetic outside the matrix (equivalent sources,
    /// fluxes); the matrix counts its own accumulations.
    pub ops: u64,
}

impl Load<'_> {
    #[inline]
    pub fn v(&self, n: usize) -> f64 {
        if n == GND {
            0.0
        } else {
            self.x[n]
        }
    }
    /// Adds a real coefficient to the matrix (for AC, its real part).
    pub fn mat(&mut self, r: usize, c: usize, v: f64) {
        match &mut self.target {
            Target::Real { mat, .. } => mat.add(r, c, v),
            Target::Complex { mat, .. } => mat.add(r, c, Complex::new(v, 0.0)),
        }
    }
    pub fn mat_c(&mut self, r: usize, c: usize, v: Complex) {
        if let Target::Complex { mat, .. } = &mut self.target {
            mat.add(r, c, v);
        }
    }
    /// Adds to the right-hand side (DC and transient only).
    pub fn rhs(&mut self, r: usize, v: f64) {
        if r == GND {
            return;
        }
        if let Target::Real { rhs, .. } = &mut self.target {
            rhs[r] += v;
            self.ops += 1;
        }
    }
    /// Adds an AC excitation to the right-hand side.
    pub fn rhs_ac(&mut self, r: usize, v: Complex) {
        if r == GND {
            return;
        }
        if let Target::Complex { rhs, .. } = &mut self.target {
            rhs[r] += v;
        }
    }
    pub fn is_ac(&self) -> bool {
        matches!(self.mode, Mode::Ac { .. })
    }
    /// A linear conductance between `a` and `b`.
    pub fn conductance(&mut self, a: usize, b: usize, g: f64) {
        self.mat(a, a, g);
        self.mat(b, b, g);
        self.mat(a, b, -g);
        self.mat(b, a, -g);
    }
    /// A current `i` flowing from node `a` through the device to node `b`, a function of the
    /// control voltages `V(p_k) − V(n_k)` with partials `i.d[k]`.
    pub fn current<const N: usize>(
        &mut self,
        a: usize,
        b: usize,
        i: Dual<N>,
        ctrl: [(usize, usize); N],
    ) {
        let vals = ctrl.map(|(p, n)| self.v(p) - self.v(n));
        self.current_at(a, b, i, ctrl, vals);
    }
    /// As [`Load::current`], with `i` evaluated at the control values `vals` (which differ from
    /// the iterate when a device limited them): the companion source is `i − Σ g_k·vals_k`.
    pub fn current_at<const N: usize>(
        &mut self,
        a: usize,
        b: usize,
        i: Dual<N>,
        ctrl: [(usize, usize); N],
        vals: [f64; N],
    ) {
        let mut ieq = i.v;
        for (k, &(p, n)) in ctrl.iter().enumerate() {
            let g = i.d[k];
            if g == 0.0 {
                continue;
            }
            self.mat(a, p, g);
            self.mat(a, n, -g);
            self.mat(b, p, -g);
            self.mat(b, n, g);
            ieq -= g * vals[k];
        }
        self.ops += 2 * N as u64 + 2;
        self.rhs(a, -ieq);
        self.rhs(b, ieq);
    }
    /// A charge `q` on the device between `a` and `b` (its current flows from `a` to `b`).
    pub fn charge<const N: usize>(
        &mut self,
        s: usize,
        a: usize,
        b: usize,
        q: Dual<N>,
        ctrl: [(usize, usize); N],
    ) {
        let vals = ctrl.map(|(p, n)| self.v(p) - self.v(n));
        self.charge_at(s, a, b, q, ctrl, vals);
    }
    pub fn charge_at<const N: usize>(
        &mut self,
        s: usize,
        a: usize,
        b: usize,
        q: Dual<N>,
        ctrl: [(usize, usize); N],
        vals: [f64; N],
    ) {
        match self.mode {
            Mode::Dc => {
                self.states.q[s][0] = q.v;
                self.states.i[s][0] = 0.0;
            }
            Mode::Tran => {
                self.states.q[s][0] = q.v;
                let i = self.states.derivative(s, q);
                self.states.i[s][0] = i.v;
                self.current_at(a, b, i, ctrl, vals);
            }
            Mode::Ac { omega } => {
                for (k, &(p, n)) in ctrl.iter().enumerate() {
                    if q.d[k] == 0.0 {
                        continue;
                    }
                    let y = Complex::new(0.0, omega * q.d[k]);
                    self.mat_c(a, p, y);
                    self.mat_c(a, n, -y);
                    self.mat_c(b, p, -y);
                    self.mat_c(b, n, y);
                }
            }
        }
    }
    /// A flux `φ = Σ coef·x[col]` whose derivative enters branch row `row` as `−dφ/dt`.
    pub fn flux(&mut self, s: usize, row: usize, terms: &[(usize, f64)]) {
        match self.mode {
            Mode::Dc => {
                let phi: f64 = terms.iter().map(|&(c, k)| k * self.v(c)).sum();
                self.states.q[s][0] = phi;
                self.states.i[s][0] = 0.0;
            }
            Mode::Tran => {
                let (a0, hist) = self.states.coefficients(s);
                let phi: f64 = terms.iter().map(|&(c, k)| k * self.v(c)).sum();
                self.states.q[s][0] = phi;
                self.states.i[s][0] = self.states.derivative(s, Dual::<0>::constant(phi)).v;
                self.ops += 4 * terms.len() as u64 + 3;
                for &(c, k) in terms {
                    self.mat(row, c, -a0 * k);
                }
                self.rhs(row, hist);
            }
            Mode::Ac { omega } => {
                for &(c, k) in terms {
                    self.mat_c(row, c, Complex::new(0.0, -omega * k));
                }
            }
        }
    }
}

/// Unknown and state allocation while a circuit is built.
#[derive(Debug, Default)]
pub struct Setup {
    pub names: Vec<String>,
    pub n_nodes: usize,
    pub state_is_flux: Vec<bool>,
    pub mem_init: Vec<f64>,
}

impl Setup {
    /// A new internal node (named `<device>#<suffix>`).
    pub fn internal(&mut self, device: &str, suffix: &str) -> usize {
        self.names.push(format!("{device}#{suffix}"));
        self.names.len() - 1
    }
    pub fn branch(&mut self, device: &str) -> usize {
        self.names.push(format!("i({device})"));
        self.names.len() - 1
    }
    pub fn state(&mut self, flux: bool) -> usize {
        self.state_is_flux.push(flux);
        self.state_is_flux.len() - 1
    }
    pub fn mem(&mut self, init: f64) -> usize {
        self.mem_init.push(init);
        self.mem_init.len() - 1
    }
}

/// A circuit element.
pub trait Device: Send + Sync {
    fn name(&self) -> &str;
    fn load(&self, ld: &mut Load);
    /// Time points where the device's stimulus has corners.
    fn breakpoints(&self, _tstop: f64, _out: &mut Vec<f64>) {}
    /// Called after an accepted time point or operating point.
    fn accept(&self, _x: &[f64], _mem: &mut [f64]) {}
    /// The device's branch unknown, if it has one.
    fn branch(&self) -> Option<usize> {
        None
    }
    /// Current through the device (A, into its first terminal) at `x`.
    fn probe_current(&self, _x: &[f64], _states: &States) -> Option<f64> {
        self.branch().map(|b| _x[b])
    }
    /// Transient with `uic`: sets the device's initial charge or flux from its `ic` parameter.
    fn apply_ic(&self, _states: &mut States) {}
    /// The device's noise generators at the operating point `sink.x` (see
    /// [`crate::analysis::noise`]); noiseless by default.
    fn noise(&self, _sink: &mut crate::analysis::noise::NoiseSink) {}
    /// Sets the DC value of an independent source (DC sweeps).
    fn set_dc(&self, _mem: &mut [f64], _value: f64) -> bool {
        false
    }
}

/// Node voltage helper for probe code.
pub fn node_v(x: &[f64], n: usize) -> f64 {
    if n == GND {
        0.0
    } else {
        x[n]
    }
}
