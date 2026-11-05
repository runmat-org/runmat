//! AC small-signal analysis: the circuit linearized at its operating point, solved as
//! `(G + jωC)·x = b` at each frequency with the complex sparse LU.

use super::engine::{Engine, LoadParams};
use super::{op, Probes};
use crate::devices::{Load, Mode, Target};
use crate::num::{dual, Complex};
use crate::results::AcWave;
use crate::sparse::lu::{Lu, LuError};
use crate::sparse::{order, Matrix};
use crate::SimError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcSweep {
    Dec,
    Oct,
    Lin,
}

pub fn frequencies(sweep: AcSweep, points: usize, fstart: f64, fstop: f64) -> Result<Vec<f64>, SimError> {
    if !(fstart > 0.0 && fstop >= fstart && points >= 1) {
        return Err(SimError::BadRequest("AC needs 0 < fstart ≤ fstop and at least one point".into()));
    }
    Ok(match sweep {
        AcSweep::Lin => {
            if points == 1 {
                vec![fstart]
            } else {
                (0..points).map(|k| fstart + (fstop - fstart) * k as f64 / (points - 1) as f64).collect()
            }
        }
        AcSweep::Dec | AcSweep::Oct => {
            let base: f64 = if sweep == AcSweep::Dec { 10.0 } else { 2.0 };
            let n = ((fstop / fstart).log(base) * points as f64 + 1e-9).floor() as usize;
            (0..=n).map(|k| fstart * base.powf(k as f64 / points as f64)).collect()
        }
    })
}

pub fn sweep(e: &mut Engine, freqs: &[f64]) -> Result<AcWave, SimError> {
    let c = e.c;
    let mut x = vec![0.0; c.size()];
    op::solve(e, &mut x, LoadParams::dc())?;
    e.assemble(&x, &LoadParams::dc());
    let probes = Probes::new(c);
    let names: Vec<String> = probes.names[..probes.unknown_count()].to_vec();
    let mut wave = AcWave::new(&names);
    let mut mat: Matrix<Complex> = Matrix::new(e.mat.pattern.clone());
    let mut ord = order::minimum_degree(&mat.pattern);
    let mut lu: Option<Lu<Complex>> = None;
    let mut mem = e.mem.clone();
    for &f in freqs {
        let omega = 2.0 * std::f64::consts::PI * f;
        let rhs = loop {
            mat.clear();
            mat.ops = 0;
            let mut rhs = vec![Complex::default(); c.size()];
            let mut ld = Load {
                mode: Mode::Ac { omega },
                target: Target::Complex { mat: &mut mat, rhs: &mut rhs },
                x: &x,
                mem: &mut mem,
                states: &mut e.states,
                time: 0.0,
                src_scale: 1.0,
                gmin: c.options.gmin,
                noncon: false,
                init_junctions: false,
                vt: e.vt,
                temp_k: c.options.temp_c + 273.15,
                ops: 0,
            };
            for d in &c.devices {
                d.load(&mut ld);
            }
            let ops = ld.ops;
            e.meter.assembly += dual::take_ops() + ops + mat.ops;
            if mat.needs_rebuild() {
                mat.rebuild();
                ord = order::minimum_degree(&mat.pattern);
                lu = None;
                continue;
            }
            break rhs;
        };
        let mut rhs = rhs;
        let refreshed = match lu.as_mut().map(|l| l.refactor(&mat.pattern, &mat.vals)) {
            Some(Some(Ok(()))) => false,
            Some(Some(Err(err))) => return Err(lu_err(e, err)),
            _ => true,
        };
        if refreshed {
            lu = Some(Lu::factor(&mat.pattern, &mat.vals, &ord).map_err(|err| lu_err(e, err))?);
        }
        let l = lu.as_ref().expect("factored");
        e.meter.factorization += l.ops;
        e.meter.solve += l.solve(&mut rhs);
        let vals: Vec<Complex> = (0..probes.unknown_count()).map(|k| rhs[probes.unknown(k)]).collect();
        wave.push(f, vals);
    }
    Ok(wave)
}

fn lu_err(e: &Engine, err: LuError) -> SimError {
    match err {
        LuError::Singular { column } => SimError::Singular { unknown: e.c.names[column].clone() },
        LuError::NonFinite => SimError::NoConvergence { analysis: "ac".into(), detail: "a matrix entry is not finite".into() },
    }
}
