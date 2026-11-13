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

pub fn frequencies(
    sweep: AcSweep,
    points: usize,
    fstart: f64,
    fstop: f64,
) -> Result<Vec<f64>, SimError> {
    if !(fstart > 0.0 && fstop >= fstart && points >= 1) {
        return Err(SimError::BadRequest(
            "AC needs 0 < fstart ≤ fstop and at least one point".into(),
        ));
    }
    Ok(match sweep {
        AcSweep::Lin => {
            if points == 1 {
                vec![fstart]
            } else {
                (0..points)
                    .map(|k| fstart + (fstop - fstart) * k as f64 / (points - 1) as f64)
                    .collect()
            }
        }
        AcSweep::Dec | AcSweep::Oct => {
            let base: f64 = if sweep == AcSweep::Dec { 10.0 } else { 2.0 };
            let n = ((fstop / fstart).log(base) * points as f64 + 1e-9).floor() as usize;
            (0..=n)
                .map(|k| fstart * base.powf(k as f64 / points as f64))
                .collect()
        }
    })
}

pub fn sweep(e: &mut Engine, freqs: &[f64]) -> Result<AcWave, SimError> {
    let c = e.c;
    let mut x = vec![0.0; c.size()];
    op::solve(e, &mut x, LoadParams::dc())?;
    e.assemble(&x, &LoadParams::dc())?;
    let probes = Probes::new(c);
    let names: Vec<String> = probes.names[..probes.unknown_count()].to_vec();
    let mut wave = AcWave::new(&names);
    let mut mat: Matrix<Complex> = Matrix::new(e.mat.pattern.clone());
    let mut ord = order::minimum_degree(&mat.pattern);
    let mut lu: Option<Lu<Complex>> = None;
    let mut mem = e.mem.clone();
    for &f in freqs {
        // At the operation limit: the frequencies solved so far are the result.
        match point(e, &mut mat, &mut ord, &mut lu, &mut mem, &x, f) {
            Ok(rhs) => {
                let vals: Vec<Complex> = (0..probes.unknown_count())
                    .map(|k| rhs[probes.unknown(k)])
                    .collect();
                wave.push(f, vals);
            }
            Err(SimError::Stopped) => break,
            Err(err) => return Err(err),
        }
    }
    Ok(wave)
}

/// One frequency: the complex assembly (kept only when it fits under the limit), then the LU
/// with what is left of the limit as its budget, then the solve when its fixed count fits.
pub(super) fn point(
    e: &mut Engine,
    mat: &mut Matrix<Complex>,
    ord: &mut Vec<usize>,
    lu: &mut Option<Lu<Complex>>,
    mem: &mut [f64],
    x: &[f64],
    f: f64,
) -> Result<Vec<Complex>, SimError> {
    let c = e.c;
    let omega = 2.0 * std::f64::consts::PI * f;
    let mut rhs = loop {
        e.room_to_assemble()?;
        mat.clear();
        mat.ops = 0;
        let mut rhs = vec![Complex::default(); c.size()];
        let mut ld = Load {
            mode: Mode::Ac { omega },
            target: Target::Complex {
                mat: &mut *mat,
                rhs: &mut rhs,
            },
            x,
            mem: &mut *mem,
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
        if let Some(r) = c.options.rshunt {
            for k in 0..c.n_nodes {
                ld.mat(k, k, 1.0 / r);
            }
        }
        let ops = ld.ops;
        e.commit_assembly(dual::take_ops() + ops + mat.ops)?;
        if mat.needs_rebuild() {
            mat.rebuild();
            *ord = order::minimum_degree(&mat.pattern);
            *lu = None;
            continue;
        }
        break rhs;
    };
    let refreshed = match lu.as_mut() {
        Some(l) => {
            let mut spent = 0;
            let r = l.refactor(&mat.pattern, &mat.vals, e.left(), &mut spent);
            e.meter.factorization += spent;
            match r {
                Some(Ok(())) => false,
                Some(Err(err)) => return Err(lu_err(e, err)),
                None => true,
            }
        }
        None => true,
    };
    if refreshed {
        *lu = None;
        let mut spent = 0;
        let r = Lu::factor(&mat.pattern, &mat.vals, ord, e.left(), &mut spent);
        e.meter.factorization += spent;
        *lu = Some(r.map_err(|err| lu_err(e, err))?);
    }
    let l = lu.as_ref().expect("factored");
    if !e.room(l.solve_ops()) {
        return Err(SimError::Stopped);
    }
    e.meter.solve += l.solve(&mut rhs);
    Ok(rhs)
}

fn lu_err(e: &mut Engine, err: LuError) -> SimError {
    match err {
        LuError::Budget => e.stop(),
        LuError::Singular { column } => SimError::Singular {
            unknown: e.c.names[column].clone(),
        },
        LuError::NonFinite => SimError::NoConvergence {
            analysis: "ac".into(),
            detail: "a matrix entry is not finite".into(),
        },
    }
}
