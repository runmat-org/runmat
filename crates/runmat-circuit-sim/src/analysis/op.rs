//! DC operating point: Newton from the initial guess, then gmin stepping, then source stepping.

use super::engine::{Engine, LoadParams};
use crate::SimError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpStrategy {
    #[default]
    Direct,
    GminStepping,
    SourceStepping,
}

/// Solves the operating point in place. `base` sets the mode, time and forcing.
pub fn solve(e: &mut Engine, x: &mut Vec<f64>, base: LoadParams) -> Result<OpStrategy, SimError> {
    let itl = e.c.options.itl1;
    let start = x.clone();
    // Nodesets guide the first solve, then are released.
    if !e.c.nodesets.is_empty() {
        let mut p = base;
        p.force_nodesets = true;
        p.init_junctions = true;
        let _ = e.newton(x, &p, itl)?;
    }
    let mut p = base;
    p.init_junctions = e.c.nodesets.is_empty();
    if e.newton(x, &p, itl)?.is_some() {
        e.accept(x);
        return Ok(OpStrategy::Direct);
    }
    // Gmin stepping: a conductance from every node to ground, reduced a decade at a time.
    *x = start.clone();
    let mut g = 1e-2;
    let mut ok = true;
    let mut first = true;
    while g > e.c.options.gmin * 0.5 {
        let mut p = base;
        p.gdiag = g;
        p.init_junctions = first;
        first = false;
        if e.newton(x, &p, itl)?.is_none() {
            ok = false;
            break;
        }
        g /= 10.0;
    }
    if ok && e.newton(x, &base, itl)?.is_some() {
        e.accept(x);
        return Ok(OpStrategy::GminStepping);
    }
    // Source stepping: ramp every independent source from zero.
    *x = start;
    let mut scale: f64 = 0.0;
    let mut step: f64 = 0.1;
    let mut p = base;
    p.src_scale = 0.0;
    p.init_junctions = true;
    if e.newton(x, &p, itl)?.is_none() {
        return Err(no_conv(e, "source stepping could not start at zero sources"));
    }
    while scale < 1.0 {
        let next = (scale + step).min(1.0);
        let saved = (x.clone(), e.mem.clone());
        let mut p = base;
        p.src_scale = next;
        if e.newton(x, &p, itl)?.is_some() {
            scale = next;
            step = (step * 1.5).min(0.5);
        } else {
            (*x, e.mem) = saved;
            step /= 4.0;
            if step < 1e-7 {
                return Err(no_conv(e, &format!("source stepping stalled at {:.4}% of the sources", scale * 100.0)));
            }
        }
    }
    e.accept(x);
    Ok(OpStrategy::SourceStepping)
}

fn no_conv(e: &Engine, why: &str) -> SimError {
    SimError::NoConvergence {
        analysis: "operating point".into(),
        detail: format!("{why} (direct Newton, gmin stepping and source stepping all failed; {} unknowns)", e.c.size()),
    }
}
