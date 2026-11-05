//! DC sweep of an independent source's value; each point starts from the previous solution.

use super::engine::{Engine, LoadParams};
use super::{op, Probes};
use crate::results::RealWave;
use crate::SimError;

pub fn sweep(e: &mut Engine, source: &str, values: &[f64]) -> Result<RealWave, SimError> {
    let c = e.c;
    let idx = *c
        .device_index
        .get(&source.to_ascii_lowercase())
        .ok_or_else(|| SimError::BadRequest(format!("no source named `{source}` to sweep")))?;
    let probes = Probes::new(c);
    let mut wave = RealWave::new(source.to_ascii_lowercase(), &probes.names);
    let mut x = vec![0.0; c.size()];
    let itl = c.options.itl2;
    for (k, &v) in values.iter().enumerate() {
        if !c.devices[idx].set_dc(&mut e.mem, v) {
            return Err(SimError::BadRequest(format!("`{source}` is not an independent source")));
        }
        let ok = k > 0 && e.newton(&mut x, &LoadParams::dc(), itl)?.is_some();
        if ok {
            e.accept(&x);
        } else {
            op::solve(e, &mut x, LoadParams::dc())?;
        }
        e.assemble(&x, &LoadParams::dc());
        wave.push(v, probes.values(c, &x, &e.states));
    }
    Ok(wave)
}
