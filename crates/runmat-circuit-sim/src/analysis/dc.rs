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
        let first = if k > 0 { e.newton(&mut x, &LoadParams::dc(), itl) } else { Ok(None) };
        let point = match first {
            Ok(Some(_)) => {
                e.accept(&x);
                Ok(())
            }
            Ok(None) => op::solve(e, &mut x, LoadParams::dc()).map(|_| ()),
            Err(err) => Err(err),
        };
        match point.and_then(|()| e.assemble(&x, &LoadParams::dc())).and_then(|_| e.probe_values(&probes, &x)) {
            // At the operation limit after the first point: the points so far are the result.
            Err(SimError::Stopped) if k > 0 => break,
            Err(err) => return Err(err),
            Ok(vals) => wave.push(v, vals),
        }
    }
    Ok(wave)
}
