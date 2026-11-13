//! Standard-normal rejection sampler over the host MT word stream.
//!
//! The 128-strip construction follows the mathematical method of Marsaglia
//! and Tsang, Journal of Statistical Software 5(8), 2000. Tables are derived
//! here from the published strip area and tail boundary; no upstream code or
//! stream-specific output sequence is embedded. RunMat makes no claim that
//! another product consumes the same MT words for its Ziggurat transform.

use std::sync::OnceLock;

use super::super::RandomState;

const STRIPS: usize = 128;
const RIGHT_TAIL: f64 = 3.442_619_855_899;
const STRIP_AREA: f64 = 9.912_563_035_262_17e-3;
const SIGNED_WORD_SCALE: f64 = 2_147_483_648.0;

struct NormalStrips {
    cutoff: [u32; STRIPS],
    width: [f64; STRIPS],
    density: [f64; STRIPS],
}

impl NormalStrips {
    fn construct() -> Self {
        let mut strips = Self {
            cutoff: [0; STRIPS],
            width: [0.0; STRIPS],
            density: [0.0; STRIPS],
        };
        let mut edge = RIGHT_TAIL;
        let mut previous = edge;
        let base = STRIP_AREA / (-0.5 * edge * edge).exp();
        strips.cutoff[0] = (edge / base * SIGNED_WORD_SCALE) as u32;
        strips.width[0] = base / SIGNED_WORD_SCALE;
        strips.width[STRIPS - 1] = edge / SIGNED_WORD_SCALE;
        strips.density[0] = 1.0;
        strips.density[STRIPS - 1] = (-0.5 * edge * edge).exp();
        for layer in (1..STRIPS - 1).rev() {
            edge = (-2.0 * (STRIP_AREA / edge + (-0.5 * edge * edge).exp()).ln()).sqrt();
            strips.cutoff[layer + 1] = (edge / previous * SIGNED_WORD_SCALE) as u32;
            previous = edge;
            strips.width[layer] = edge / SIGNED_WORD_SCALE;
            strips.density[layer] = (-0.5 * edge * edge).exp();
        }
        strips
    }
}

fn strips() -> &'static NormalStrips {
    static STRIPS: OnceLock<NormalStrips> = OnceLock::new();
    STRIPS.get_or_init(NormalStrips::construct)
}

fn word(state: &mut RandomState) -> u32 {
    state
        .mt_mut()
        .expect("normal MT draw requires a unique admitted state")
        .next_u32()
}

/// Open uniform for logarithms, using one fresh MT word per rejection draw.
fn positive_uniform(state: &mut RandomState) -> f64 {
    (f64::from(word(state)) + 0.5) * (1.0 / 4_294_967_296.0)
}

pub(super) fn next(state: &mut RandomState) -> f64 {
    let table = strips();
    loop {
        let signed = word(state) as i32;
        let layer = (signed as u32 & 127) as usize;
        let value = f64::from(signed) * table.width[layer];
        if signed.unsigned_abs() < table.cutoff[layer] {
            return value;
        }
        if layer == 0 {
            loop {
                let excess = -positive_uniform(state).ln() / RIGHT_TAIL;
                let ordinate = -positive_uniform(state).ln();
                if 2.0 * ordinate >= excess * excess {
                    return if signed >= 0 {
                        RIGHT_TAIL + excess
                    } else {
                        -RIGHT_TAIL - excess
                    };
                }
            }
        }
        let threshold = table.density[layer]
            + positive_uniform(state) * (table.density[layer - 1] - table.density[layer]);
        if threshold < (-0.5 * value * value).exp() {
            return value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_geometry_is_finite_monotone_and_has_a_rejection_layer() {
        let strips = strips();
        assert_eq!(strips.cutoff[1], 0);
        assert!(strips.cutoff[0] > 0);
        for index in 1..STRIPS {
            assert!(strips.width[index].is_finite());
            assert!(strips.width[index] > 0.0);
            assert!(strips.density[index].is_finite());
        }
        for index in 2..STRIPS {
            assert!(strips.width[index] >= strips.width[index - 1]);
        }
    }
}
