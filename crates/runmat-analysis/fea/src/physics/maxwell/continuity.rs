//! One original weak charge-continuity authority for solve and retained validation.
use super::{ComplexPhasor, ComplexVector3};
use crate::topology::{AnalysisProjection, EdgeQuadraturePoint};
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct ChargeContinuity {
    rows: BTreeMap<usize, usize>,
    conducted: Vec<ComplexPhasor>,
    displaced: Vec<ComplexPhasor>,
    impressed: Vec<ComplexPhasor>,
    term_scale: Vec<f64>,
}
impl ChargeContinuity {
    /// Test rows are the corner nodes off grounded faces: exactly the nodal gradients that
    /// belong to the free edge space.
    pub fn new(projection: &AnalysisProjection, grounded_nodes: &BTreeSet<usize>) -> Self {
        let interior_nodes: BTreeSet<_> = projection
            .elements
            .iter()
            .flat_map(|element| element.node_indices[..4].iter().copied())
            .filter(|node| !grounded_nodes.contains(node))
            .collect();
        let rows: BTreeMap<_, _> = interior_nodes
            .into_iter()
            .enumerate()
            .map(|(row, node)| (node, row))
            .collect();
        let zero = vec![ComplexPhasor::ZERO; rows.len()];
        Self {
            term_scale: vec![0.0; rows.len()],
            rows,
            conducted: zero.clone(),
            displaced: zero.clone(),
            impressed: zero,
        }
    }
    pub fn accumulate(
        &mut self,
        corners: &[usize],
        point: &EdgeQuadraturePoint,
        conduction: ComplexVector3,
        displacement: ComplexVector3,
        source: ComplexVector3,
        omega: f64,
    ) {
        for (local, node) in corners.iter().enumerate() {
            if let Some(&row) = self.rows.get(node) {
                for axis in 0..3 {
                    let factor = point.corner_gradients[local][axis] * point.volume_weight_m3;
                    self.conducted[row] += conduction[axis] * factor;
                    self.displaced[row] += displacement[axis].times_i() * (omega * factor);
                    self.impressed[row] += source[axis] * factor;
                    // Original uncancelled quadrature terms, not cancelled integrals.
                    self.term_scale[row] += factor.abs()
                        * (conduction[axis].abs()
                            + omega * displacement[axis].abs()
                            + source[axis].abs());
                }
            }
        }
    }
    pub fn relative_defect(&self) -> f64 {
        let defect = self
            .conducted
            .iter()
            .zip(&self.displaced)
            .zip(&self.impressed)
            .map(|((a, b), c)| *a + *b + *c)
            .fold(0.0_f64, |norm, value| {
                norm.hypot(value.real).hypot(value.imaginary)
            });
        let scale = self
            .term_scale
            .iter()
            .fold(0.0_f64, |norm, value| norm.hypot(*value));
        if !scale.is_finite() || !defect.is_finite() {
            // A mathematically unrepresentable norm is not zero charge error.
            // Both solve acceptance and retained validation reject nonfinite evidence.
            f64::NAN
        } else if scale > 0.0 {
            defect / scale
        } else {
            defect
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_charge_norm_does_not_square_away_extreme_finite_terms() {
        for magnitude in [1e-300, 1.0, 1e300] {
            let continuity = ChargeContinuity {
                rows: BTreeMap::from([(0, 0)]),
                conducted: vec![ComplexPhasor::new(magnitude, 0.0)],
                displaced: vec![ComplexPhasor::ZERO],
                impressed: vec![ComplexPhasor::ZERO],
                term_scale: vec![magnitude],
            };
            assert_eq!(continuity.relative_defect(), 1.0);
        }
        let overflow = ChargeContinuity {
            rows: BTreeMap::from([(0, 0), (1, 1)]),
            conducted: vec![ComplexPhasor::ZERO; 2],
            displaced: vec![ComplexPhasor::ZERO; 2],
            impressed: vec![ComplexPhasor::ZERO; 2],
            term_scale: vec![f64::MAX; 2],
        };
        assert!(!overflow.relative_defect().is_finite());
    }
}
