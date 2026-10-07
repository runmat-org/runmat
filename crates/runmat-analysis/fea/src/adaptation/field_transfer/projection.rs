use std::collections::BTreeMap;

use runmat_meshing_core::{SolverEntityTransfer, SolverTransferMethod, StableDigest};

use super::{SolverFieldTransferError, SolverFieldTransferMethod};

pub(super) fn apply_projection(
    projection: &SolverEntityTransfer,
    source_values: &BTreeMap<StableDigest, &[f64]>,
    component_count: usize,
) -> Result<Vec<f64>, SolverFieldTransferError> {
    let mut result = Vec::with_capacity(component_count);
    for component in 0..component_count {
        let mut scale = 0.0_f64;
        let mut raw = crate::algebra::CompensatedSum::default();
        let mut first = None;
        let mut constant = true;
        let mut lost_product = false;
        for source in &projection.sources {
            let values = source_values
                .get(&source.stable_identity)
                .ok_or(SolverFieldTransferError::UnsupportedTopologyChange)?;
            let value = values[component];
            if let Some(first) = first {
                constant &= first == value;
            } else {
                first = Some(value);
            }
            scale = scale.max(value.abs());
            let term = source.weight * value;
            lost_product |= term == 0.0 && source.weight != 0.0 && value != 0.0;
            raw.add(term);
        }
        // Valid interpolation weights form a partition of unity. Preserve constants,
        // including the smallest subnormal value, without underflowing each product.
        let value = if constant {
            first.unwrap_or(0.0)
        } else if raw.total().is_finite() && !lost_product {
            raw.total()
        } else {
            // Rescale only after ordinary compensated arithmetic loses range; do not
            // erase a tiny surviving contribution in an otherwise finite cancellation.
            let mut sum = crate::algebra::CompensatedSum::default();
            for source in &projection.sources {
                let original = source_values[&source.stable_identity][component];
                let normalized = original / scale;
                let term = source.weight * normalized;
                if (normalized == 0.0 && original != 0.0 && source.weight != 0.0)
                    || (term == 0.0 && normalized != 0.0 && source.weight != 0.0)
                {
                    return Err(SolverFieldTransferError::InvalidFieldValues);
                }
                sum.add(term);
            }
            sum.total() * scale
        };
        if !value.is_finite() {
            return Err(SolverFieldTransferError::InvalidFieldValues);
        }
        result.push(value);
    }
    Ok(result)
}

/// Average finite operands without overflowing a large sum or underflowing tiny halves.
pub(super) fn linear_midpoint(left: f64, right: f64) -> f64 {
    let sum = left + right;
    if sum.is_finite() {
        0.5 * sum
    } else {
        0.5 * left + 0.5 * right
    }
}

pub(super) fn field_transfer_method(method: SolverTransferMethod) -> SolverFieldTransferMethod {
    match method {
        SolverTransferMethod::BarycentricInterpolation => {
            SolverFieldTransferMethod::BarycentricInterpolation
        }
        SolverTransferMethod::QuadraticLagrangeInterpolation => {
            SolverFieldTransferMethod::QuadraticLagrangeInterpolation
        }
        SolverTransferMethod::CentroidProjection => SolverFieldTransferMethod::CentroidProjection,
        SolverTransferMethod::ConservativeProjection => {
            SolverFieldTransferMethod::ConservativeProjection
        }
    }
}

#[cfg(test)]
mod range_tests {
    use super::*;
    use runmat_meshing_core::SolverTransferSource;
    #[test]
    fn linear_midpoints_preserve_finite_large_and_subnormal_values() {
        let tiny = f64::from_bits(1);
        assert_eq!(linear_midpoint(f64::MAX, f64::MAX), f64::MAX);
        assert_eq!(linear_midpoint(tiny, tiny), tiny);
        assert_eq!(linear_midpoint(tiny, 2.0 * tiny), 2.0 * tiny);
        assert_eq!(linear_midpoint(f64::MAX, -f64::MAX), 0.0);
    }

    #[test]
    fn tet10_centroid_projection_preserves_representable_large_cancellation() {
        // At the Tet10 centroid, corner shapes are -1/8 and midside shapes 1/4.
        // These nodal values give exactly MAX: four negative corners contribute
        // MAX/2; four positive and two negative midsides contribute MAX/2.
        let values: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                vec![if !(4..8).contains(&i) {
                    -f64::MAX
                } else {
                    f64::MAX
                }]
            })
            .collect();
        let sources = (0..10)
            .map(|i| SolverTransferSource {
                stable_identity: StableDigest::from_bytes([i as u8 + 1; 32]),
                weight: if i < 4 { -0.125 } else { 0.25 },
            })
            .collect();
        let projection = SolverEntityTransfer {
            target_stable_identity: StableDigest::from_bytes([99; 32]),
            method: SolverTransferMethod::QuadraticLagrangeInterpolation,
            sources,
        };
        let borrowed = values
            .iter()
            .enumerate()
            .map(|(i, v)| (StableDigest::from_bytes([i as u8 + 1; 32]), v.as_slice()))
            .collect();
        assert_eq!(
            apply_projection(&projection, &borrowed, 1).unwrap(),
            vec![f64::MAX]
        );
    }
}
