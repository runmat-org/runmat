use crate::assembly::SolidRecoveryElementSummary;

use super::error::StructuralFieldRecoveryError;

pub(super) fn recover_volume_weighted_nodal_scalar(
    elements: &[SolidRecoveryElementSummary],
    element_values: &[f64],
    node_count: usize,
) -> Result<Vec<f64>, StructuralFieldRecoveryError> {
    if element_values.len() != elements.len() {
        return Err(StructuralFieldRecoveryError::ElementValueCountMismatch {
            expected: elements.len(),
            actual: element_values.len(),
        });
    }
    let mut nodal = vec![0.0_f64; node_count];
    let mut weights = vec![0.0_f64; node_count];
    for (element, value) in elements.iter().zip(element_values.iter().copied()) {
        let volume = element.geometry.volume_m3().map_err(|error| {
            StructuralFieldRecoveryError::NonFiniteResult {
                element_id: format!("{} ({error})", element.element_id),
            }
        })?;
        for &node_index in element.geometry.node_indices() {
            let accumulator = nodal.get_mut(node_index).ok_or_else(|| {
                StructuralFieldRecoveryError::InvalidNodeIndex {
                    element_id: element.element_id.clone(),
                    node_index,
                }
            })?;
            *accumulator += value * volume;
            weights[node_index] += volume;
        }
    }
    for (value, weight) in nodal.iter_mut().zip(weights) {
        if weight > 0.0 {
            *value /= weight;
        }
    }
    Ok(nodal)
}

#[cfg(test)]
mod tests {
    use crate::assembly::{
        elements::solid::SolidMaterial, SolidRecoveryElementSummary, SolidRecoveryGeometry,
    };

    use super::recover_volume_weighted_nodal_scalar;

    #[test]
    fn adjacent_tetrahedra_use_physical_volume_weights() {
        let material = SolidMaterial {
            youngs_modulus_pa: 1.0,
            poisson_ratio: 0.25,
            density_kg_per_m3: 1.0,
        };
        let elements = vec![
            SolidRecoveryElementSummary {
                element_id: "small".into(),
                region_id: "body".into(),
                material,
                thermal_eigenstrain: None,
                geometry: SolidRecoveryGeometry::Tetrahedron4 {
                    node_indices: [0, 1, 2, 3],
                    coordinates_m: [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            },
            SolidRecoveryElementSummary {
                element_id: "large".into(),
                region_id: "body".into(),
                material,
                thermal_eigenstrain: None,
                geometry: SolidRecoveryGeometry::Tetrahedron4 {
                    node_indices: [0, 4, 5, 6],
                    coordinates_m: [
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 2.0],
                    ],
                },
            },
        ];

        let recovered = recover_volume_weighted_nodal_scalar(&elements, &[10.0, 20.0], 7)
            .expect("valid adjacent tetrahedra should recover");

        assert!((recovered[0] - 170.0 / 9.0).abs() < 1.0e-12);
        assert_eq!(&recovered[1..4], &[10.0, 10.0, 10.0]);
        assert_eq!(&recovered[4..7], &[20.0, 20.0, 20.0]);
    }
}
