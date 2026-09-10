use crate::assembly::{
    elements::solid::{
        strain_displacement_matrix, Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry,
    },
    SolidRecoveryElementSummary, SolidRecoveryGeometry,
};

use super::error::StructuralFieldRecoveryError;

const COMPONENT_COUNT: usize = 6;

pub(super) struct SolidRecoveredFields {
    pub strain: Vec<f64>,
    pub stress: Vec<f64>,
    pub strain_energy_density: Vec<f64>,
}

pub(super) fn recover_solid_fields(
    elements: &[SolidRecoveryElementSummary],
    displacement: &[f64],
) -> Result<SolidRecoveredFields, StructuralFieldRecoveryError> {
    let mut strain = Vec::with_capacity(elements.len() * COMPONENT_COUNT);
    let mut stress = Vec::with_capacity(elements.len() * COMPONENT_COUNT);
    let mut strain_energy_density = Vec::with_capacity(elements.len());
    for element in elements {
        let element_strain = recover_element_strain(element, displacement)?;
        let element_stress = recover_element_stress(element, element_strain)?;
        let energy = 0.5
            * element_strain
                .iter()
                .zip(element_stress)
                .map(|(strain, stress)| strain * stress)
                .sum::<f64>();
        if !energy.is_finite() {
            return Err(StructuralFieldRecoveryError::NonFiniteResult {
                element_id: element.element_id.clone(),
            });
        }
        strain.extend(element_strain);
        stress.extend(element_stress);
        strain_energy_density.push(energy);
    }
    Ok(SolidRecoveredFields {
        strain,
        stress,
        strain_energy_density,
    })
}

fn recover_element_strain(
    element: &SolidRecoveryElementSummary,
    displacement: &[f64],
) -> Result<[f64; COMPONENT_COUNT], StructuralFieldRecoveryError> {
    match &element.geometry {
        SolidRecoveryGeometry::Tetrahedron4 {
            node_indices,
            coordinates_m,
        } => {
            let element_displacement = gather_displacement::<4, 12>(node_indices, displacement)?;
            let b = strain_displacement_matrix(Tetrahedron4ElementGeometry {
                nodes_m: *coordinates_m,
            })
            .map_err(|source| StructuralFieldRecoveryError::Tetrahedron4 {
                element_id: element.element_id.clone(),
                source,
            })?;
            Ok(std::array::from_fn(|component| {
                b[component]
                    .iter()
                    .zip(element_displacement)
                    .map(|(shape, value)| shape * value)
                    .sum()
            }))
        }
        SolidRecoveryGeometry::Tetrahedron10 {
            node_indices,
            coordinates_m,
        } => Tetrahedron10ElementGeometry {
            nodes_m: *coordinates_m,
        }
        .volume_average_strain(gather_displacement::<10, 30>(node_indices, displacement)?)
        .map_err(|source| StructuralFieldRecoveryError::Tetrahedron10 {
            element_id: element.element_id.clone(),
            source,
        }),
    }
}

fn gather_displacement<const N: usize, const D: usize>(
    node_indices: &[usize; N],
    displacement: &[f64],
) -> Result<[f64; D], StructuralFieldRecoveryError> {
    debug_assert_eq!(D, N * 3);
    let mut gathered = [0.0; D];
    for (local_node, node_index) in node_indices.iter().copied().enumerate() {
        let start = node_index
            .checked_mul(3)
            .ok_or(StructuralFieldRecoveryError::InvalidDisplacement { node_index })?;
        let nodal = displacement
            .get(start..start + 3)
            .filter(|values| values.iter().all(|value| value.is_finite()))
            .ok_or(StructuralFieldRecoveryError::InvalidDisplacement { node_index })?;
        gathered[local_node * 3..local_node * 3 + 3].copy_from_slice(nodal);
    }
    Ok(gathered)
}

fn recover_element_stress(
    element: &SolidRecoveryElementSummary,
    strain: [f64; COMPONENT_COUNT],
) -> Result<[f64; COMPONENT_COUNT], StructuralFieldRecoveryError> {
    let lambda = element.material.lame_lambda_pa().map_err(|_| {
        StructuralFieldRecoveryError::InvalidMaterial {
            element_id: element.element_id.clone(),
        }
    })?;
    let mu = element.material.shear_modulus_pa().map_err(|_| {
        StructuralFieldRecoveryError::InvalidMaterial {
            element_id: element.element_id.clone(),
        }
    })?;
    let trace = strain[0] + strain[1] + strain[2];
    let stress = [
        lambda * trace + 2.0 * mu * strain[0],
        lambda * trace + 2.0 * mu * strain[1],
        lambda * trace + 2.0 * mu * strain[2],
        mu * strain[3],
        mu * strain[4],
        mu * strain[5],
    ];
    if stress.iter().all(|value| value.is_finite()) {
        Ok(stress)
    } else {
        Err(StructuralFieldRecoveryError::NonFiniteResult {
            element_id: element.element_id.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::elements::solid::SolidMaterial;

    #[test]
    fn recovery_uses_each_elements_assigned_material() {
        let geometry = SolidRecoveryGeometry::Tetrahedron4 {
            node_indices: [0, 1, 2, 3],
            coordinates_m: [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        };
        let material = |youngs_modulus_pa| SolidMaterial {
            youngs_modulus_pa,
            poisson_ratio: 0.25,
            density_kg_per_m3: 1.0,
        };
        let elements = [
            SolidRecoveryElementSummary {
                element_id: "soft".into(),
                region_id: "soft-region".into(),
                material: material(10.0),
                geometry: geometry.clone(),
            },
            SolidRecoveryElementSummary {
                element_id: "hard".into(),
                region_id: "hard-region".into(),
                material: material(20.0),
                geometry,
            },
        ];
        let displacement = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let recovered = recover_solid_fields(&elements, &displacement).unwrap();

        assert_eq!(&recovered.strain[..6], &recovered.strain[6..]);
        for (soft, hard) in recovered.stress[..6].iter().zip(&recovered.stress[6..]) {
            assert!((*hard - 2.0 * soft).abs() <= 1.0e-14);
        }
    }
}
