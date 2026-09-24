use crate::assembly::{
    elements::solid::{
        strain_displacement_matrix, Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry,
    },
    SolidRecoveryElementSummary, SolidRecoveryGeometry,
};
use crate::planning::resource::{
    execute_scoped_chunks, LocalExecutionPlan, ScopedLocalExecutionError,
};
use crate::progress::{cancellation_token, FeaCancellationToken};

use super::error::StructuralFieldRecoveryError;

const COMPONENT_COUNT: usize = 6;

#[derive(Debug, PartialEq)]
pub(super) struct SolidRecoveredFields {
    pub strain: Vec<f64>,
    pub stress: Vec<f64>,
    pub strain_energy_density: Vec<f64>,
    pub integration_point_strain: Vec<f64>,
    pub integration_point_stress: Vec<f64>,
}

impl SolidRecoveredFields {
    fn with_element_capacity(element_count: usize, retain_points: bool) -> Self {
        Self {
            strain: Vec::with_capacity(element_count * COMPONENT_COUNT),
            stress: Vec::with_capacity(element_count * COMPONENT_COUNT),
            strain_energy_density: Vec::with_capacity(element_count),
            integration_point_strain: Vec::with_capacity(if retain_points {
                element_count * RECOVERY_POINT_COUNT * COMPONENT_COUNT
            } else {
                0
            }),
            integration_point_stress: Vec::with_capacity(if retain_points {
                element_count * RECOVERY_POINT_COUNT * COMPONENT_COUNT
            } else {
                0
            }),
        }
    }

    fn append(&mut self, other: &mut Self) {
        self.strain.append(&mut other.strain);
        self.stress.append(&mut other.stress);
        self.strain_energy_density
            .append(&mut other.strain_energy_density);
        self.integration_point_strain
            .append(&mut other.integration_point_strain);
        self.integration_point_stress
            .append(&mut other.integration_point_stress);
    }
}

const RECOVERY_POINT_COUNT: usize = 4;

struct ElementRecoveryPoints {
    strain: [[f64; COMPONENT_COUNT]; RECOVERY_POINT_COUNT],
    integration_weight_m3: [f64; RECOVERY_POINT_COUNT],
}

fn recover_solid_fields_checked(
    elements: &[SolidRecoveryElementSummary],
    displacement: &[f64],
    cancellation: &FeaCancellationToken,
    retain_points: bool,
) -> Result<SolidRecoveredFields, StructuralFieldRecoveryError> {
    let mut fields = SolidRecoveredFields::with_element_capacity(elements.len(), retain_points);
    for element in elements {
        if cancellation.is_cancelled() {
            return Err(StructuralFieldRecoveryError::Cancelled);
        }
        let recovered = recover_element_strain(element, displacement)?;
        let point_stress = recovered
            .strain
            .map(|point_strain| recover_element_stress(element, point_strain))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let volume = recovered.integration_weight_m3.iter().sum::<f64>();
        if !volume.is_finite() || volume <= 0.0 {
            return Err(StructuralFieldRecoveryError::NonFiniteResult {
                element_id: element.element_id.clone(),
            });
        }
        let element_strain = std::array::from_fn::<_, COMPONENT_COUNT, _>(|component| {
            recovered
                .strain
                .iter()
                .zip(recovered.integration_weight_m3)
                .map(|(value, weight)| value[component] * weight)
                .sum::<f64>()
                / volume
        });
        let element_stress = std::array::from_fn::<_, COMPONENT_COUNT, _>(|component| {
            point_stress
                .iter()
                .zip(recovered.integration_weight_m3)
                .map(|(value, weight)| value[component] * weight)
                .sum::<f64>()
                / volume
        });
        let energy = recovered
            .strain
            .iter()
            .zip(&point_stress)
            .zip(recovered.integration_weight_m3)
            .map(|((point_strain, point_stress), weight)| {
                0.5 * weight
                    * point_strain
                        .iter()
                        .zip(point_stress)
                        .map(|(strain, stress)| strain * stress)
                        .sum::<f64>()
            })
            .sum::<f64>()
            / volume;
        if !energy.is_finite()
            || element_strain.iter().any(|value| !value.is_finite())
            || element_stress.iter().any(|value| !value.is_finite())
        {
            return Err(StructuralFieldRecoveryError::NonFiniteResult {
                element_id: element.element_id.clone(),
            });
        }
        fields.strain.extend(element_strain);
        fields.stress.extend(element_stress);
        fields.strain_energy_density.push(energy);
        if retain_points {
            fields
                .integration_point_strain
                .extend(recovered.strain.into_iter().flatten());
            fields
                .integration_point_stress
                .extend(point_stress.into_iter().flatten());
        }
    }
    Ok(fields)
}

pub(super) fn recover_solid_fields_with_plan(
    elements: &[SolidRecoveryElementSummary],
    displacement: &[f64],
    plan: LocalExecutionPlan,
    retain_points: bool,
) -> Result<SolidRecoveredFields, StructuralFieldRecoveryError> {
    let cancellation = cancellation_token();
    if cancellation.is_cancelled() {
        return Err(StructuralFieldRecoveryError::Cancelled);
    }
    if elements.is_empty() || plan.is_serial() {
        return recover_solid_fields_checked(elements, displacement, &cancellation, retain_points);
    }
    let partitions = execute_scoped_chunks(plan, elements.len() as u64, |chunk| {
        recover_solid_fields_checked(
            &elements[chunk.start() as usize..chunk.end_exclusive() as usize],
            displacement,
            &cancellation,
            retain_points,
        )
    })
    .map_err(|error| match error {
        ScopedLocalExecutionError::Work { source, .. } => source,
        other => StructuralFieldRecoveryError::ScopedExecution {
            reason: other.to_string(),
        },
    })?;
    if cancellation.is_cancelled() {
        return Err(StructuralFieldRecoveryError::Cancelled);
    }
    let mut combined = SolidRecoveredFields::with_element_capacity(elements.len(), retain_points);
    for mut partition in partitions {
        combined.append(&mut partition);
    }
    Ok(combined)
}

fn recover_element_strain(
    element: &SolidRecoveryElementSummary,
    displacement: &[f64],
) -> Result<ElementRecoveryPoints, StructuralFieldRecoveryError> {
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
            let strain = std::array::from_fn(|component| {
                b[component]
                    .iter()
                    .zip(element_displacement)
                    .map(|(shape, value)| shape * value)
                    .sum()
            });
            let volume = Tetrahedron4ElementGeometry {
                nodes_m: *coordinates_m,
            }
            .volume_m3()
            .map_err(|source| StructuralFieldRecoveryError::Tetrahedron4 {
                element_id: element.element_id.clone(),
                source,
            })?;
            Ok(ElementRecoveryPoints {
                strain: [strain; RECOVERY_POINT_COUNT],
                integration_weight_m3: [volume / RECOVERY_POINT_COUNT as f64; RECOVERY_POINT_COUNT],
            })
        }
        SolidRecoveryGeometry::Tetrahedron10 {
            node_indices,
            coordinates_m,
        } => Tetrahedron10ElementGeometry {
            nodes_m: *coordinates_m,
        }
        .quadrature_strain(gather_displacement::<10, 30>(node_indices, displacement)?)
        .map(|points| ElementRecoveryPoints {
            strain: points.map(|point| point.strain),
            integration_weight_m3: points.map(|point| point.integration_weight_m3),
        })
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
    use crate::planning::resource::LocalThreadCount;
    use crate::progress::{replace_fea_progress_context, FeaProgressContext};

    fn two_threads() -> LocalExecutionPlan {
        LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap()
    }

    #[test]
    fn recovery_rejects_pre_cancelled_serial_and_parallel_plans() {
        let element = SolidRecoveryElementSummary {
            element_id: "cancelled".into(),
            region_id: "body".into(),
            material: SolidMaterial {
                youngs_modulus_pa: 10.0,
                poisson_ratio: 0.25,
                density_kg_per_m3: 1.0,
            },
            geometry: SolidRecoveryGeometry::Tetrahedron4 {
                node_indices: [0, 1, 2, 3],
                coordinates_m: [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        };
        let elements = [element.clone(), element];
        let cancellation = FeaCancellationToken::new();
        cancellation.cancel();
        let _context = replace_fea_progress_context(FeaProgressContext {
            cancellation,
            ..FeaProgressContext::default()
        });
        for plan in [LocalExecutionPlan::serial(), two_threads()] {
            let error =
                recover_solid_fields_with_plan(&elements, &[0.0; 12], plan, true).unwrap_err();
            assert_eq!(error, StructuralFieldRecoveryError::Cancelled);
        }
    }

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

        let recovered = recover_solid_fields_with_plan(
            &elements,
            &displacement,
            LocalExecutionPlan::serial(),
            true,
        )
        .unwrap();
        for _ in 0..5 {
            assert_eq!(
                recover_solid_fields_with_plan(&elements, &displacement, two_threads(), true)
                    .unwrap(),
                recovered,
            );
        }
        let history = recover_solid_fields_with_plan(
            &elements,
            &displacement,
            LocalExecutionPlan::serial(),
            false,
        )
        .unwrap();
        assert_eq!(history.strain, recovered.strain);
        assert_eq!(history.stress, recovered.stress);
        assert_eq!(
            history.strain_energy_density,
            recovered.strain_energy_density
        );
        assert!(history.integration_point_strain.is_empty());
        assert!(history.integration_point_stress.is_empty());
        assert_eq!(history.integration_point_strain.capacity(), 0);
        assert_eq!(history.integration_point_stress.capacity(), 0);
        assert_eq!(
            recover_solid_fields_with_plan(&elements, &displacement, two_threads(), false).unwrap(),
            history,
        );

        assert_eq!(&recovered.strain[..6], &recovered.strain[6..]);
        for (soft, hard) in recovered.stress[..6].iter().zip(&recovered.stress[6..]) {
            assert!((*hard - 2.0 * soft).abs() <= 1.0e-14);
        }
        let (soft_points, hard_points) = recovered.integration_point_stress.split_at(4 * 6);
        for (soft, hard) in soft_points.iter().zip(hard_points) {
            assert!((*hard - 2.0 * soft).abs() <= 1.0e-14);
        }
        assert!(
            (recovered.strain_energy_density[1] - 2.0 * recovered.strain_energy_density[0]).abs()
                <= 1.0e-14
        );
    }

    #[test]
    fn tetrahedron10_recovery_uses_assigned_material_at_every_integration_point() {
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let midpoint = |left: [f64; 3], right: [f64; 3]| {
            std::array::from_fn(|axis| 0.5 * (left[axis] + right[axis]))
        };
        let coordinates_m = [
            corners[0],
            corners[1],
            corners[2],
            corners[3],
            midpoint(corners[0], corners[1]),
            midpoint(corners[1], corners[2]),
            midpoint(corners[2], corners[0]),
            midpoint(corners[0], corners[3]),
            midpoint(corners[1], corners[3]),
            midpoint(corners[2], corners[3]),
        ];
        let material = |youngs_modulus_pa| SolidMaterial {
            youngs_modulus_pa,
            poisson_ratio: 0.25,
            density_kg_per_m3: 1.0,
        };
        let geometry = SolidRecoveryGeometry::Tetrahedron10 {
            node_indices: std::array::from_fn(|index| index),
            coordinates_m,
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
                material: material(30.0),
                geometry,
            },
        ];
        let displacement = coordinates_m
            .into_iter()
            .flat_map(|coordinate| [coordinate[0].powi(2), 0.0, 0.0])
            .collect::<Vec<_>>();

        let recovered = recover_solid_fields_with_plan(
            &elements,
            &displacement,
            LocalExecutionPlan::serial(),
            true,
        )
        .unwrap();
        assert_eq!(
            recover_solid_fields_with_plan(&elements, &displacement, two_threads(), true).unwrap(),
            recovered,
        );

        let (soft, hard) = recovered.integration_point_stress.split_at(4 * 6);
        for (soft, hard) in soft.iter().zip(hard) {
            assert!((*hard - 3.0 * soft).abs() <= 1.0e-13);
        }
    }
}
