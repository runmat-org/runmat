use crate::algebra::coercivity::NumericalCapacityError;
use crate::assembly::{
    elements::solid::{
        strain_displacement_matrix, Tetrahedron10CapacityError, Tetrahedron10ElementGeometry,
        Tetrahedron4ElementGeometry,
    },
    SolidRecoveryElementSummary, SolidRecoveryGeometry,
};
use crate::planning::resource::{
    execute_scoped_chunks, LocalExecutionPlan, ScopedLocalExecutionError,
};
use crate::progress::{cancellation_token, FeaCancellationToken};
use runmat_simulation_usage::{AsyncCapacityOwner, OperationCount};

use super::error::StructuralFieldRecoveryError;

const COMPONENT_COUNT: usize = 6;

mod stress;
mod thermal_integrals;
use stress::{recover_element_stress, thermal_eigenstrain_at};

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
    let future = recover_solid_fields_checked_with_capacity::<std::convert::Infallible>(
        elements,
        displacement,
        cancellation,
        retain_points,
        None,
    );
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match std::future::Future::poll(future.as_mut(), &mut context) {
        std::task::Poll::Ready(result) => result.map_err(|error| match error {
            NumericalCapacityError::Numerical(error) => error,
            NumericalCapacityError::Admission(never) => match never {},
        }),
        std::task::Poll::Pending => unreachable!("immediate solid recovery cannot suspend"),
    }
}

async fn recover_solid_fields_checked_with_capacity<A>(
    elements: &[SolidRecoveryElementSummary],
    displacement: &[f64],
    cancellation: &FeaCancellationToken,
    retain_points: bool,
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<SolidRecoveredFields, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    let mut fields = SolidRecoveredFields::with_element_capacity(elements.len(), retain_points);
    for element in elements {
        if cancellation.is_cancelled() {
            return Err(StructuralFieldRecoveryError::Cancelled.into());
        }
        let recovered = recover_element_strain_with_capacity(element, displacement, owner).await?;
        let mut point_stress = Vec::with_capacity(RECOVERY_POINT_COUNT);
        for (point, &point_strain) in recovered.strain.iter().enumerate() {
            // One response completes 23 outer operations. Its Lamé and shear
            // material primitives reserve another 6 + 3 inside the response.
            if let Some(owner) = owner {
                owner
                    .ensure_capacity(OperationCount::from_operations(23 + 6 + 3))
                    .await
                    .map_err(NumericalCapacityError::Admission)?;
            }
            point_stress.push(
                crate::algebra::capacity::measured(
                    owner,
                    runmat_simulation_usage::UsageCategory::Solve,
                    OperationCount::from_operations(23),
                    || {
                        recover_element_stress(
                            element,
                            point_strain,
                            thermal_eigenstrain_at(element, point),
                        )
                    },
                )
                .await?,
            );
        }
        let (element_strain, element_stress, energy) =
            if let Some(thermal) = &element.solved_thermal_quadrature {
                thermal_integrals::integrate_with_capacity(element, thermal, displacement, owner)
                    .await?
            } else {
                crate::algebra::capacity::measured(
                    owner,
                    runmat_simulation_usage::UsageCategory::Solve,
                    OperationCount::from_operations(179),
                    || {
                        let volume = recovered.integration_weight_m3.iter().sum::<f64>();
                        if !volume.is_finite() || volume <= 0.0 {
                            return Err(StructuralFieldRecoveryError::NonFiniteResult {
                                element_id: element.element_id.clone(),
                            });
                        }
                        let element_strain =
                            std::array::from_fn::<_, COMPONENT_COUNT, _>(|component| {
                                recovered
                                    .strain
                                    .iter()
                                    .zip(recovered.integration_weight_m3)
                                    .map(|(value, weight)| value[component] * weight)
                                    .sum::<f64>()
                                    / volume
                            });
                        let element_stress =
                            std::array::from_fn::<_, COMPONENT_COUNT, _>(|component| {
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
                            .enumerate()
                            .map(|(point, ((point_strain, point_stress), weight))| {
                                let eigenstrain = thermal_eigenstrain_at(element, point);
                                0.5 * weight
                                    * point_strain
                                        .iter()
                                        .enumerate()
                                        .zip(point_stress)
                                        .map(|((component, strain), stress)| {
                                            (strain - if component < 3 { eigenstrain } else { 0.0 })
                                                * stress
                                        })
                                        .sum::<f64>()
                            })
                            .sum::<f64>()
                            / volume;
                        Ok((element_strain, element_stress, energy))
                    },
                )
                .await?
            };
        if !energy.is_finite()
            || element_strain.iter().any(|value| !value.is_finite())
            || element_stress.iter().any(|value| !value.is_finite())
        {
            return Err(StructuralFieldRecoveryError::NonFiniteResult {
                element_id: element.element_id.clone(),
            }
            .into());
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
    let usage = runmat_simulation_usage::UsageContext::capture();
    let partitions = execute_scoped_chunks(plan, elements.len() as u64, |chunk| {
        let _usage = usage
            .as_ref()
            .map(runmat_simulation_usage::UsageContext::enter);
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

pub(super) async fn recover_solid_fields_with_plan_and_capacity<A>(
    elements: &[SolidRecoveryElementSummary],
    displacement: &[f64],
    plan: LocalExecutionPlan,
    retain_points: bool,
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<SolidRecoveredFields, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    match owner {
        None => recover_solid_fields_with_plan(elements, displacement, plan, retain_points)
            .map_err(NumericalCapacityError::Numerical),
        Some(owner) => {
            let cancellation = cancellation_token();
            recover_solid_fields_checked_with_capacity(
                elements,
                displacement,
                &cancellation,
                retain_points,
                Some(owner),
            )
            .await
        }
    }
}

async fn recover_element_strain_with_capacity<A>(
    element: &SolidRecoveryElementSummary,
    displacement: &[f64],
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<ElementRecoveryPoints, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    match &element.geometry {
        SolidRecoveryGeometry::Tetrahedron4 {
            node_indices,
            coordinates_m,
        } => {
            if let Some(owner) = owner {
                owner
                    .ensure_capacity(OperationCount::from_operations(120))
                    .await
                    .map_err(NumericalCapacityError::Admission)?;
            }
            let element_displacement = gather_displacement::<4, 12>(node_indices, displacement)?;
            let b = strain_displacement_matrix(Tetrahedron4ElementGeometry {
                nodes_m: coordinates_m.canonical_value(),
            })
            .map_err(|source| StructuralFieldRecoveryError::Tetrahedron4 {
                element_id: element.element_id.clone(),
                source,
            })?;
            let strain = crate::algebra::capacity::measured(
                owner,
                runmat_simulation_usage::UsageCategory::Solve,
                OperationCount::from_operations(6 * (12 + 11)),
                || {
                    Ok::<_, StructuralFieldRecoveryError>(std::array::from_fn(|component| {
                        b[component]
                            .iter()
                            .zip(element_displacement)
                            .map(|(shape, value)| shape * value)
                            .sum()
                    }))
                },
            )
            .await?;
            if let Some(owner) = owner {
                owner
                    .ensure_capacity(OperationCount::from_operations(24))
                    .await
                    .map_err(NumericalCapacityError::Admission)?;
            }
            let volume = Tetrahedron4ElementGeometry {
                nodes_m: coordinates_m.canonical_value(),
            }
            .volume_m3()
            .map_err(|source| StructuralFieldRecoveryError::Tetrahedron4 {
                element_id: element.element_id.clone(),
                source,
            })?;
            crate::algebra::capacity::measured(
                owner,
                runmat_simulation_usage::UsageCategory::Solve,
                OperationCount::from_operations(1),
                || {
                    Ok::<_, StructuralFieldRecoveryError>(ElementRecoveryPoints {
                        strain: [strain; RECOVERY_POINT_COUNT],
                        integration_weight_m3: [volume / RECOVERY_POINT_COUNT as f64;
                            RECOVERY_POINT_COUNT],
                    })
                },
            )
            .await
        }
        SolidRecoveryGeometry::Tetrahedron10 {
            node_indices,
            coordinates_m,
        } => {
            let geometry = Tetrahedron10ElementGeometry {
                nodes_m: coordinates_m.canonical_value(),
            };
            let displacement = gather_displacement::<10, 30>(node_indices, displacement)?;
            let points = match owner {
                Some(owner) => geometry
                    .quadrature_strain_with_capacity(displacement, owner)
                    .await
                    .map_err(|error| match error {
                        Tetrahedron10CapacityError::Numerical(source) => {
                            NumericalCapacityError::Numerical(
                                StructuralFieldRecoveryError::Tetrahedron10 {
                                    element_id: element.element_id.clone(),
                                    source,
                                },
                            )
                        }
                        Tetrahedron10CapacityError::Admission(error) => {
                            NumericalCapacityError::Admission(error)
                        }
                    })?,
                None => geometry.quadrature_strain(displacement).map_err(|source| {
                    StructuralFieldRecoveryError::Tetrahedron10 {
                        element_id: element.element_id.clone(),
                        source,
                    }
                })?,
            };
            Ok(ElementRecoveryPoints {
                strain: points.map(|point| point.strain),
                integration_weight_m3: points.map(|point| point.integration_weight_m3),
            })
        }
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

#[cfg(test)]
#[path = "solid/thermoelastic_tests.rs"]
mod thermoelastic_tests;

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
                youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(10.0),
                poisson_ratio: runmat_types::physical::PoissonRatio::new(0.25),
                density_kg_per_m3: runmat_types::physical::MassDensity::new(1.0),
            },
            thermal_eigenstrain: None,
            prescribed_temperature_k: None,
            solved_thermal_quadrature: None,
            geometry: SolidRecoveryGeometry::Tetrahedron4 {
                node_indices: [0, 1, 2, 3],
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]),
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

    // witness: FEA-STRUCTURAL-SOLID-001/material_recovery @ linear_structural/tet4/numerical_kernel;
    //   coverage: partial;
    //   boundary: Two Tet4 elements with different Young's moduli recover different stresses from the
    //   same strain, in serial and with two threads; Tet10 recovery is covered by the recovery field
    //   tests.
    #[test]
    fn recovery_uses_each_elements_assigned_material() {
        let geometry = SolidRecoveryGeometry::Tetrahedron4 {
            node_indices: [0, 1, 2, 3],
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        };
        let material = |youngs_modulus_pa| SolidMaterial {
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(youngs_modulus_pa),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(0.25),
            density_kg_per_m3: runmat_types::physical::MassDensity::new(1.0),
        };
        let elements = [
            SolidRecoveryElementSummary {
                element_id: "soft".into(),
                region_id: "soft-region".into(),
                material: material(10.0),
                thermal_eigenstrain: None,
                prescribed_temperature_k: None,
                solved_thermal_quadrature: None,
                geometry: geometry.clone(),
            },
            SolidRecoveryElementSummary {
                element_id: "hard".into(),
                region_id: "hard-region".into(),
                material: material(20.0),
                thermal_eigenstrain: None,
                prescribed_temperature_k: None,
                solved_thermal_quadrature: None,
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

    // witness: FEA-STRUCTURAL-REFERENCE-001/patch_reference_convergence @
    //   linear_structural/tet4/numerical_kernel; coverage: partial;
    //   boundary: A connected Tet4 affine patch recovers analytic strain, stress and energy.
    #[test]
    fn connected_tet4_affine_patch_recovers_analytic_strain_stress_and_energy() {
        // Four conforming cells share the centroid of the unit tetrahedron. The imposed affine
        // displacement has constant epsilon = diag(0.1, 0.2, 0.3) in every cell, independent of
        // its shape functions or partitioning. E = 100 Pa and nu = 0.25 give lambda = mu = 40 Pa.
        let coordinates = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25],
        ];
        let connectivity = [[4, 1, 2, 3], [0, 4, 2, 3], [0, 1, 4, 3], [0, 1, 2, 4]];
        let elements = connectivity.map(|node_indices| SolidRecoveryElementSummary {
            element_id: format!("tet-{node_indices:?}"),
            region_id: "body".into(),
            material: SolidMaterial {
                youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(100.0),
                poisson_ratio: runmat_types::physical::PoissonRatio::new(0.25),
                density_kg_per_m3: runmat_types::physical::MassDensity::new(1.0),
            },
            thermal_eigenstrain: None,
            prescribed_temperature_k: None,
            solved_thermal_quadrature: None,
            geometry: SolidRecoveryGeometry::Tetrahedron4 {
                node_indices,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(node_indices.map(|node| coordinates[node])),
            },
        });
        let displacement = coordinates
            .into_iter()
            .flat_map(|point| [0.1 * point[0], 0.2 * point[1], 0.3 * point[2]])
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
            recovered
        );
        let expected_strain = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0];
        let expected_stress = [32.0, 40.0, 48.0, 0.0, 0.0, 0.0];
        for strain in recovered.strain.chunks_exact(6) {
            for (actual, expected) in strain.iter().zip(expected_strain) {
                assert!((actual - expected).abs() <= 1.0e-12);
            }
        }
        for stress in recovered.stress.chunks_exact(6) {
            for (actual, expected) in stress.iter().zip(expected_stress) {
                assert!((actual - expected).abs() <= 1.0e-11);
            }
        }
        for energy in &recovered.strain_energy_density {
            assert!((energy - 12.8).abs() <= 1.0e-11);
        }
        for strain in recovered.integration_point_strain.chunks_exact(6) {
            for (actual, expected) in strain.iter().zip(expected_strain) {
                assert!((actual - expected).abs() <= 1.0e-12);
            }
        }
        for stress in recovered.integration_point_stress.chunks_exact(6) {
            for (actual, expected) in stress.iter().zip(expected_stress) {
                assert!((actual - expected).abs() <= 1.0e-11);
            }
        }

        let mut heterogeneous = elements.clone();
        heterogeneous[1].material.youngs_modulus_pa =
            runmat_types::physical::YoungsModulus::new(200.0);
        heterogeneous[3].material.youngs_modulus_pa =
            runmat_types::physical::YoungsModulus::new(200.0);
        let mixed = recover_solid_fields_with_plan(
            &heterogeneous,
            &displacement,
            LocalExecutionPlan::serial(),
            true,
        )
        .unwrap();
        assert_eq!(mixed.strain, recovered.strain);
        for element in 0..4 {
            let scale = if element % 2 == 1 { 2.0 } else { 1.0 };
            for (actual, reference) in mixed.stress[element * 6..][..6]
                .iter()
                .zip(&recovered.stress[element * 6..][..6])
            {
                assert!((actual - scale * reference).abs() <= 1.0e-11);
            }
            assert!(
                (mixed.strain_energy_density[element]
                    - scale * recovered.strain_energy_density[element])
                    .abs()
                    <= 1.0e-11
            );
        }
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
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(youngs_modulus_pa),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(0.25),
            density_kg_per_m3: runmat_types::physical::MassDensity::new(1.0),
        };
        let geometry = SolidRecoveryGeometry::Tetrahedron10 {
            node_indices: std::array::from_fn(|index| index),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates_m),
        };
        let elements = [
            SolidRecoveryElementSummary {
                element_id: "soft".into(),
                region_id: "soft-region".into(),
                material: material(10.0),
                thermal_eigenstrain: None,
                prescribed_temperature_k: None,
                solved_thermal_quadrature: None,
                geometry: geometry.clone(),
            },
            SolidRecoveryElementSummary {
                element_id: "hard".into(),
                region_id: "hard-region".into(),
                material: material(30.0),
                thermal_eigenstrain: None,
                prescribed_temperature_k: None,
                solved_thermal_quadrature: None,
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
