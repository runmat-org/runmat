use crate::assembly::SolidRecoveryElementSummary;

use super::error::StructuralFieldRecoveryError;
use crate::algebra::{capacity::measured, coercivity::NumericalCapacityError};
use runmat_simulation_usage::{AsyncCapacityOwner, LedgerError, OperationCount, UsageCategory};

/// Nodal stress tensors recovered from each element's own stress field.
///
/// Each element evaluates its stress at its nodes from its integration-point samples: a Tet10
/// fits the linear field through its four symmetric points exactly and evaluates it at corners
/// and midsides; a Tet4 is constant. The tensors of the elements sharing a node are averaged
/// component by component with volume weights, within one region only: at a node shared by
/// several regions, each region is averaged on its own and the region whose averaged tensor has
/// the largest von Mises stress supplies the nodal value, so material interfaces are never
/// blended. Scalars (von Mises, principal stresses) are derived from these averaged tensors.
#[cfg(test)]
pub(super) fn recover_volume_weighted_nodal_tensor(
    elements: &[SolidRecoveryElementSummary],
    integration_point_stress: &[f64],
    node_count: usize,
) -> Result<Vec<f64>, StructuralFieldRecoveryError> {
    use futures::FutureExt;
    recover_volume_weighted_nodal_tensor_with_capacity::<std::convert::Infallible>(
        elements,
        integration_point_stress,
        node_count,
        None,
    )
    .now_or_never()
    .expect("native nodal recovery capacity is immediately ready")
    .map_err(|error| match error {
        NumericalCapacityError::Numerical(error) => error,
        NumericalCapacityError::Admission(never) => match never {},
    })
}

pub(super) async fn recover_volume_weighted_nodal_tensor_with_capacity<A>(
    elements: &[SolidRecoveryElementSummary],
    integration_point_stress: &[f64],
    node_count: usize,
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<Vec<f64>, NumericalCapacityError<StructuralFieldRecoveryError, A>> {
    const POINTS: usize = 4;
    const COMPONENTS: usize = 6;
    if integration_point_stress.len() != elements.len() * POINTS * COMPONENTS {
        return Err(StructuralFieldRecoveryError::ElementValueCountMismatch {
            expected: elements.len() * POINTS * COMPONENTS,
            actual: integration_point_stress.len(),
        }
        .into());
    }
    // Per node, per region: volume-weighted tensor sums. Most nodes touch one region.
    let mut sums: Vec<Vec<(&str, [f64; COMPONENTS], f64)>> = vec![Vec::new(); node_count];
    for (element, samples) in elements
        .iter()
        .zip(integration_point_stress.chunks_exact(POINTS * COMPONENTS))
    {
        let volume = element
            .geometry
            .volume_m3_with_capacity(owner)
            .await
            .map_err(|error| {
                error.map_numerical(|error| {
                    StructuralFieldRecoveryError::geometry(&element.element_id, error)
                })
            })?;
        let nodes = element.geometry.node_indices().len();
        // Six component multiply-adds and one volume sum at each node; Tet10
        // also extrapolates four samples to four corners and six midsides.
        let operations = nodes * 13 + if nodes == 10 { 217 } else { 0 };
        measured(owner, UsageCategory::Solve, count(operations)?, || {
            let points: [[f64; COMPONENTS]; POINTS] = std::array::from_fn(|point| {
                std::array::from_fn(|component| samples[point * COMPONENTS + component])
            });
            let node_indices = element.geometry.node_indices();
            let nodal: Vec<[f64; COMPONENTS]> = if node_indices.len() == 10 {
                crate::assembly::elements::solid::tetrahedron10::extrapolate_quadrature_to_nodes(
                    points,
                )
                .to_vec()
            } else {
                // Constant-strain element: every sample carries the element's stress.
                vec![points[0]; node_indices.len()]
            };
            for (&node_index, tensor) in node_indices.iter().zip(nodal) {
                let entries = sums.get_mut(node_index).ok_or_else(|| {
                    StructuralFieldRecoveryError::InvalidNodeIndex {
                        element_id: element.element_id.clone(),
                        node_index,
                    }
                })?;
                let slot = match entries
                    .iter()
                    .position(|(region, _, _)| *region == element.region_id.as_str())
                {
                    Some(slot) => slot,
                    None => {
                        entries.push((element.region_id.as_str(), [0.0; COMPONENTS], 0.0));
                        entries.len() - 1
                    }
                };
                let (_, sum, weight) = &mut entries[slot];
                for component in 0..COMPONENTS {
                    sum[component] += tensor[component] * volume;
                }
                *weight += volume;
            }
            Ok::<_, StructuralFieldRecoveryError>(())
        })
        .await?;
    }
    let mut nodal = vec![0.0_f64; node_count * COMPONENTS];
    for (node, entries) in sums.into_iter().enumerate() {
        let regions = entries
            .iter()
            .filter(|(_, _, weight)| *weight > 0.0)
            .count();
        let comparisons = count(regions.saturating_sub(1))?
            .checked_mul(34)
            .map_err(LedgerError::from)
            .map_err(StructuralFieldRecoveryError::from)?;
        let operations = count(regions)?
            .checked_mul(6)
            .and_then(|work| work.checked_add(comparisons))
            .map_err(LedgerError::from)
            .map_err(StructuralFieldRecoveryError::from)?;
        measured(owner, UsageCategory::Solve, operations, || {
            let averaged = entries
                .into_iter()
                .filter(|(_, _, weight)| *weight > 0.0)
                .map(|(_, sum, weight)| sum.map(|value| value / weight))
                .max_by(|left, right| {
                    von_mises(left)
                        .partial_cmp(&von_mises(right))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(tensor) = averaged {
                nodal[node * COMPONENTS..(node + 1) * COMPONENTS].copy_from_slice(&tensor);
            }
            Ok::<_, StructuralFieldRecoveryError>(())
        })
        .await?;
    }
    Ok(nodal)
}

fn count(length: usize) -> Result<OperationCount, StructuralFieldRecoveryError> {
    Ok(OperationCount::from_operations(
        u64::try_from(length).map_err(|_| LedgerError::CountOverflow)?,
    ))
}

fn von_mises(tensor: &[f64; 6]) -> f64 {
    let [sxx, syy, szz, txy, tyz, txz] = *tensor;
    (0.5 * ((sxx - syy).powi(2) + (syy - szz).powi(2) + (szz - sxx).powi(2))
        + 3.0 * (txy.powi(2) + tyz.powi(2) + txz.powi(2)))
    .sqrt()
}

#[cfg(test)]
mod tests {
    use crate::assembly::{
        elements::solid::SolidMaterial, SolidRecoveryElementSummary, SolidRecoveryGeometry,
    };

    use super::recover_volume_weighted_nodal_tensor;

    #[test]
    fn adjacent_tetrahedra_use_physical_volume_weights() {
        let material = SolidMaterial {
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(1.0),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(0.25),
            density_kg_per_m3: runmat_types::physical::MassDensity::new(1.0),
        };
        let elements = vec![
            SolidRecoveryElementSummary {
                element_id: "small".into(),
                region_id: "body".into(),
                material,
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
            },
            SolidRecoveryElementSummary {
                element_id: "large".into(),
                region_id: "body".into(),
                material,
                thermal_eigenstrain: None,
                prescribed_temperature_k: None,
                solved_thermal_quadrature: None,
                geometry: SolidRecoveryGeometry::Tetrahedron4 {
                    node_indices: [0, 4, 5, 6],
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new([
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 2.0],
                    ]),
                },
            },
        ];

        // Constant-stress Tet4 elements: sigma_xx = 10 and 20 at every sample.
        let samples = [10.0, 20.0]
            .iter()
            .flat_map(|value| (0..4).flat_map(move |_| [*value, 0.0, 0.0, 0.0, 0.0, 0.0]))
            .collect::<Vec<_>>();
        let recovered = recover_volume_weighted_nodal_tensor(&elements, &samples, 7)
            .expect("valid adjacent tetrahedra should recover");
        let sxx = |node: usize| recovered[node * 6];

        assert!((sxx(0) - 170.0 / 9.0).abs() < 1.0e-12);
        assert_eq!([sxx(1), sxx(2), sxx(3)], [10.0, 10.0, 10.0]);
        assert_eq!([sxx(4), sxx(5), sxx(6)], [20.0, 20.0, 20.0]);
    }

    // witness: FEA-STRUCTURAL-REFERENCE-001/patch_reference_convergence @
    //   linear_structural/tet10/numerical_kernel; coverage: partial;
    //   check: tet10-stress-recovery-patch; boundary: nodal stress recovery reproduces a linear stress field exactly at every corner
    //   and midside node of two Tet10 elements sharing a face (patch test), and does not blend
    //   two regions at their interface node. Curved Tet10 geometry is not exercised.
    #[test]
    fn tet10_nodal_stress_is_exact_for_a_linear_field_and_keeps_regions_apart() {
        use crate::assembly::elements::solid::tetrahedron10::interpolate_nodal_samples;
        let material = SolidMaterial {
            youngs_modulus_pa: runmat_types::physical::Quantity::new(1.0),
            poisson_ratio: runmat_types::physical::Quantity::new(0.25),
            density_kg_per_m3: runmat_types::physical::Quantity::new(1.0),
        };
        let field = |x: [f64; 3]| -> [f64; 6] {
            [
                3.0 + 2.0 * x[0] - x[2],
                -1.0 + 0.5 * x[1],
                4.0 * x[0] + x[1] + x[2],
                0.25 - x[1],
                1.5 * x[2],
                -2.0 + x[0],
            ]
        };
        let corners_a = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let corners_b = [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ];
        let tet10 = |corners: [[f64; 3]; 4]| -> [[f64; 3]; 10] {
            let edges = runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;
            std::array::from_fn(|node| {
                if node < 4 {
                    corners[node]
                } else {
                    let [a, b] = edges[node - 4];
                    std::array::from_fn(|axis| 0.5 * (corners[a][axis] + corners[b][axis]))
                }
            })
        };
        let coordinates = [tet10(corners_a), tet10(corners_b)];
        // Shared face nodes 1, 2, 3 and the midsides between them carry one global index.
        let edges = runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;
        let mut indices = [[0usize; 10]; 2];
        let mut positions: Vec<[f64; 3]> = Vec::new();
        for (element, nodes) in coordinates.iter().enumerate() {
            for (local, position) in nodes.iter().enumerate() {
                let found = positions.iter().position(|existing| existing == position);
                indices[element][local] = found.unwrap_or_else(|| {
                    positions.push(*position);
                    positions.len() - 1
                });
            }
        }
        let _ = edges;
        let element = |id: &str, region: &str, element: usize| SolidRecoveryElementSummary {
            element_id: id.into(),
            region_id: region.into(),
            material,
            thermal_eigenstrain: None,
            prescribed_temperature_k: None,
            solved_thermal_quadrature: None,
            geometry: SolidRecoveryGeometry::Tetrahedron10 {
                node_indices: indices[element],
                coordinates_m: runmat_types::physical::Quantity::new(coordinates[element]),
            },
        };
        // Samples at the canonical four points of each element, from the linear field.
        let samples_for = |element: usize, scale: f64| -> Vec<f64> {
            crate::topology::quadrature::TETRAHEDRON_FOUR_POINT_RULE
                .iter()
                .flat_map(|barycentric| {
                    let position: [f64; 3] = std::array::from_fn(|axis| {
                        (0..4)
                            .map(|corner| barycentric[corner] * coordinates[element][corner][axis])
                            .sum()
                    });
                    field(position).map(|value| value * scale)
                })
                .collect()
        };
        let same_region = [element("a", "body", 0), element("b", "body", 1)];
        let samples = [samples_for(0, 1.0), samples_for(1, 1.0)].concat();
        let recovered =
            recover_volume_weighted_nodal_tensor(&same_region, &samples, positions.len()).unwrap();
        for (node, position) in positions.iter().enumerate() {
            let expected = field(*position);
            for component in 0..6 {
                let value = recovered[node * 6 + component];
                assert!(
                    (value - expected[component]).abs()
                        <= 1e-12 * (1.0 + expected[component].abs()),
                    "node {node} component {component}: {value} vs {}",
                    expected[component]
                );
            }
        }
        // Two regions: the doubled field in region "outer" governs the shared nodes; it is not
        // averaged with the inner region's values.
        let two_regions = [element("a", "inner", 0), element("b", "outer", 1)];
        let samples = [samples_for(0, 1.0), samples_for(1, 2.0)].concat();
        let recovered =
            recover_volume_weighted_nodal_tensor(&two_regions, &samples, positions.len()).unwrap();
        let shared = indices[0][1];
        let expected = field(positions[shared]).map(|value| 2.0 * value);
        for component in 0..6 {
            assert!((recovered[shared * 6 + component] - expected[component]).abs() < 1e-12);
        }
        let _ = interpolate_nodal_samples;
    }
}
