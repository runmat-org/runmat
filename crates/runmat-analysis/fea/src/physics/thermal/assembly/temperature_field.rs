use crate::{
    excitation::{RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations},
    topology::AnalysisProjection,
};

use super::ThermalAssemblyError;

pub(super) fn prescribed_temperatures(
    projection: &AnalysisProjection,
    excitations: &RealizedExcitations,
) -> Result<Vec<Option<f64>>, ThermalAssemblyError> {
    // Ownership remains in the realized inventory; only error diagnostics copy identifiers.
    let mut whole_model = None::<(f64, &str)>;
    let mut regional = vec![None::<(f64, &str)>; projection.nodes.len()];

    for excitation in excitations.iter() {
        let RealizedExcitationKind::TemperatureField(value) = excitation.kind else {
            continue;
        };
        if value.imaginary != 0.0 {
            return Err(ThermalAssemblyError::ComplexTemperatureField {
                excitation_id: excitation.excitation_id.0.clone(),
            });
        }

        match &excitation.target {
            RealizedExcitationTarget::WholeModel => {
                if let Some((existing, first_excitation_id)) = &whole_model {
                    if *existing != value.real {
                        return Err(ThermalAssemblyError::ConflictingTemperatureFields {
                            first_excitation_id: (*first_excitation_id).to_owned(),
                            second_excitation_id: excitation.excitation_id.0.clone(),
                            location: "the whole model".to_string(),
                        });
                    }
                }
                whole_model = Some((value.real, excitation.excitation_id.0.as_str()));
            }
            RealizedExcitationTarget::Region(target) => {
                let mut selected = vec![false; projection.nodes.len()];
                for element in &projection.elements {
                    if target.selects_volume_element(&element.region_id, &element.provenance) {
                        for &node_index in &element.node_indices {
                            selected[node_index] = true;
                        }
                    }
                }
                for face in projection.boundary_faces.iter().filter(|face| {
                    face.provenance
                        .iter()
                        .any(|entity| target.matches_persistent_entity(entity))
                }) {
                    for &node_index in &face.node_indices {
                        selected[node_index] = true;
                    }
                }
                if !selected.iter().any(|selected| *selected) {
                    return Err(ThermalAssemblyError::UnmappedTemperatureField {
                        excitation_id: excitation.excitation_id.0.clone(),
                    });
                }
                for (node_index, selected) in selected.into_iter().enumerate() {
                    if !selected {
                        continue;
                    }
                    if let Some((existing, first_excitation_id)) = &regional[node_index] {
                        if *existing != value.real {
                            return Err(ThermalAssemblyError::ConflictingTemperatureFields {
                                first_excitation_id: (*first_excitation_id).to_owned(),
                                second_excitation_id: excitation.excitation_id.0.clone(),
                                location: format!(
                                    "solver node {}",
                                    projection.nodes[node_index].node_id
                                ),
                            });
                        }
                    }
                    regional[node_index] = Some((value.real, excitation.excitation_id.0.as_str()));
                }
            }
            _ => {
                return Err(ThermalAssemblyError::UnmappedTemperatureField {
                    excitation_id: excitation.excitation_id.0.clone(),
                });
            }
        }
    }

    Ok(regional
        .into_iter()
        .map(|regional| {
            regional
                .map(|(value, _)| value)
                .or_else(|| whole_model.as_ref().map(|(value, _)| *value))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        CoordinateFrame, ExcitationCaseId, ExcitationDistribution, ExcitationId,
        ExcitationKinematics, PhysicalQuantityKind, ProjectionRequirement, RegionTarget, SignalId,
    };
    use runmat_meshing_core::ElementOrder;

    use crate::excitation::{ComplexScalar, RealizedExcitation, RealizedExcitationPolicy};

    use super::*;

    fn temperature(id: &str, target: RealizedExcitationTarget, kelvin: f64) -> RealizedExcitation {
        RealizedExcitation {
            excitation_id: ExcitationId(id.to_string()),
            case_ids: vec![ExcitationCaseId(format!("{id}.case"))],
            signal_id: SignalId(format!("{id}.signal")),
            quantity: PhysicalQuantityKind::Temperature,
            sample: crate::excitation::ProjectionSample::Static,
            target,
            kind: RealizedExcitationKind::TemperatureField(ComplexScalar::real(kelvin)),
            policy: RealizedExcitationPolicy {
                frame: CoordinateFrame::Global,
                distribution: ExcitationDistribution::TopologyField,
                kinematics: ExcitationKinematics::FixedFrame,
                projection: ProjectionRequirement {
                    preserve_resultant: false,
                    preserve_moment: false,
                    relative_tolerance: 1.0e-12,
                },
            },
        }
    }

    #[test]
    fn regional_temperature_field_overrides_whole_model_temperature() {
        let mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let excitations = RealizedExcitations::new(vec![
            temperature("whole", RealizedExcitationTarget::WholeModel, 300.0),
            temperature(
                "face",
                RealizedExcitationTarget::Region(RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("face:0"),
                )),
                350.0,
            ),
        ]);

        let prescribed = prescribed_temperatures(&projection, &excitations).unwrap();

        assert_eq!(
            prescribed
                .iter()
                .filter(|value| **value == Some(350.0))
                .count(),
            3
        );
        assert_eq!(
            prescribed
                .iter()
                .filter(|value| **value == Some(300.0))
                .count(),
            1
        );
    }

    #[test]
    fn overlapping_regional_temperature_fields_fail_closed() {
        let mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let excitations = RealizedExcitations::new(vec![
            temperature(
                "first",
                RealizedExcitationTarget::Region(RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("face:0"),
                )),
                300.0,
            ),
            temperature(
                "second",
                RealizedExcitationTarget::Region(RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("face:1"),
                )),
                350.0,
            ),
        ]);

        assert!(matches!(
            prescribed_temperatures(&projection, &excitations),
            Err(ThermalAssemblyError::ConflictingTemperatureFields {
                first_excitation_id, second_excitation_id, location,
            }) if first_excitation_id == "first" && second_excitation_id == "second"
                && location.starts_with("solver node ")
        ));
    }

    #[test]
    fn long_identifiers_and_equal_overlaps_preserve_temperature_values() {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
            let projection = AnalysisProjection::derive(&mesh).unwrap();
            let id = "temperature_owner".repeat(2048);
            let excitations = RealizedExcitations::new(vec![
                temperature("whole", RealizedExcitationTarget::WholeModel, 300.0),
                temperature(
                    &id,
                    RealizedExcitationTarget::Region(RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_face("face:0"),
                    )),
                    350.0,
                ),
                temperature(
                    "same",
                    RealizedExcitationTarget::Region(RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_face("face:0"),
                    )),
                    350.0,
                ),
            ]);
            let prescribed = prescribed_temperatures(&projection, &excitations).unwrap();
            let face_selector = runmat_meshing_core::fixtures::fixture_face("face:0");
            let face = projection
                .boundary_faces_matching(&face_selector)
                .next()
                .unwrap();
            for (index, value) in prescribed.iter().enumerate() {
                assert_eq!(
                    *value,
                    Some(if face.node_indices.contains(&index) {
                        350.0
                    } else {
                        300.0
                    })
                );
            }
        }
    }
}
