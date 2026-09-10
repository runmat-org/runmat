use crate::{
    excitation::{RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations},
    topology::AnalysisProjection,
};

use super::ThermalAssemblyError;

pub(super) fn prescribed_temperatures(
    projection: &AnalysisProjection,
    excitations: &RealizedExcitations,
) -> Result<Vec<Option<f64>>, ThermalAssemblyError> {
    let mut whole_model = None::<(f64, String)>;
    let mut regional = vec![None::<(f64, String)>; projection.nodes.len()];

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
                            first_excitation_id: first_excitation_id.clone(),
                            second_excitation_id: excitation.excitation_id.0.clone(),
                            location: "the whole model".to_string(),
                        });
                    }
                }
                whole_model = Some((value.real, excitation.excitation_id.0.clone()));
            }
            RealizedExcitationTarget::Region(region_id) => {
                let mut selected = vec![false; projection.nodes.len()];
                for element in &projection.elements {
                    if element.region_id.matches_analysis_selection(region_id) {
                        for &node_index in &element.node_indices {
                            selected[node_index] = true;
                        }
                    }
                }
                for face in projection.boundary_faces_matching(region_id) {
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
                                first_excitation_id: first_excitation_id.clone(),
                                second_excitation_id: excitation.excitation_id.0.clone(),
                                location: format!(
                                    "solver node {}",
                                    projection.nodes[node_index].node_id
                                ),
                            });
                        }
                    }
                    regional[node_index] = Some((value.real, excitation.excitation_id.0.clone()));
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
        ExcitationKinematics, PhysicalQuantityKind, ProjectionRequirement, SignalId,
    };
    use runmat_meshing_core::ElementOrder;

    use crate::excitation::{ComplexScalar, RealizedExcitation, RealizedExcitationPolicy};

    use super::*;

    fn temperature(id: &str, target: RealizedExcitationTarget, kelvin: f64) -> RealizedExcitation {
        RealizedExcitation {
            excitation_id: ExcitationId(id.to_string()),
            case_id: ExcitationCaseId(format!("{id}.case")),
            signal_id: SignalId(format!("{id}.signal")),
            quantity: PhysicalQuantityKind::Temperature,
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
                RealizedExcitationTarget::Region("face:0".to_string()),
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
                RealizedExcitationTarget::Region("face:0".to_string()),
                300.0,
            ),
            temperature(
                "second",
                RealizedExcitationTarget::Region("face:1".to_string()),
                350.0,
            ),
        ]);

        assert!(matches!(
            prescribed_temperatures(&projection, &excitations),
            Err(ThermalAssemblyError::ConflictingTemperatureFields { .. })
        ));
    }
}
