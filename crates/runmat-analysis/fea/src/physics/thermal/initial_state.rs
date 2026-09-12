use runmat_analysis_core::{
    AnalysisModel, EntityDimension, ExpectedTopology, InitialCondition, InitialConditionTarget,
    InitialValue, PhysicalSignal, SignalRepresentation, TabulatedSignalDomain,
};
use thiserror::Error;

use crate::{
    contracts::ThermalInitialStateEvidence,
    excitation::{expect_scalar, sample_signal, ExcitationSamplePoint},
    topology::AnalysisProjection,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RealizedThermalInitialState {
    pub temperature_k: Vec<f64>,
    pub evidence: ThermalInitialStateEvidence,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub(crate) enum ThermalInitialStateError {
    #[error("thermal solve cannot resolve its procedure step")]
    MissingStep,
    #[error("thermal initial condition {initial_condition_id} must target volume elements")]
    InvalidTarget { initial_condition_id: String },
    #[error(
        "thermal initial condition {initial_condition_id} does not resolve any volume elements"
    )]
    UnmappedRegion { initial_condition_id: String },
    #[error(
        "thermal initial condition {initial_condition_id} has temperature below absolute zero"
    )]
    InvalidTemperature { initial_condition_id: String },
    #[error("thermal initial condition {initial_condition_id} conflicts at solver node {node_id}")]
    ConflictingTemperature {
        initial_condition_id: String,
        node_id: u64,
    },
    #[error(
        "thermal initial temperature {initial_temperature_k} K conflicts with prescribed temperature {prescribed_temperature_k} K at solver node {node_id}"
    )]
    PrescribedTemperatureConflict {
        node_id: u64,
        initial_temperature_k: f64,
        prescribed_temperature_k: f64,
    },
    #[error("thermal initial condition {initial_condition_id} cannot sample its signal: {detail}")]
    SignalSample {
        initial_condition_id: String,
        detail: String,
    },
    #[error("thermal initial condition {initial_condition_id} references a field artifact that has not been materialized into the solve request")]
    UnmaterializedField { initial_condition_id: String },
}

pub(crate) fn realize(
    model: &AnalysisModel,
    projection: &AnalysisProjection,
    procedure_step_id: &str,
    reference_temperature_k: f64,
    prescribed_temperature_k: &[Option<f64>],
) -> Result<RealizedThermalInitialState, ThermalInitialStateError> {
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id.as_str() == procedure_step_id)
        .ok_or(ThermalInitialStateError::MissingStep)?;

    let mut assigned = vec![None; projection.nodes.len()];
    let mut applied_condition_ids = Vec::new();
    for initial_condition in &model.initial_conditions {
        if !initial_condition.applies_to_step(step) {
            continue;
        }
        let InitialCondition::Temperature(condition) = initial_condition else {
            continue;
        };
        let value = realize_value(model, condition)?;
        if value < 0.0 {
            return Err(ThermalInitialStateError::InvalidTemperature {
                initial_condition_id: condition.initial_condition_id.clone(),
            });
        }
        let nodes = target_nodes(projection, condition)?;
        for node in nodes {
            if assigned[node].is_some_and(|existing| existing != value) {
                return Err(ThermalInitialStateError::ConflictingTemperature {
                    initial_condition_id: condition.initial_condition_id.clone(),
                    node_id: projection.nodes[node].node_id,
                });
            }
            assigned[node] = Some(value);
        }
        applied_condition_ids.push(condition.initial_condition_id.clone());
    }

    let authored_node_count = assigned.iter().filter(|value| value.is_some()).count();
    let prescribed_node_count = prescribed_temperature_k
        .iter()
        .filter(|value| value.is_some())
        .count();
    let mut authored_and_prescribed_node_count = 0;
    let mut reference_node_count = 0;
    let mut temperature_k = Vec::with_capacity(projection.nodes.len());
    for (node_index, authored) in assigned.into_iter().enumerate() {
        let prescribed = prescribed_temperature_k.get(node_index).copied().flatten();
        match (authored, prescribed) {
            (Some(initial), Some(boundary)) if initial != boundary => {
                return Err(ThermalInitialStateError::PrescribedTemperatureConflict {
                    node_id: projection.nodes[node_index].node_id,
                    initial_temperature_k: initial,
                    prescribed_temperature_k: boundary,
                });
            }
            (Some(initial), Some(_)) => {
                authored_and_prescribed_node_count += 1;
                temperature_k.push(initial);
            }
            (Some(initial), None) => temperature_k.push(initial),
            (None, Some(boundary)) => temperature_k.push(boundary),
            (None, None) => {
                reference_node_count += 1;
                temperature_k.push(reference_temperature_k);
            }
        }
    }
    let minimum_temperature_k = temperature_k.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum_temperature_k = temperature_k
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(RealizedThermalInitialState {
        evidence: ThermalInitialStateEvidence {
            procedure_step_id: step.step_id.0.clone(),
            reference_temperature_k,
            applied_initial_condition_ids: applied_condition_ids,
            authored_node_count,
            prescribed_node_count,
            authored_and_prescribed_node_count,
            reference_node_count,
            minimum_temperature_k,
            maximum_temperature_k,
        },
        temperature_k,
    })
}

fn realize_value(
    model: &AnalysisModel,
    condition: &runmat_analysis_core::InitialConditionValue<runmat_analysis_core::Temperature>,
) -> Result<f64, ThermalInitialStateError> {
    match &condition.value {
        InitialValue::Uniform(value) => Ok(value.kelvin),
        InitialValue::SignalSample {
            signal_id,
            abscissa,
            approach,
        } => {
            let signal = model
                .signals
                .iter()
                .find(|signal| signal.signal_id() == signal_id)
                .ok_or_else(|| ThermalInitialStateError::SignalSample {
                    initial_condition_id: condition.initial_condition_id.clone(),
                    detail: "signal does not exist".into(),
                })?;
            let PhysicalSignal::Temperature(signal) = signal else {
                return Err(ThermalInitialStateError::SignalSample {
                    initial_condition_id: condition.initial_condition_id.clone(),
                    detail: "signal is not a temperature".into(),
                });
            };
            let sample = match &signal.representation {
                SignalRepresentation::Tabulated(table) => match &table.domain {
                    TabulatedSignalDomain::TimeSeconds => ExcitationSamplePoint::TimeSeconds {
                        value: *abscissa,
                        approach: *approach,
                    },
                    TabulatedSignalDomain::StudyParameter { parameter_id } => {
                        ExcitationSamplePoint::StudyParameter {
                            parameter_id,
                            canonical_value: *abscissa,
                            approach: *approach,
                        }
                    }
                    TabulatedSignalDomain::FrequencyHertz => {
                        return Err(ThermalInitialStateError::SignalSample {
                            initial_condition_id: condition.initial_condition_id.clone(),
                            detail: "frequency signals cannot define an initial state".into(),
                        });
                    }
                },
                _ => ExcitationSamplePoint::TimeSeconds {
                    value: *abscissa,
                    approach: *approach,
                },
            };
            let value = sample_signal(signal, sample, |value| value.kelvin)
                .and_then(expect_scalar)
                .map_err(|error| ThermalInitialStateError::SignalSample {
                    initial_condition_id: condition.initial_condition_id.clone(),
                    detail: error.to_string(),
                })?;
            if value.imaginary != 0.0 {
                return Err(ThermalInitialStateError::SignalSample {
                    initial_condition_id: condition.initial_condition_id.clone(),
                    detail: "initial temperature cannot be complex".into(),
                });
            }
            Ok(value.real)
        }
        InitialValue::FieldArtifact(_) => Err(ThermalInitialStateError::UnmaterializedField {
            initial_condition_id: condition.initial_condition_id.clone(),
        }),
        InitialValue::UniformVector3(_) => {
            unreachable!("core validation rejects vector temperature initial conditions")
        }
    }
}

fn target_nodes(
    projection: &AnalysisProjection,
    condition: &runmat_analysis_core::InitialConditionValue<runmat_analysis_core::Temperature>,
) -> Result<Vec<usize>, ThermalInitialStateError> {
    match &condition.target {
        InitialConditionTarget::WholeModel => Ok((0..projection.nodes.len()).collect()),
        InitialConditionTarget::Region(target) => {
            if target.entity_dimension != EntityDimension::Volume
                || target.expected_topology != ExpectedTopology::MeshElement
            {
                return Err(ThermalInitialStateError::InvalidTarget {
                    initial_condition_id: condition.initial_condition_id.clone(),
                });
            }
            let mut selected = vec![false; projection.nodes.len()];
            let mut matched = false;
            for element in &projection.elements {
                if element
                    .region_id
                    .matches_analysis_selection(&target.region_id)
                {
                    matched = true;
                    for &node in &element.node_indices {
                        selected[node] = true;
                    }
                }
            }
            if !matched {
                return Err(ThermalInitialStateError::UnmappedRegion {
                    initial_condition_id: condition.initial_condition_id.clone(),
                });
            }
            Ok(selected
                .iter()
                .enumerate()
                .filter_map(|(node, selected)| selected.then_some(node))
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AnalysisSequence, CoordinateFrame, InitialConditionValue, InitialFieldArtifactRef,
        ProcedureStep, RegionTarget, Temperature, ThermalDomain,
    };
    use runmat_meshing_core::ElementOrder;

    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};

    fn thermal_model() -> (
        AnalysisModel,
        runmat_meshing_core::SolverMeshArtifact,
        AnalysisProjection,
    ) {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.thermal = Some(ThermalDomain {
            reference_temperature: Temperature::new(293.15),
        });
        model.sequence = AnalysisSequence::from_steps(
            "thermal_sequence",
            vec![ProcedureStep::thermal_transient("thermal_step")],
        );
        let mesh =
            runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        (model, mesh, projection)
    }

    fn uniform_condition(
        id: &str,
        target: InitialConditionTarget,
        kelvin: f64,
    ) -> InitialCondition {
        InitialCondition::Temperature(InitialConditionValue {
            initial_condition_id: id.into(),
            target,
            frame: CoordinateFrame::Global,
            value: InitialValue::Uniform(Temperature::new(kelvin)),
            applies_before_step_id: "thermal_step".into(),
        })
    }

    #[test]
    fn volume_initial_temperature_maps_to_canonical_nodes() {
        let (mut model, _mesh, projection) = thermal_model();
        let selector = projection.elements[0]
            .region_id
            .analysis_selection_id()
            .unwrap();
        model.initial_conditions = vec![uniform_condition(
            "heated_volume",
            InitialConditionTarget::Region(RegionTarget::volume(selector)),
            325.0,
        )];

        let state = realize(
            &model,
            &projection,
            "thermal_step",
            293.15,
            &vec![None; projection.nodes.len()],
        )
        .unwrap();
        assert!(state.temperature_k.iter().all(|value| *value == 325.0));
        assert_eq!(state.evidence.authored_node_count, projection.nodes.len());
        assert_eq!(state.evidence.reference_node_count, 0);
        assert_eq!(
            state.evidence.applied_initial_condition_ids,
            ["heated_volume"]
        );
    }

    #[test]
    fn conflicting_initial_temperatures_fail_at_the_shared_node() {
        let (mut model, _mesh, projection) = thermal_model();
        model.initial_conditions = vec![
            uniform_condition("whole", InitialConditionTarget::WholeModel, 300.0),
            uniform_condition("other", InitialConditionTarget::WholeModel, 310.0),
        ];

        assert!(matches!(
            realize(
                &model,
                &projection,
                "thermal_step",
                293.15,
                &vec![None; projection.nodes.len()],
            ),
            Err(ThermalInitialStateError::ConflictingTemperature { .. })
        ));
    }

    #[test]
    fn unmaterialized_initial_field_fails_closed() {
        let (mut model, _mesh, projection) = thermal_model();
        model.initial_conditions = vec![InitialCondition::Temperature(InitialConditionValue {
            initial_condition_id: "imported_temperature".into(),
            target: InitialConditionTarget::WholeModel,
            frame: CoordinateFrame::Global,
            value: InitialValue::FieldArtifact(InitialFieldArtifactRef {
                artifact_id: "artifact".into(),
                content_digest: "sha256:content".into(),
                topology_digest: "sha256:topology".into(),
                field_id: "temperature".into(),
            }),
            applies_before_step_id: "thermal_step".into(),
        })];

        assert!(matches!(
            realize(
                &model,
                &projection,
                "thermal_step",
                293.15,
                &vec![None; projection.nodes.len()],
            ),
            Err(ThermalInitialStateError::UnmaterializedField { .. })
        ));
    }

    #[test]
    fn prescribed_temperature_is_part_of_the_initial_state() {
        let (model, _mesh, projection) = thermal_model();
        let mut prescribed = vec![None; projection.nodes.len()];
        prescribed[0] = Some(310.0);

        let state = realize(&model, &projection, "thermal_step", 293.15, &prescribed).unwrap();

        assert_eq!(state.temperature_k[0], 310.0);
        assert_eq!(state.evidence.authored_node_count, 0);
        assert_eq!(state.evidence.prescribed_node_count, 1);
        assert_eq!(state.evidence.authored_and_prescribed_node_count, 0);
        assert_eq!(state.evidence.reference_node_count, 3);
    }

    #[test]
    fn conflicting_authored_and_prescribed_initial_temperatures_fail() {
        let (mut model, _mesh, projection) = thermal_model();
        model.initial_conditions = vec![uniform_condition(
            "authored_start",
            InitialConditionTarget::WholeModel,
            300.0,
        )];
        let prescribed = vec![Some(310.0); projection.nodes.len()];

        assert!(matches!(
            realize(&model, &projection, "thermal_step", 293.15, &prescribed,),
            Err(ThermalInitialStateError::PrescribedTemperatureConflict { .. })
        ));
    }
}
