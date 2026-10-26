use runmat_analysis_core::{AnalysisModel, InitialCondition, InitialValue, PhysicalSignal};
use runmat_canonical_codec::{CanonicalCodecError, SerializedCanonicalProfile};
use serde::{ser::SerializeSeq, Serialize, Serializer};
use thiserror::Error;

pub(crate) mod fields;

use crate::{
    contracts::ThermalInitialStateEvidence,
    excitation::{expect_scalar, sample_signal},
    topology::AnalysisProjection,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RealizedThermalInitialState {
    pub temperature_k:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Temperature, Vec<f64>>,
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
        "thermal initial temperature {initial_temperature_k} K conflicts with prescribed temperature {} K at solver node {node_id}", .prescribed_temperature_k.canonical_value()
    )]
    PrescribedTemperatureConflict {
        node_id: u64,
        initial_temperature_k: f64,
        prescribed_temperature_k:
            runmat_types::physical::Quantity<runmat_types::physical::kind::Temperature, f64>,
    },
    #[error("thermal initial condition {initial_condition_id} cannot sample its signal: {detail}")]
    SignalSample {
        initial_condition_id: String,
        detail: String,
    },
    #[error("thermal initial field admission failed: {detail}")]
    FieldAdmission { detail: String },
}

fn selected_conditions<'a>(
    model: &'a AnalysisModel,
    step: &'a runmat_analysis_core::ProcedureStep,
) -> impl Iterator<
    Item = &'a runmat_analysis_core::InitialConditionValue<runmat_analysis_core::Temperature>,
> + Clone {
    model
        .initial_conditions
        .iter()
        .filter_map(move |condition| {
            if !condition.applies_to_step(step) {
                return None;
            }
            match condition {
                InitialCondition::Temperature(value) => Some(value),
                _ => None,
            }
        })
}

pub(crate) fn selected_condition_count(
    model: &AnalysisModel,
    step: &runmat_analysis_core::ProcedureStep,
) -> usize {
    selected_conditions(model, step).count()
}

/// Profiles the producer's selected IDs without retaining copies or realizing temperature arrays.
/// The shared applies_to_step selection may allocate a temporary object-reference ID.
pub(crate) fn evidence_profile(
    model: &AnalysisModel,
    step: &runmat_analysis_core::ProcedureStep,
) -> Result<SerializedCanonicalProfile, CanonicalCodecError> {
    struct SelectedIds<'a> {
        model: &'a AnalysisModel,
        step: &'a runmat_analysis_core::ProcedureStep,
    }
    impl Serialize for SelectedIds<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let selected = selected_conditions(self.model, self.step);
            let mut sequence = serializer.serialize_seq(Some(selected.clone().count()))?;
            for condition in selected {
                sequence.serialize_element(condition.initial_condition_id.as_str())?;
            }
            sequence.end()
        }
    }
    ThermalInitialStateEvidence::projected_profile(
        step.step_id.as_str(),
        &SelectedIds { model, step },
        crate::execution::THERMAL_WORKLOAD_LIMITS,
    )
}

pub(crate) fn realize(
    model: &AnalysisModel,
    projection: &AnalysisProjection,
    procedure_step_id: &str,
    reference_temperature_k: f64,
    prescribed_temperature_k: &[Option<f64>],
    fields: &[crate::execution::InitialFieldBinding],
) -> Result<RealizedThermalInitialState, ThermalInitialStateError> {
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id.as_str() == procedure_step_id)
        .ok_or(ThermalInitialStateError::MissingStep)?;

    let mut assigned = vec![None; projection.nodes.len()];
    let mut applied_condition_ids = Vec::new();
    // Authored fields hold one value per mesh node. The second-side node of a conductance
    // interface (appended after the mesh nodes, with its mesh node's id) takes its node's value.
    let mesh_node = |node: usize| {
        projection
            .node_index(projection.nodes[node].node_id)
            .unwrap_or(node)
    };
    let mesh_node_count = (0..projection.nodes.len())
        .filter(|&node| mesh_node(node) == node)
        .count();
    for condition in selected_conditions(model, step) {
        let field = fields::values(
            model,
            condition,
            runmat_execution::Digest::from_bytes(*projection.mesh_digest.bytes()),
            mesh_node_count,
            fields,
        )
        .map_err(|error| ThermalInitialStateError::FieldAdmission {
            detail: error.to_string(),
        })?;
        let uniform = if field.is_none() {
            Some(realize_value(model, condition)?)
        } else {
            None
        };
        if uniform.is_some_and(|value| value < 0.0) {
            return Err(ThermalInitialStateError::InvalidTemperature {
                initial_condition_id: condition.initial_condition_id.clone(),
            });
        }
        let nodes = target_nodes(projection, condition)?;
        for node in nodes {
            let value = match (field, uniform) {
                (Some(values), _) => values[mesh_node(node)],
                (_, Some(value)) => value,
                _ => unreachable!("admitted initial condition has a field or uniform value"),
            };
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

    // Evidence counts mesh nodes; a second-side interface node is part of its mesh node.
    let is_mesh_node = |node: usize| mesh_node(node) == node;
    let authored_node_count = assigned
        .iter()
        .enumerate()
        .filter(|(node, value)| is_mesh_node(*node) && value.is_some())
        .count();
    let prescribed_node_count = prescribed_temperature_k
        .iter()
        .enumerate()
        .filter(|(node, value)| is_mesh_node(*node) && value.is_some())
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
                    prescribed_temperature_k: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Temperature,
                        _,
                    >::new(boundary),
                });
            }
            (Some(initial), Some(_)) => {
                authored_and_prescribed_node_count += usize::from(is_mesh_node(node_index));
                temperature_k.push(initial);
            }
            (Some(initial), None) => temperature_k.push(initial),
            (None, Some(boundary)) => temperature_k.push(boundary),
            (None, None) => {
                reference_node_count += usize::from(is_mesh_node(node_index));
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
            reference_temperature_k: runmat_types::physical::Temperature::new(
                reference_temperature_k,
            ),
            applied_initial_condition_ids: applied_condition_ids,
            authored_node_count,
            prescribed_node_count,
            authored_and_prescribed_node_count,
            reference_node_count,
            minimum_temperature_k: runmat_types::physical::Temperature::new(minimum_temperature_k),
            maximum_temperature_k: runmat_types::physical::Temperature::new(maximum_temperature_k),
        },
        temperature_k: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Temperature,
            _,
        >::new(temperature_k),
    })
}

fn realize_value(
    model: &AnalysisModel,
    condition: &runmat_analysis_core::InitialConditionValue<runmat_analysis_core::Temperature>,
) -> Result<f64, ThermalInitialStateError> {
    match &condition.value {
        InitialValue::Uniform(value) => Ok(value.canonical_value()),
        InitialValue::SignalSample {
            signal_id,
            abscissa,
            approach,
        } => {
            let mut matching = model
                .signals
                .iter()
                .filter(|signal| signal.signal_id() == signal_id);
            let signal = matching
                .next()
                .filter(|_| matching.next().is_none())
                .ok_or_else(|| ThermalInitialStateError::SignalSample {
                    initial_condition_id: condition.initial_condition_id.clone(),
                    detail: "signal is missing or duplicated".into(),
                })?;
            let PhysicalSignal::Temperature(signal) = signal else {
                return Err(ThermalInitialStateError::SignalSample {
                    initial_condition_id: condition.initial_condition_id.clone(),
                    detail: "signal is not a temperature".into(),
                });
            };
            let sample = crate::excitation::initial_signal_sample_point(
                &signal.representation,
                *abscissa,
                *approach,
            )
            .map_err(|error| ThermalInitialStateError::SignalSample {
                initial_condition_id: condition.initial_condition_id.clone(),
                detail: error.to_string(),
            })?;
            let value = sample_signal(signal, sample, |value| value.canonical_value())
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
        InitialValue::FieldArtifact(_) => Err(ThermalInitialStateError::FieldAdmission {
            detail: "nodal field must be admitted before scalar realization".into(),
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
    crate::excitation::initial_target_nodes(projection, &condition.target).map_err(|error| {
        match error {
            crate::excitation::InitialTargetError::InvalidTarget => {
                ThermalInitialStateError::InvalidTarget {
                    initial_condition_id: condition.initial_condition_id.clone(),
                }
            }
            crate::excitation::InitialTargetError::UnmappedRegion => {
                ThermalInitialStateError::UnmappedRegion {
                    initial_condition_id: condition.initial_condition_id.clone(),
                }
            }
            crate::excitation::InitialTargetError::InvalidSource => {
                ThermalInitialStateError::FieldAdmission {
                    detail: "invalid initial target source node".into(),
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    mod profile;
    use runmat_analysis_core::{
        AnalysisSequence, CoordinateFrame, InitialConditionTarget, InitialConditionValue,
        InitialFieldArtifactRef, ProcedureStep, RegionTarget, Temperature, ThermalDomain,
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
            &[],
        )
        .unwrap();
        assert!(state
            .temperature_k
            .canonical_values()
            .iter()
            .all(|value| *value == 325.0));
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
                &[],
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
                &[],
            ),
            Err(ThermalInitialStateError::FieldAdmission { .. })
        ));
    }

    #[test]
    fn prescribed_temperature_is_part_of_the_initial_state() {
        let (model, _mesh, projection) = thermal_model();
        let mut prescribed = vec![None; projection.nodes.len()];
        prescribed[0] = Some(310.0);

        let state = realize(
            &model,
            &projection,
            "thermal_step",
            293.15,
            &prescribed,
            &[],
        )
        .unwrap();

        assert_eq!(state.temperature_k.canonical_values()[0], 310.0);
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
            realize(
                &model,
                &projection,
                "thermal_step",
                293.15,
                &prescribed,
                &[]
            ),
            Err(ThermalInitialStateError::PrescribedTemperatureConflict { .. })
        ));
    }
}
