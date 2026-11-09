//! Observation results of a cross-section's line parameters.
//!
//! A line parameter is a global reduction of the section's two potential solves (the energy
//! form of the capacitance matrices and what follows from them). It reads no recovered field,
//! so each authored line-parameter output is evaluated directly from the solved parameters:
//! one `ObservationResult` per lowered spec, under a plan identity derived from the step's
//! specs and the section's mesh, with one-partition evidence whose source is the solved result.
//! Publication and assessments then read these results like any other reduction.
use runmat_analysis_core::{
    AnalysisField, AnalysisFieldLayout, LineParameterScope, ObservationQuantity,
    ObservationTarget, ProcedureStep, SpatialReduction,
};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

use super::contract::{OBSERVATION_LIMITS, OBSERVATION_PREFIX};
use super::distributed::ObservationPartitionEvidence;
use super::partition::ObservationPartitionPlanIdentity;
use super::{ObservationError, ObservationResult, ObservationResultIdentity};
use super::ResolvedObservationPlanIdentity;
use crate::physics::electrical::cross_section::LineParameters;
use crate::results::field::{RecoveredFieldIdentity, RecoveredFieldSample};

const PLAN_DOMAIN: &str = "line-parameter-observation-plan";
const SOURCE_DOMAIN: &str = "line-parameter-source";

/// The topology every line-parameter result is bound to: the section's canonical mesh.
pub fn line_parameter_topology_digest(mesh: &SolverMeshArtifact) -> Digest {
    Digest::from_bytes(*mesh.canonical_digest.bytes())
}

/// The plan identity of a line-parameter step on `mesh`: its specs, in authored order.
pub fn line_parameter_plan_identity(
    step: &ProcedureStep,
    mesh: &SolverMeshArtifact,
) -> Result<ResolvedObservationPlanIdentity, ObservationError> {
    let specs = step
        .observations
        .iter()
        .map(|request| {
            request.spec().ok_or_else(|| {
                ObservationError::Invalid(
                    "a line-parameter step publishes only line-parameter reductions".into(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ResolvedObservationPlanIdentity(
        crate::contracts::canonical_digest::canonical_contract_digest(
            OBSERVATION_PREFIX,
            PLAN_DOMAIN,
            &(
                1u16,
                line_parameter_topology_digest(mesh),
                &step.step_id,
                &step.independent_domain,
                &specs,
            ),
            OBSERVATION_LIMITS,
        )?,
    ))
}

fn source_identity(parameters: &LineParameters) -> Result<Digest, ObservationError> {
    Ok(crate::contracts::canonical_digest::canonical_contract_digest(
        OBSERVATION_PREFIX,
        SOURCE_DOMAIN,
        parameters,
        OBSERVATION_LIMITS,
    )?)
}

/// One result per authored observation of `step`, evaluated from `parameters` solved on `mesh`.
pub fn evaluate_line_parameter_observations(
    step: &ProcedureStep,
    mesh: &SolverMeshArtifact,
    parameters: &LineParameters,
) -> Result<Vec<ObservationResult>, ObservationError> {
    let plan_identity = line_parameter_plan_identity(step, mesh)?;
    let topology_digest = line_parameter_topology_digest(mesh);
    let source = RecoveredFieldIdentity::derived(source_identity(parameters)?);
    let sample = RecoveredFieldSample::Final;
    step.observations
        .iter()
        .map(|request| {
            let spec = request.spec().ok_or_else(|| {
                ObservationError::Invalid(
                    "a line-parameter step publishes only line-parameter reductions".into(),
                )
            })?;
            if spec.spatial_reduction() != SpatialReduction::LineParameter {
                return Err(ObservationError::Invalid(format!(
                    "observation `{}` is not a line parameter",
                    spec.observation_id.0
                )));
            }
            let value_shape = spec.result_value_shape();
            let values = line_parameter_values(spec.quantity, &spec.target, parameters)
                .map_err(|reason| ObservationError::Target {
                    observation_id: spec.observation_id.0.clone(),
                    reason,
                })?;
            let expected = value_shape.component_count().unwrap_or(0);
            if values.len() != expected {
                return Err(ObservationError::Invalid(format!(
                    "observation `{}` declares {expected} values and the section has {}",
                    spec.observation_id.0,
                    values.len()
                )));
            }
            let shape =
                super::contract::observation_result_storage_shape(None, &value_shape, spec.complex_representation)?;
            let contribution = crate::contracts::canonical_digest::canonical_contract_digest(
                OBSERVATION_PREFIX,
                "line-parameter-contribution",
                &(plan_identity, &spec.observation_id, source, &values),
                OBSERVATION_LIMITS,
            )?;
            let partition_evidence = ObservationPartitionEvidence::direct(
                ObservationPartitionPlanIdentity::derived(plan_identity.digest()),
                source,
                contribution,
                sample,
            )?;
            let mut result = ObservationResult {
                schema_version: 1,
                logical_identity: ObservationResultIdentity(Digest::sha256([])),
                plan_identity,
                topology_digest,
                observation_id: spec.observation_id.clone(),
                sample,
                quantity: spec.quantity,
                location: spec.output_location,
                value_shape: value_shape.clone(),
                complex_representation: spec.complex_representation,
                entities: Vec::new(),
                field: AnalysisField::host_f64(
                    spec.observation_id.0.clone(),
                    AnalysisFieldLayout::new(spec.quantity, spec.output_location, value_shape)
                        .representation(spec.complex_representation),
                    shape,
                    values,
                ),
                partial_access: spec.storage.partial_access,
                partition_evidence,
            };
            result.logical_identity = result.derive_identity()?;
            result.validate()?;
            Ok(result)
        })
        .collect()
}

/// Check published results against the step's plan: one valid result per spec, in order,
/// under the plan identity of `step` on `mesh`.
pub fn validate_line_parameter_observations(
    step: &ProcedureStep,
    mesh: &SolverMeshArtifact,
    results: &[ObservationResult],
) -> Result<(), ObservationError> {
    let plan_identity = line_parameter_plan_identity(step, mesh)?;
    let topology_digest = line_parameter_topology_digest(mesh);
    if results.len() != step.observations.len() {
        return Err(ObservationError::Invalid(
            "line-parameter results do not cover the step's outputs exactly".into(),
        ));
    }
    for (request, result) in step.observations.iter().zip(results) {
        result.validate()?;
        let spec = request.spec().ok_or_else(|| {
            ObservationError::Invalid("line-parameter step has a non-reduction output".into())
        })?;
        if result.plan_identity != plan_identity
            || result.topology_digest != topology_digest
            || result.observation_id != spec.observation_id
            || result.quantity != spec.quantity
            || result.value_shape != spec.result_value_shape()
        {
            return Err(ObservationError::Identity(
                "line-parameter result differs from its plan",
            ));
        }
    }
    Ok(())
}

fn line_parameter_values(
    quantity: ObservationQuantity,
    target: &ObservationTarget,
    parameters: &LineParameters,
) -> Result<Vec<f64>, String> {
    let scope = quantity
        .line_parameter_scope()
        .ok_or_else(|| format!("{quantity:?} is not a line parameter"))?;
    Ok(match scope {
        LineParameterScope::Conductor => {
            let ObservationTarget::Conductor { conductor_id } = target else {
                return Err("a per-line value names one signal conductor".into());
            };
            let index = parameters
                .conductors
                .iter()
                .position(|name| name == conductor_id)
                .ok_or_else(|| {
                    format!("`{conductor_id}` is not a signal conductor of the section")
                })?;
            vec![match quantity {
                ObservationQuantity::LineImpedance => parameters.impedance_ohm[index],
                ObservationQuantity::LineEffectivePermittivity => {
                    parameters.effective_permittivity[index]
                }
                ObservationQuantity::LineDelay => parameters.delay_s_per_m[index],
                _ => unreachable!("conductor scope"),
            }]
        }
        LineParameterScope::Pair => {
            let pair = parameters
                .pair
                .as_ref()
                .ok_or("a pair value needs exactly two signal conductors")?;
            vec![match quantity {
                ObservationQuantity::DifferentialImpedance => pair.differential_impedance_ohm,
                ObservationQuantity::CommonModeImpedance => pair.common_mode_impedance_ohm,
                ObservationQuantity::OddModeImpedance => pair.odd_mode_impedance_ohm,
                ObservationQuantity::EvenModeImpedance => pair.even_mode_impedance_ohm,
                ObservationQuantity::NearEndCoupling => pair.near_end_coupling,
                ObservationQuantity::FarEndCoupling => pair.far_end_coupling,
                _ => unreachable!("pair scope"),
            }]
        }
        LineParameterScope::Matrix => {
            let matrix = match quantity {
                ObservationQuantity::LineCapacitance => &parameters.capacitance_f_per_m,
                ObservationQuantity::LineVacuumCapacitance => {
                    &parameters.vacuum_capacitance_f_per_m
                }
                ObservationQuantity::LineInductance => &parameters.inductance_h_per_m,
                _ => unreachable!("matrix scope"),
            };
            matrix.iter().flatten().copied().collect()
        }
    })
}
