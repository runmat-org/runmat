//! Native recovered fields of statistical structural procedures on the canonical solver mesh.
//!
//! A random-response step's node displacement is its root-mean-square displacement, and its
//! element strain and stress are their root-mean-square values; a response-spectrum step's are
//! its combined peak (envelope) values. These are the fields authored outputs of those steps
//! observe. They are recovered once, from the step's own solver result, into the family-neutral
//! volume-mesh catalog that the shared observation engine evaluates.

use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape, RecoveryPolicy,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, FieldRecoveryCost, FieldRecoveryPlan,
    FieldRecoveryRequest, FieldRecoveryRequirement, FieldRecoverySource, FieldTopologyIdentity,
    SolverResultIdentity, StepIdentity,
};
use crate::post::StructuralLinearFieldQuantity;
use crate::results::field::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition, RecoveredFieldEntityCounts,
    RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation, RecoveredFieldSample,
};
use crate::topology::AnalysisProjection;

/// One statistic field of a statistical step, in canonical mesh order.
pub struct StatisticalFieldValues<'a> {
    pub quantity: StructuralLinearFieldQuantity,
    pub field_id: String,
    pub shape: &'a [usize],
    pub values: &'a [f64],
}

fn definition(
    quantity: StructuralLinearFieldQuantity,
    field_id: &str,
) -> AnalysisExecutionResult<Option<RecoveredFieldDefinition>> {
    let (observed, location, domain, shape) = match quantity {
        StructuralLinearFieldQuantity::Displacement => (
            ObservationQuantity::Displacement,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Vector3,
        ),
        StructuralLinearFieldQuantity::Strain => (
            ObservationQuantity::Strain,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            ObservationValueShape::SymmetricTensor3,
        ),
        StructuralLinearFieldQuantity::Stress => (
            ObservationQuantity::Stress,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            ObservationValueShape::SymmetricTensor3,
        ),
        // Rotations and shell/beam resultants have no solid-mesh observation domain here.
        _ => return Ok(None),
    };
    RecoveredFieldDefinition::new(
        field_id,
        observed,
        location,
        domain,
        shape,
        RecoveredFieldNumericRepresentation::Real,
    )
    .map(Some)
    .map_err(invalid)
}

fn entity_count(
    domain: RecoveredFieldEntityDomain,
    projection: &AnalysisProjection,
) -> AnalysisExecutionResult<u64> {
    let count = match domain {
        RecoveredFieldEntityDomain::Nodes => projection.nodes.len(),
        RecoveredFieldEntityDomain::VolumeElements => projection.elements.len(),
        _ => {
            return Err(invalid(
                "statistical fields live on nodes or volume elements",
            ))
        }
    };
    u64::try_from(count).map_err(|_| invalid("statistical field entity count overflows"))
}

/// The recovery request of a statistical step's fields: one native requirement per field, on
/// the canonical volume mesh, sourced from the step's own solver result.
pub fn statistical_field_request(
    producer: StepIdentity,
    source: SolverResultIdentity,
    mesh: &SolverMeshArtifact,
    fields: &[StatisticalFieldValues<'_>],
) -> AnalysisExecutionResult<FieldRecoveryRequest> {
    let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
    let mut requirements = Vec::new();
    for field in fields {
        let Some(definition) = definition(field.quantity, &field.field_id)? else {
            continue;
        };
        let count = entity_count(definition.entity_domain(), &projection)?;
        requirements.push(FieldRecoveryRequirement {
            support: Default::default(),
            quantity: definition.quantity(),
            source_location: definition.location(),
            output_location: definition.location(),
            recovery: RecoveryPolicy::Native,
            value_shape: definition.value_shape().clone(),
            entity_domain: definition.entity_domain(),
            cost: FieldRecoveryCost {
                source_entity_count: count,
                output_entity_count: count,
                topology_incidence_count: 0,
                arithmetic_operation_count: count,
            },
        });
    }
    if requirements.is_empty() {
        return Err(invalid("statistical step publishes no observable field"));
    }
    FieldRecoveryRequest::new(
        producer,
        FieldRecoverySource::SolverResult(source),
        FieldTopologyIdentity::VolumeMesh {
            mesh_digest: mesh.canonical_digest,
        },
        FieldRecoveryPlan { requirements },
    )
}

/// The recovered field artifact for `request`, built from the statistical step's own values.
pub fn statistical_recovered_fields(
    request: &FieldRecoveryRequest,
    mesh: &SolverMeshArtifact,
    fields: &[StatisticalFieldValues<'_>],
) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
    let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
    let mut recovered = Vec::new();
    for field in fields {
        let Some(definition) = definition(field.quantity, &field.field_id)? else {
            continue;
        };
        let count = entity_count(definition.entity_domain(), &projection)? as usize;
        if field.shape.first() != Some(&count) {
            return Err(invalid(
                "statistical field shape differs from the canonical mesh projection",
            ));
        }
        let layout = definition.layout();
        recovered.push(
            RecoveredField::new(
                definition,
                RecoveredFieldSample::Final,
                AnalysisField::host_f64(
                    field.field_id.clone(),
                    layout,
                    field.shape.to_vec(),
                    field.values.to_vec(),
                ),
            )
            .map_err(invalid)?,
        );
    }
    let artifact = RecoveredFieldArtifact::new(
        request.identity(),
        request.source,
        request.topology.discretization_digest(),
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: projection.nodes.len(),
            volume_elements: projection.elements.len(),
            volume_integration_points: 0,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        recovered,
    )
    .map_err(invalid)?;
    request.plan.validate_definitions(
        artifact
            .fields()
            .iter()
            .map(|field| field.definition().clone()),
    )?;
    Ok(artifact)
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}

/// The root-mean-square fields of a random-response result.
pub fn random_response_statistical_fields(
    result: &crate::solve::dynamics::ModalRandomResponseResult,
) -> Vec<StatisticalFieldValues<'_>> {
    result
        .fields()
        .iter()
        .map(|field| StatisticalFieldValues {
            quantity: field.quantity(),
            field_id: StatisticalProcedure::RandomResponse.field_id(field.quantity()),
            shape: field.shape(),
            values: field.rms(),
        })
        .collect()
}

/// The statistical procedure whose step fields a recovery copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatisticalProcedure {
    /// Root-mean-square response to a random excitation.
    RandomResponse,
    /// Combined peak (envelope) response to a response spectrum.
    ResponseSpectrum,
}

impl StatisticalProcedure {
    pub fn field_id(self, quantity: StructuralLinearFieldQuantity) -> String {
        match self {
            Self::RandomResponse => {
                crate::contracts::RandomResponseFieldSelector::root_mean_square(quantity).field_id()
            }
            Self::ResponseSpectrum => {
                crate::contracts::ResponseSpectrumFieldSelector::Deformation(quantity).field_id()
            }
        }
    }
}

/// Field definitions a statistical recovery `request` publishes, in requirement order;
/// resource planning profiles these before any value is recovered.
pub fn statistical_field_definitions(
    procedure: StatisticalProcedure,
    request: &FieldRecoveryRequest,
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    request.validate()?;
    request
        .plan
        .requirements
        .iter()
        .map(|requirement| {
            let quantity = match requirement.quantity {
                ObservationQuantity::Displacement => StructuralLinearFieldQuantity::Displacement,
                ObservationQuantity::Strain => StructuralLinearFieldQuantity::Strain,
                ObservationQuantity::Stress => StructuralLinearFieldQuantity::Stress,
                _ => return Err(invalid("statistical recovery quantity is unsupported")),
            };
            definition(quantity, &procedure.field_id(quantity))?
                .ok_or_else(|| invalid("statistical recovery quantity is unsupported"))
        })
        .collect()
}

/// The combined-peak deformation fields of a response-spectrum recovery. A response that is not
/// convertible to displacement has none, so its authored field outputs fail closed.
pub fn response_spectrum_statistical_fields(
    recovery: &crate::solve::dynamics::ResponseSpectrumFieldRecovery,
) -> Vec<StatisticalFieldValues<'_>> {
    recovery
        .fields()
        .unwrap_or_default()
        .iter()
        .map(|field| StatisticalFieldValues {
            quantity: field.quantity(),
            field_id: StatisticalProcedure::ResponseSpectrum.field_id(field.quantity()),
            shape: field.shape(),
            values: field.values(),
        })
        .collect()
}
