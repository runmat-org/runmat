use runmat_analysis_core::{AnalysisModel, ConnectorKind, StructuralElementKind};

use crate::topology::AnalysisProjection;

use super::ResourcePlanningError;

// Conservative, platform-stable allocation envelopes for one retained canonical
// quadrature point and one simultaneously resolved input point. They include
// the outer Vec slot, nested node/shape Vec payload and capacity growth, and
// allocation overhead. Tri6 is the largest admitted foundation face.
const TRI6_CANONICAL_BYTES: u64 = 1_024;
const TRI6_RESOLVED_BYTES: u64 = 512;
const TRI3_CANONICAL_BYTES: u64 = 640;
const TRI3_RESOLVED_BYTES: u64 = 384;
const SURFACE_FACE_WORKSPACE_BYTES: u64 = 1_024;
const EXPLICIT_NODE_MAP_BYTES: u64 = 256;
// Per connector: bounded canonical entry, contribution provenance, response enum, Vec slots,
// and allocation overhead. Identifier bytes are charged separately for the simultaneous
// authored, assembly-plan, recovered-plan, and response/provenance copies.
const CONNECTOR_METADATA_BYTES: u64 = 4_096;
// Two published arrays (three force components and one energy) and their field wrappers.
const CONNECTOR_RESULT_FIELD_BYTES: u64 = 1_024;
const CONNECTOR_ID_COPIES: u64 = 4;
// Covers the largest connector response's numeric values plus both published field arrays.
const CONNECTOR_RESULT_VALUES: u64 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ConnectorBufferEstimate {
    pub assembly_peak_bytes: u64,
    pub recovery_retained_bytes: u64,
    pub recovery_value_count: u64,
}

pub(super) fn estimate_connector_buffers(
    model: &AnalysisModel,
    projection: Option<&AnalysisProjection>,
    topology_node_count: u64,
) -> Result<ConnectorBufferEstimate, ResourcePlanningError> {
    let mut retained = 0_u64;
    let mut transient_peak = 0_u64;
    for connector in &model.connectors {
        let id_bytes = multiply(
            u64::try_from(connector.connector_id.len())
                .map_err(|_| ResourcePlanningError::Overflow("connector identifier bytes"))?,
            CONNECTOR_ID_COPIES,
            "connector identifier copies",
        )?;
        retained = sum(
            retained,
            sum(CONNECTOR_METADATA_BYTES, id_bytes, "connector metadata")?,
            "all connector metadata",
        )?;
        let ConnectorKind::SurfaceNormalFoundation { target, .. } = &connector.kind else {
            continue;
        };
        let (solid_faces, solid_samples) = projection.map_or(
            Ok::<_, ResourcePlanningError>((0_u64, 0_u64)),
            |projection| {
                projection
                    .boundary_faces
                    .iter()
                    .filter(|face| {
                        face.provenance
                            .iter()
                            .any(|entity| target.matches_persistent_entity(entity))
                    })
                    .try_fold((0_u64, 0_u64), |(faces, samples), face| {
                        let mut face_samples = 0_u64;
                        projection
                            .visit_boundary_quadrature(face, |_| face_samples += 1)
                            .map_err(|error| {
                                ResourcePlanningError::Invalid(format!(
                                "surface foundation `{}` has invalid boundary quadrature: {error}",
                                connector.connector_id
                            ))
                            })?;
                        Ok((
                            sum(faces, 1, "foundation solid face count")?,
                            sum(samples, face_samples, "foundation solid samples")?,
                        ))
                    })
            },
        )?;
        let shell_faces = model.structural.as_ref().map_or(0_usize, |structural| {
            structural
                .elements
                .iter()
                .filter(|element| {
                    element.region_id == target.region_id
                        && matches!(element.kind, StructuralElementKind::Shell(_))
                })
                .count()
        });
        let shell_faces = u64::try_from(shell_faces)
            .map_err(|_| ResourcePlanningError::Overflow("foundation shell face count"))?;
        if solid_faces == 0 && shell_faces == 0 {
            return Err(ResourcePlanningError::Invalid(format!(
                "surface foundation `{}` resolves no structural boundary faces",
                connector.connector_id
            )));
        }
        let shell_samples = multiply(shell_faces, 3, "foundation shell samples")?;
        let canonical = sum(
            multiply(
                solid_samples,
                TRI6_CANONICAL_BYTES,
                "foundation canonical solid samples",
            )?,
            multiply(
                shell_samples,
                TRI3_CANONICAL_BYTES,
                "foundation canonical shell samples",
            )?,
            "foundation canonical samples",
        )?;
        let face_workspace = multiply(
            sum(solid_faces, shell_faces, "foundation face count")?,
            SURFACE_FACE_WORKSPACE_BYTES,
            "foundation face workspace",
        )?;
        let explicit_node_workspace = if shell_faces > 0 {
            let authored_nodes = u64::try_from(
                model
                    .structural
                    .as_ref()
                    .map_or(0, |structural| structural.nodes.len()),
            )
            .map_err(|_| ResourcePlanningError::Overflow("foundation authored node count"))?;
            multiply(
                sum(authored_nodes, topology_node_count, "foundation node maps")?,
                EXPLICIT_NODE_MAP_BYTES,
                "foundation node-map workspace",
            )?
        } else {
            0
        };
        let transient = sum(
            sum(
                multiply(
                    solid_samples,
                    TRI6_RESOLVED_BYTES,
                    "foundation resolved solid samples",
                )?,
                multiply(
                    shell_samples,
                    TRI3_RESOLVED_BYTES,
                    "foundation resolved shell samples",
                )?,
                "foundation resolved samples",
            )?,
            sum(
                face_workspace,
                explicit_node_workspace,
                "foundation resolver workspace",
            )?,
            "foundation transient buffers",
        )?;
        retained = sum(retained, canonical, "all foundation retained buffers")?;
        transient_peak = transient_peak.max(transient);
    }
    let connector_count = u64::try_from(model.connectors.len())
        .map_err(|_| ResourcePlanningError::Overflow("connector count"))?;
    Ok(ConnectorBufferEstimate {
        assembly_peak_bytes: sum(retained, transient_peak, "connector assembly buffer peak")?,
        // Assembly retains one plan while recovery owns its clone. The response vector and the
        // force/energy field arrays are additional retained objects, not assembly scratch.
        recovery_retained_bytes: sum(
            multiply(retained, 2, "two retained connector plans")?,
            multiply(
                connector_count,
                CONNECTOR_RESULT_FIELD_BYTES,
                "connector response and fields",
            )?,
            "connector recovery retained bytes",
        )?,
        recovery_value_count: multiply(
            connector_count,
            CONNECTOR_RESULT_VALUES,
            "connector response value count",
        )?,
    })
}

pub(super) fn include_connector_buffers(
    sparse_contribution_bytes: u64,
    connector_quadrature_bytes: u64,
) -> Result<u64, ResourcePlanningError> {
    sum(
        sparse_contribution_bytes,
        connector_quadrature_bytes,
        "assembly contributions and connector buffers",
    )
}

fn sum(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

fn multiply(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}
