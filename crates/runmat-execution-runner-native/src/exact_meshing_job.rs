//! One-call native composition of exact geometry admission and the canonical meshing DAG.

use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_geometry_io::{ExactCadImportOptions, GeometryImportContext};
use runmat_meshing_core::{
    MeshingDomainModel, MeshingRequest, RegionMaterialAssignment,
    MESHING_DOMAIN_MODEL_SCHEMA_VERSION,
};
use runmat_meshing_execution::{
    execute_exact_meshing_dag, prepare_domain_model_input, prepare_domain_model_objects,
    ExactMeshingDagRun, ExactMeshingDagRunError, ExactMeshingDagRunResult, MeshingArtifactAccess,
    MeshingRunEvidenceContext,
};

use crate::{
    admit_prepared_exact_geometry, prepare_exact_geometry_admission, ExactGeometryAdmissionError,
    NativeExactMeshingExecutor, NativeExecutionConfig, NativeExecutionError,
    NativeMeshingExecutionPolicy, NativeProgramSession,
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeMeshingDomain {
    pub default_material_id: Option<String>,
    pub region_materials: BTreeMap<String, String>,
}

pub struct NativeExactMeshingJob<'a> {
    pub source_name: &'a str,
    pub source_bytes: &'a [u8],
    pub import_options: ExactCadImportOptions,
    pub request: MeshingRequest,
    pub domain: NativeMeshingDomain,
    pub program_revision: ProgramRevision,
    pub capability_cohort: Option<String>,
    pub preferred_edges_per_partition: u32,
    pub preferred_faces_per_partition: u32,
    pub inventory_limits: ObjectInventoryLimits,
    pub evidence: MeshingRunEvidenceContext,
    pub execution: NativeMeshingExecutionPolicy,
}

#[derive(Debug, thiserror::Error)]
pub enum NativeExactMeshingJobError {
    #[error(transparent)]
    Admission(#[from] ExactGeometryAdmissionError),
    #[error("meshing request tolerance differs from exact geometry import policy")]
    ToleranceMismatch,
    #[error("native meshing domain is invalid: {0}")]
    InvalidDomain(String),
    #[error("domain-model artifact preparation failed: {0}")]
    DomainModel(#[from] runmat_meshing_execution::MeshingExecutionError),
    #[error("domain-model object persistence failed: {0}")]
    Store(#[from] runmat_execution_artifact::ArtifactError),
    #[error("native meshing executor configuration failed: {0}")]
    Executor(#[from] runmat_meshing_execution::MeshingStageExecutionError),
    #[error("native execution session failed: {0}")]
    Session(#[from] NativeExecutionError),
    #[error(transparent)]
    Run(#[from] ExactMeshingDagRunError),
}

/// Executes exact CAD import through atomic solver-artifact publication in one native session.
///
/// Execution authority is derived from the session, so ordinary local use requires no Server or
/// user-managed artifact credentials. The returned artifact identity remains independent of that
/// physical authority.
pub fn mesh_exact_geometry(
    mut config: NativeExecutionConfig,
    mut job: NativeExactMeshingJob<'_>,
) -> Result<ExactMeshingDagRunResult, NativeExactMeshingJobError> {
    let prepared_geometry = prepare_exact_geometry_admission(
        job.source_name,
        job.source_bytes,
        &job.import_options,
        &GeometryImportContext::new(),
        job.inventory_limits,
    )?;
    let admitted_tolerance = prepared_geometry.document().tolerance;
    if !same_requested_tolerance(&job.request.tolerance, &admitted_tolerance) {
        return Err(NativeExactMeshingJobError::ToleranceMismatch);
    }
    job.request.tolerance = admitted_tolerance;
    job.evidence.platform.exact_kernel_abi =
        prepared_geometry.document().source.kernel_version.clone();
    let domain_model = resolve_domain_model(prepared_geometry.topology(), &job.domain)?;
    config.enable_exact_meshing(
        prepared_geometry.document(),
        &job.request,
        job.capability_cohort.as_deref(),
    )?;
    let session = NativeProgramSession::new(config)?;
    let access = session_artifact_access(&session);
    let geometry = admit_prepared_exact_geometry(
        &session,
        prepared_geometry,
        access.clone(),
        job.inventory_limits,
    )?;

    let domain_objects = prepare_domain_model_objects(domain_model, job.inventory_limits)?;
    let domain = prepare_domain_model_input(domain_objects, access.clone(), job.inventory_limits)?;
    let mut store = session.object_store();
    for object in &domain.domain_model_objects().objects {
        store.write_verified(object)?;
    }
    let mut executor = NativeExactMeshingExecutor::new(&session, job.execution)?;
    execute_exact_meshing_dag(
        ExactMeshingDagRun {
            geometry: &geometry,
            domain_model: &domain,
            request: job.request,
            artifact_access: access,
            capability_cohort: job.capability_cohort,
            program_revision: job.program_revision,
            preferred_edges_per_partition: job.preferred_edges_per_partition,
            preferred_faces_per_partition: job.preferred_faces_per_partition,
            inventory_limits: job.inventory_limits,
            evidence: job.evidence,
        },
        &mut executor,
    )
    .map_err(NativeExactMeshingJobError::from)
}

fn resolve_domain_model(
    topology: &runmat_geometry_core::ExactBRepTopology,
    domain: &NativeMeshingDomain,
) -> Result<MeshingDomainModel, NativeExactMeshingJobError> {
    let mut matched = BTreeSet::new();
    let mut region_materials = Vec::with_capacity(topology.regions.len());
    for region in &topology.regions {
        let source_id = &region.id.source_topology_id;
        let material_id = if let Some(material_id) = domain.region_materials.get(source_id) {
            matched.insert(source_id.clone());
            material_id.clone()
        } else {
            domain.default_material_id.clone().ok_or_else(|| {
                NativeExactMeshingJobError::InvalidDomain(format!(
                    "exact region {source_id:?} has no material assignment"
                ))
            })?
        };
        region_materials.push(RegionMaterialAssignment {
            region_id: region.id.clone(),
            material_id,
        });
    }
    if let Some(unknown) = domain
        .region_materials
        .keys()
        .find(|source_id| !matched.contains(*source_id))
    {
        return Err(NativeExactMeshingJobError::InvalidDomain(format!(
            "material assignment names unknown exact region {unknown:?}"
        )));
    }
    let model = MeshingDomainModel {
        schema_version: MESHING_DOMAIN_MODEL_SCHEMA_VERSION,
        region_materials,
        contact_ids: topology
            .contacts
            .iter()
            .map(|contact| contact.id.clone())
            .collect(),
    };
    model
        .validate_against_exact_topology(topology)
        .map_err(|error| NativeExactMeshingJobError::InvalidDomain(error.to_string()))?;
    Ok(model)
}

fn session_artifact_access(session: &NativeProgramSession) -> MeshingArtifactAccess {
    MeshingArtifactAccess {
        authorization_scope: format!("native-meshing-session:{}", session.scope_id()),
        encryption_context: Digest::sha256(
            [
                b"runmat-native-meshing-session\0".as_slice(),
                session.scope_id().bytes(),
            ]
            .concat(),
        ),
    }
}

fn same_requested_tolerance(
    requested: &runmat_geometry_core::GeometryTolerancePolicy,
    admitted: &runmat_geometry_core::GeometryTolerancePolicy,
) -> bool {
    requested.absolute_floor_m == admitted.absolute_floor_m
        && requested.model_relative_term == admitted.model_relative_term
        && requested.requested_deviation_m == admitted.requested_deviation_m
        && requested.maximum_healing_displacement_m == admitted.maximum_healing_displacement_m
}

#[cfg(test)]
mod tests {
    use runmat_geometry_core::GeometryTolerancePolicy;

    use super::same_requested_tolerance;
    #[cfg(feature = "occt-native")]
    use super::{resolve_domain_model, NativeMeshingDomain};

    #[test]
    fn source_tolerance_is_resolved_by_import_but_user_policy_must_match() {
        let requested = GeometryTolerancePolicy {
            source_tolerance_m: 0.0,
            absolute_floor_m: 1.0e-12,
            model_relative_term: 1.0e-12,
            requested_deviation_m: 1.0e-4,
            maximum_healing_displacement_m: 1.0e-6,
        };
        let mut admitted = requested;
        admitted.source_tolerance_m = 2.0e-8;
        assert!(same_requested_tolerance(&requested, &admitted));
        admitted.requested_deviation_m = 2.0e-4;
        assert!(!same_requested_tolerance(&requested, &admitted));
    }

    #[cfg(feature = "occt-native")]
    #[test]
    fn uniform_material_intent_resolves_after_exact_import() {
        let imported = runmat_geometry_io::import_exact_cad(
            "box.brep",
            include_bytes!("../../runmat-geometry/io/tests/fixtures/box.brep"),
            runmat_geometry_io::GeometryFormat::Brep,
            &runmat_geometry_io::ExactCadImportOptions::default(),
            &runmat_geometry_io::GeometryImportContext::new(),
        )
        .unwrap();
        let model = resolve_domain_model(
            &imported.topology,
            &NativeMeshingDomain {
                default_material_id: Some("steel".into()),
                region_materials: Default::default(),
            },
        )
        .unwrap();
        assert_eq!(
            model.region_materials.len(),
            imported.topology.regions.len()
        );
        assert!(model
            .region_materials
            .iter()
            .all(|assignment| assignment.material_id == "steel"));
    }

    #[cfg(feature = "occt-native")]
    #[test]
    fn material_intent_rejects_missing_and_unknown_regions() {
        let imported = runmat_geometry_io::import_exact_cad(
            "box.brep",
            include_bytes!("../../runmat-geometry/io/tests/fixtures/box.brep"),
            runmat_geometry_io::GeometryFormat::Brep,
            &runmat_geometry_io::ExactCadImportOptions::default(),
            &runmat_geometry_io::GeometryImportContext::new(),
        )
        .unwrap();
        assert!(resolve_domain_model(&imported.topology, &NativeMeshingDomain::default()).is_err());

        let domain = NativeMeshingDomain {
            default_material_id: Some("steel".into()),
            region_materials: [("not-a-region".into(), "copper".into())].into(),
        };
        assert!(resolve_domain_model(&imported.topology, &domain).is_err());
    }
}
