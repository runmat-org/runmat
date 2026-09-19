//! Remote composition for a typed study whose workloads are derived after exact meshing.

use std::collections::BTreeMap;
use std::sync::Arc;

use runmat_execution::Digest;
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use super::exact_meshing::{MeshedGeometryRegions, RemoteExactMeshingComposition};
use super::pool_execution::RemotePoolExecutionOutcome;
use super::pool_tasks::{execution_failure_outcome, packaged};
use super::RemotePoolDriver;
use crate::{NativeExecutionResult, StructuralTransientRecoveryPublisher};

pub(super) async fn execute(
    pool: Arc<RemotePoolDriver>,
    request: ProgramExecutionRequest,
    run_id: String,
    resources: AllocationResources,
    recovered: Option<&super::recovery::RestoredObjectClosure>,
    recovery_publisher: &mut super::recovery::RemoteRecoveryPublisher<'_>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    match execute_for_composition(
        Arc::clone(&pool),
        &request,
        &run_id,
        &resources,
        recovered,
        recovery_publisher,
    )
    .await?
    {
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Completed(
            response,
        ) => packaged(response, pool).map(RemotePoolExecutionOutcome::Completed),
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Failed(message) => {
            execution_failure_outcome(pool, message)
        }
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Cancelled => {
            Ok(RemotePoolExecutionOutcome::Cancelled)
        }
        super::exact_structural_study::RemoteExactStructuralStudyComposition::Indeterminate(
            message,
        ) => Ok(RemotePoolExecutionOutcome::Indeterminate(message)),
    }
}

pub(super) async fn execute_for_composition(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    run_id: &str,
    resources: &AllocationResources,
    recovered: Option<&super::recovery::RestoredObjectClosure>,
    recovery_publisher: &mut dyn StructuralTransientRecoveryPublisher,
) -> NativeExecutionResult<super::exact_structural_study::RemoteExactStructuralStudyComposition> {
    let decoded = crate::decode_structural_study_root(request, pool.as_ref())?;
    let geometry_contract =
        crate::decode_geometry_meshing_root(&decoded.geometry_meshing, pool.as_ref())?;
    let completed = match super::exact_meshing::execute_geometry_for_composition(
        Arc::clone(&pool),
        &decoded.geometry_meshing,
        run_id,
        resources,
    )
    .await?
    {
        RemoteExactMeshingComposition::Completed(completed) => completed,
        RemoteExactMeshingComposition::Terminal(outcome) => {
            return Ok(super::exact_structural_study::composition(outcome))
        }
    };
    let regions = match &completed.regions {
        MeshedGeometryRegions::Materialized(regions) => regions,
        MeshedGeometryRegions::ImporterAliases(_) => {
            return Ok(
                super::exact_structural_study::RemoteExactStructuralStudyComposition::Failed(
                    "typed structural studies require source-neutral materialized geometry".into(),
                ),
            )
        }
    };
    let product = match runmat_analysis_fea::study::build_structural_study_product(
        runmat_analysis_fea::study::StructuralStudyProductInput {
            study: &decoded.study,
            geometry: &completed.document,
            regions,
            materials: &decoded.materials,
            mesh: &completed.result.artifact,
            planning: decoded.planning,
        },
    ) {
        Ok(product) => product,
        Err(error) => {
            return Ok(
                super::exact_structural_study::RemoteExactStructuralStudyComposition::Failed(
                    error.to_string(),
                ),
            )
        }
    };
    let geometry_source_digest =
        Digest::from_bytes(*completed.result.artifact.geometry.source_digest.bytes());
    let compiled_geometry_output = completed.materialization.clone().ok_or_else(|| {
        crate::NativeExecutionError::Protocol(
            "typed structural study geometry has no materialization receipt".into(),
        )
    })?;
    let mut completed = completed;
    completed.regions = MeshedGeometryRegions::ImporterAliases(BTreeMap::new());
    let execution_view = crate::DecodedExactStructuralStudyRoot {
        authored_study_digest: decoded.authored_study_digest,
        frozen_project_digest: decoded.frozen_project_digest,
        geometry_source_digest,
        workloads: product.workloads,
        analysis_accelerator_requests: decoded.analysis_accelerator_requests,
        input_materializations: Vec::new(),
        exact_meshing: decoded.geometry_meshing,
        meshing_request: geometry_contract.request.meshing_request,
        artifact_access: decoded.artifact_access,
        compiled_study: Some(product.compiled),
        compiled_geometry_output: Some(compiled_geometry_output),
    };
    super::exact_structural_study::execute_with_completed_meshing(
        pool,
        request,
        execution_view,
        completed,
        &BTreeMap::new(),
        resources,
        super::exact_structural_study::StructuralTransientRecoveryAdmission::Available {
            restored: recovered,
            publisher: recovery_publisher,
        },
    )
    .await
}
