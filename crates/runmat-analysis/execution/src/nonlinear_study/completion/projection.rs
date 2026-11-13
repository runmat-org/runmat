//! Consume validated cold completion buffers before original block projection.
//! Immutable root authentication remains upstream; prefix import uses the same coordinator.
use super::*;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};

pub struct OriginalNonlinearSolidBlockProjection {
    pub workload: runmat_analysis_fea::execution::NonlinearSolidWorkload,
    pub mesh: SolverMeshArtifact,
    pub block: ImportedNonlinearSolidBlock,
    pub observations: Option<ImportedObservations>,
}
impl OriginalNonlinearSolidBlockProjection {
    pub fn retained_storage_bytes(&self, maximum: u64) -> AnalysisBridgeResult<u64> {
        let profile = profile_serialized(
            &(
                &self.workload,
                &self.mesh,
                self.observations.as_ref().map(|o| (&o.manifest, &o.output)),
            ),
            limits(maximum)?,
        )?;
        let block = self.block.retained_storage_bytes(maximum)?;
        let observations = self
            .observations
            .as_ref()
            .map_or(Ok(0), |o| o.catalog_objects.retained_storage_bytes())?;
        profile
            .materialization_resident_bytes_estimate
            .checked_add(block)
            .and_then(|n| n.checked_add(observations))
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>() as u64))
            .ok_or_else(|| limit("original projection retained storage overflow"))
    }
}

fn limits(maximum: u64) -> AnalysisBridgeResult<CanonicalLimits> {
    Ok(CanonicalLimits::content_bounded(
        usize::try_from(maximum)
            .map_err(|_| limit("original projection limit exceeds address space"))?,
        65_536,
        64,
    ))
}

impl CompletedNonlinearSolidStudy {
    /// Move only original inputs into prefix replay. Unneeded decoded terminal
    /// buffers release after complete root validation, before a second replay begins.
    pub fn into_original_block_projection(
        self,
        source: &impl CacheImport,
        ordinal: u64,
        additional_retained_bytes: u64,
    ) -> AnalysisBridgeResult<OriginalNonlinearSolidBlockProjection> {
        workspace::validate(&self.closures)?;
        let terminal_ordinal = self.terminal_block.block.request.block_ordinal;
        let Self {
            replay,
            compiled,
            workload,
            terminal_block,
            accuracy,
            fields,
            mut observations,
            observation_catalog,
            states,
            baseline_usage,
            terminal_usage,
            publication,
            mut closures,
        } = self;
        drop((
            terminal_block,
            accuracy,
            fields,
            observation_catalog,
            states,
            baseline_usage,
            terminal_usage,
            publication,
        ));
        if ordinal != terminal_ordinal {
            observations = None;
        }
        // Retain exact original scheduler responses for the requested prefix.
        // Vec slot capacity stays charged even after discarded closures release.
        closures.retain(|c| {
            matches!(c.phase,
            NonlinearSolidStudyResultPhase::Block { ordinal: original } if original <= ordinal)
        });
        let maximum = workload.model.sequence.steps[0]
            .accuracy
            .resources
            .resident_byte_budget();
        let observer_profile = profile_serialized(
            &observations.as_ref().map(|o| (&o.manifest, &o.output)),
            limits(maximum)?,
        )?
        .materialization_resident_bytes_estimate;
        let observer_wire = observations
            .as_ref()
            .map_or(Ok(0), |o| o.catalog_objects.retained_storage_bytes())?;
        let slots = (closures.capacity() as u64)
            .checked_mul(std::mem::size_of::<NonlinearSolidStudyResultClosure>() as u64)
            .ok_or_else(|| limit("original prefix slot storage overflow"))?;
        let caller = additional_retained_bytes
            .checked_add(observer_profile)
            .and_then(|n| n.checked_add(observer_wire))
            .and_then(|n| n.checked_add(slots))
            .ok_or_else(|| limit("original prefix caller storage overflow"))?;
        let preparation = NonlinearSolidStudyPreparation {
            local_execution: replay.local_execution,
            product: NonlinearSolidStudyProduct { compiled, workload },
            mesh: replay.mesh,
            accepted_block_limit: replay.accepted_block_limit,
            frozen_project_digest: replay.frozen_project_digest,
            program_revision: replay.program_revision,
            artifact_access: replay.artifact_access,
            output_policy: replay.output_policy,
        };
        let mut study = NonlinearSolidStudyCoordinator::new(preparation)?;
        retained::admit_pending(&mut study, caller, &closures)?;
        let block = import_prefix(source, &mut study, &closures, ordinal)?;
        Ok(OriginalNonlinearSolidBlockProjection {
            workload: study.preparation.product.workload,
            mesh: study.preparation.mesh,
            block,
            observations,
        })
    }
}

pub(super) fn import_prefix(
    source: &impl CacheImport,
    study: &mut NonlinearSolidStudyCoordinator,
    closures: &[NonlinearSolidStudyResultClosure],
    ordinal: u64,
) -> AnalysisBridgeResult<ImportedNonlinearSolidBlock> {
    workspace::validate(closures)?;
    study.import_workspace_bytes = Some(0);
    study.admit_retention(None, None)?;
    for closure in closures {
        let NonlinearSolidStudyResultPhase::Block { ordinal: original } = closure.phase else {
            break;
        };
        workspace::admit(source, study, closure)?;
        let task = study.prepare_block()?;
        if task.request.block_ordinal != original || task.host.workload != closure.workload {
            return Err(invalid("original nonlinear prefix phase/workload differs"));
        }
        study.complete_block(source, closure.response.clone(), closure.execution.clone())?;
        if original == ordinal {
            return Ok(study.block.take().expect("completed original block"));
        }
    }
    Err(invalid(
        "requested original nonlinear output block is absent",
    ))
}
