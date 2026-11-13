use super::*;
use runmat_analysis_execution::{
    StructuralBucklingStudyPreparation, StructuralPrestressedModalStudyPreparation,
};

pub(super) enum Branch {
    Buckling,
    Modal,
    Frequency,
    ModalFrequency,
}

// Field-only conversion keeps one execution context; no numerical model is projected here.
macro_rules! preparation {
    ($kind:ident, $source:ident) => {
        $kind {
            authored_study_digest: $source.authored_study_digest,
            frozen_project_digest: $source.frozen_project_digest,
            workloads: $source.workloads,
            mesh: $source.mesh,
            artifact_access: $source.artifact_access,
            program_revision: $source.program_revision,
            chunk_bytes: $source.chunk_bytes,
            inventory_limits: $source.inventory_limits,
            required_capabilities: $source.required_capabilities,
            accelerator_requests: $source.accelerator_requests,
            local_execution: $source.local_execution,
        }
    };
}

pub(super) async fn execute(
    pool: &mut impl AnalysisTaskPool,
    branch: Branch,
    preparation: StructuralPrestressedFrequencyStudyPreparation,
    policy: StructuralSequenceExecutionPolicy,
    source: BoundThermalExecution,
) -> NativeExecutionResult<ThermalPrestressStructuralCompletion> {
    Ok(match branch {
        Branch::Buckling => ThermalPrestressStructuralCompletion::Buckling(Box::new(
            Box::pin(crate::analysis_buckling::execute_with_thermal_source(
                pool,
                preparation!(StructuralBucklingStudyPreparation, preparation),
                policy,
                Some(source),
            ))
            .await?,
        )),
        Branch::Modal => ThermalPrestressStructuralCompletion::Modal(Box::new(
            Box::pin(
                crate::analysis_prestressed_modal::execute_with_thermal_source(
                    pool,
                    preparation!(StructuralPrestressedModalStudyPreparation, preparation),
                    policy,
                    Some(source),
                ),
            )
            .await?,
        )),
        Branch::Frequency => ThermalPrestressStructuralCompletion::Frequency(Box::new(
            Box::pin(
                crate::analysis_prestressed_frequency::execute_with_thermal_source(
                    pool,
                    preparation,
                    policy,
                    Some(source),
                ),
            )
            .await?,
        )),
        Branch::ModalFrequency => ThermalPrestressStructuralCompletion::ModalFrequency(Box::new(
            Box::pin(
                crate::analysis_prestressed_modal_frequency::execute_with_thermal_source(
                    pool,
                    preparation,
                    policy,
                    Some(source),
                ),
            )
            .await?,
        )),
    })
}
