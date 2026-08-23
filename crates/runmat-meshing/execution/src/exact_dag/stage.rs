use runmat_meshing_core::{
    MeshingCapabilityRequirement, MeshingPartitionDescriptor, MeshingPartitionKind,
    MeshingStageKind,
};

use crate::{MeshingExecutionError, MeshingExecutionResult, MeshingHostWorkload};

pub(super) fn validate_seed_capabilities(seed: &MeshingHostWorkload) -> MeshingExecutionResult<()> {
    let mut host = 0;
    let mut exact = 0;
    let mut algorithm = 0;
    let mut order = 0;
    let mut cohort = 0;
    for capability in &seed.workload.required_capabilities {
        match capability {
            MeshingCapabilityRequirement::HostWorkload { .. } => host += 1,
            MeshingCapabilityRequirement::ExactCadKernel { .. } => exact += 1,
            MeshingCapabilityRequirement::MeshingAlgorithm { .. } => algorithm += 1,
            MeshingCapabilityRequirement::ElementOrder { .. } => order += 1,
            MeshingCapabilityRequirement::DeterministicPlatformCohort { .. } => cohort += 1,
        }
    }
    let expected_cohort = usize::from(seed.stage_identity.capability_cohort.is_some());
    if (host, exact, algorithm, order, cohort) != (1, 1, 1, 1, expected_cohort) {
        return Err(invalid(
            "exact meshing DAG seed capabilities are incomplete or ambiguous",
        ));
    }
    Ok(())
}

pub(super) fn capabilities_for_stage(
    seed: &MeshingHostWorkload,
    stage: MeshingStageKind,
) -> MeshingExecutionResult<Vec<MeshingCapabilityRequirement>> {
    let version = match stage {
        MeshingStageKind::CurveMesh => &seed.resolved_request.algorithms.curve,
        MeshingStageKind::SurfaceMesh => &seed.resolved_request.algorithms.surface,
        MeshingStageKind::Tetrahedralization => &seed.resolved_request.algorithms.tetrahedron,
        MeshingStageKind::OrderElevation => &seed.resolved_request.algorithms.optimization,
        _ => return Err(invalid("exact meshing DAG received an unsupported stage")),
    };
    let mut capabilities = seed
        .workload
        .required_capabilities
        .iter()
        .map(|capability| match capability {
            MeshingCapabilityRequirement::MeshingAlgorithm { .. } => {
                MeshingCapabilityRequirement::MeshingAlgorithm {
                    version: version.clone(),
                }
            }
            capability => capability.clone(),
        })
        .collect::<Vec<_>>();
    capabilities.sort();
    Ok(capabilities)
}

pub(super) fn whole_partition(kind: MeshingPartitionKind) -> MeshingPartitionDescriptor {
    MeshingPartitionDescriptor {
        kind,
        partition_index: 0,
        partition_count: 1,
        entity_range: None,
    }
}

fn invalid(reason: impl Into<String>) -> MeshingExecutionError {
    MeshingExecutionError::Invalid(reason.into())
}
