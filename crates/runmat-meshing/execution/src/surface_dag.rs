//! Meshing-owned geometric dependencies for exact surface convergence.
//!
//! This planner produces immutable host workloads and canonically aligned roots. It deliberately
//! has no tasks, attempts, workers, placement, retry, or lifecycle state; those remain owned by
//! shared execution after callers project each planned stage onto their chosen backend.

use runmat_execution::value::ValueRef;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingCapabilityRequirement, MeshingInputKind, MeshingInputRef,
    MeshingPartitionDescriptor, MeshingPartitionKind, MeshingStageIdentity, MeshingStageKind,
    MeshingWorkloadRequest, StableDigest, MESHING_IDENTITY_SCHEMA_VERSION,
    MESHING_WORKLOAD_SCHEMA_VERSION,
};
use runmat_meshing_surface::{face_partition_descriptors, MAX_EXACT_FACE_PARTITIONS};

use crate::task::{validate_input, validate_inputs};
use crate::{MeshingExecutionError, MeshingExecutionResult, MeshingHostWorkload};

/// One meshing-owned stage with roots ordered exactly like its canonical prerequisites.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedMeshingStage {
    host: MeshingHostWorkload,
    input_roots: Vec<ValueRef>,
}

impl PlannedMeshingStage {
    pub const fn host(&self) -> &MeshingHostWorkload {
        &self.host
    }

    pub fn input_roots(&self) -> &[ValueRef] {
        &self.input_roots
    }

    pub fn program_request(
        &self,
        revision: ProgramRevision,
    ) -> MeshingExecutionResult<ProgramExecutionRequest> {
        self.host.program_request(revision, &self.input_roots)
    }
}

/// A complete deterministic face-partition pass bound to one current shared curve.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactSurfacePassPlan {
    context: SurfaceDagContext,
    pass_index: u32,
    curve_root: ValueRef,
    partitions: Vec<PlannedMeshingStage>,
}

impl ExactSurfacePassPlan {
    pub const fn pass_index(&self) -> u32 {
        self.pass_index
    }

    pub const fn curve_root(&self) -> &ValueRef {
        &self.curve_root
    }

    pub fn partitions(&self) -> &[PlannedMeshingStage] {
        &self.partitions
    }
}

/// Deterministic exact-surface DAG construction over one admitted geometry and request.
#[derive(Clone, Debug)]
pub struct ExactSurfaceDagPlanner {
    seed: MeshingHostWorkload,
    geometry_root: ValueRef,
    context: SurfaceDagContext,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SurfaceDagContext {
    geometry_digest: StableDigest,
    request_digest: StableDigest,
    authorization_scope: String,
    encryption_context: runmat_execution::Digest,
    capability_cohort: Option<String>,
}

impl ExactSurfaceDagPlanner {
    /// Seeds surface planning from a validated host that consumes the admitted exact geometry.
    ///
    /// Reusing the seed preserves the already admitted host ABI, exact-kernel ABI, element order,
    /// deterministic cohort, artifact authority, geometry revision, and resolved request.
    pub fn from_exact_host(
        seed: &MeshingHostWorkload,
        geometry_root: ValueRef,
    ) -> MeshingExecutionResult<Self> {
        seed.validate()?;
        let exact_inputs = seed
            .workload
            .inputs
            .iter()
            .filter(|input| input.kind == MeshingInputKind::ExactGeometry)
            .collect::<Vec<_>>();
        let [exact_input] = exact_inputs.as_slice() else {
            return Err(invalid(
                "exact surface DAG seed must have one authoritative exact-geometry input",
            ));
        };
        if seed.geometry_document.is_none() {
            return Err(invalid(
                "exact surface DAG seed must carry its authoritative geometry document",
            ));
        }
        validate_input(&geometry_root, exact_input, &seed.artifact_access)?;
        validate_seed_capabilities(seed)?;
        let context = SurfaceDagContext {
            geometry_digest: exact_input.digest,
            request_digest: seed.resolved_request.canonical_digest()?,
            authorization_scope: seed.artifact_access.authorization_scope.clone(),
            encryption_context: seed.artifact_access.encryption_context,
            capability_cohort: seed.stage_identity.capability_cohort.clone(),
        };
        Ok(Self {
            seed: seed.clone(),
            geometry_root,
            context,
        })
    }

    pub fn begin_surface_pass(
        &self,
        topology: &ExactBRepTopology,
        curve_root: ValueRef,
        preferred_faces_per_partition: u32,
    ) -> MeshingExecutionResult<ExactSurfacePassPlan> {
        self.build_surface_pass(topology, 0, curve_root, preferred_faces_per_partition)
    }

    pub fn next_surface_pass(
        &self,
        previous: &ExactSurfacePassPlan,
        topology: &ExactBRepTopology,
        refined_curve_root: ValueRef,
        preferred_faces_per_partition: u32,
    ) -> MeshingExecutionResult<ExactSurfacePassPlan> {
        self.validate_pass(previous)?;
        if refined_curve_root.logical_digest == previous.curve_root.logical_digest {
            return Err(invalid(
                "a restarted surface pass requires a newly refined shared curve",
            ));
        }
        let pass_index = previous
            .pass_index
            .checked_add(1)
            .ok_or_else(|| invalid("surface pass index overflowed"))?;
        if pass_index >= self.seed.resolved_request.resources.maximum_recursion_depth {
            return Err(invalid(
                "surface convergence exhausted the resolved recursion budget",
            ));
        }
        self.build_surface_pass(
            topology,
            pass_index,
            refined_curve_root,
            preferred_faces_per_partition,
        )
    }

    pub fn surface_join(
        &self,
        pass: &ExactSurfacePassPlan,
        partition_roots: Vec<ValueRef>,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        self.validate_partition_roots(pass, &partition_roots)?;
        let mut roots = Vec::with_capacity(2 + partition_roots.len());
        roots.push(self.geometry_root.clone());
        roots.push(pass.curve_root.clone());
        roots.extend(partition_roots);
        self.build_stage(
            MeshingStageKind::SurfaceMesh,
            whole_partition(MeshingPartitionKind::DeterministicJoin),
            roots,
        )
    }

    pub fn curve_refinement(
        &self,
        pass: &ExactSurfacePassPlan,
        partition_roots: Vec<ValueRef>,
        surface_pass_root: ValueRef,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        self.validate_partition_roots(pass, &partition_roots)?;
        let mut roots = Vec::with_capacity(3 + partition_roots.len());
        roots.push(self.geometry_root.clone());
        roots.push(pass.curve_root.clone());
        roots.extend(partition_roots);
        roots.push(surface_pass_root);
        self.build_stage(
            MeshingStageKind::CurveMesh,
            whole_partition(MeshingPartitionKind::WholeStage),
            roots,
        )
    }

    fn build_surface_pass(
        &self,
        topology: &ExactBRepTopology,
        pass_index: u32,
        curve_root: ValueRef,
        preferred_faces_per_partition: u32,
    ) -> MeshingExecutionResult<ExactSurfacePassPlan> {
        let descriptors = face_partition_descriptors(topology, preferred_faces_per_partition)
            .map_err(|error| invalid(error.to_string()))?;
        let partitions = descriptors
            .into_iter()
            .map(|partition| {
                self.build_stage(
                    MeshingStageKind::SurfaceMesh,
                    partition,
                    vec![self.geometry_root.clone(), curve_root.clone()],
                )
            })
            .collect::<MeshingExecutionResult<Vec<_>>>()?;
        Ok(ExactSurfacePassPlan {
            context: self.context.clone(),
            pass_index,
            curve_root,
            partitions,
        })
    }

    fn validate_partition_roots(
        &self,
        pass: &ExactSurfacePassPlan,
        roots: &[ValueRef],
    ) -> MeshingExecutionResult<()> {
        self.validate_pass(pass)?;
        if roots.len() != pass.partitions.len()
            || roots.is_empty()
            || roots.len() > MAX_EXACT_FACE_PARTITIONS
        {
            return Err(invalid(
                "surface barrier requires one result for every planned face partition",
            ));
        }
        Ok(())
    }

    fn validate_pass(&self, pass: &ExactSurfacePassPlan) -> MeshingExecutionResult<()> {
        if pass.context != self.context {
            return Err(invalid(
                "surface pass belongs to a different geometry, request, or artifact authority",
            ));
        }
        Ok(())
    }

    fn build_stage(
        &self,
        stage: MeshingStageKind,
        partition: MeshingPartitionDescriptor,
        roots: Vec<ValueRef>,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        let mut dependencies = roots
            .into_iter()
            .map(|root| {
                let kind = if root.logical_digest.bytes() == self.context.geometry_digest.bytes() {
                    MeshingInputKind::ExactGeometry
                } else {
                    MeshingInputKind::StageArtifact
                };
                (
                    MeshingInputRef {
                        kind,
                        digest: StableDigest::from_bytes(*root.logical_digest.bytes()),
                    },
                    root,
                )
            })
            .collect::<Vec<_>>();
        dependencies.sort_by(|left, right| left.0.cmp(&right.0));
        let inputs = dependencies
            .iter()
            .map(|(input, _)| input.clone())
            .collect::<Vec<_>>();
        let input_roots = dependencies
            .into_iter()
            .map(|(_, root)| root)
            .collect::<Vec<_>>();
        let identity = MeshingStageIdentity {
            schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
            stage,
            geometry: self.seed.stage_identity.geometry.clone(),
            resolved_request_digest: self.context.request_digest,
            tolerance_policy_digest: self.seed.stage_identity.tolerance_policy_digest,
            metric_policy_digest: self.seed.stage_identity.metric_policy_digest,
            algorithm_set_digest: self.seed.stage_identity.algorithm_set_digest,
            deterministic_seed: self.seed.resolved_request.deterministic_seed,
            prerequisites: inputs.clone(),
            capability_cohort: self.context.capability_cohort.clone(),
        };
        let workload = MeshingWorkloadRequest {
            schema_version: MESHING_WORKLOAD_SCHEMA_VERSION,
            stage,
            stage_identity_digest: identity.canonical_digest()?,
            partition,
            inputs,
            required_capabilities: capabilities_for_stage(&self.seed, stage)?,
        };
        let host = MeshingHostWorkload::new(
            workload,
            identity,
            self.seed.resolved_request.clone(),
            self.seed.artifact_access.clone(),
            self.seed.geometry_document.clone(),
        )?;
        validate_inputs(&host.workload, &input_roots, &host.artifact_access)?;
        Ok(PlannedMeshingStage { host, input_roots })
    }
}

fn validate_seed_capabilities(seed: &MeshingHostWorkload) -> MeshingExecutionResult<()> {
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
            "exact surface DAG seed capabilities are incomplete or ambiguous",
        ));
    }
    Ok(())
}

fn capabilities_for_stage(
    seed: &MeshingHostWorkload,
    stage: MeshingStageKind,
) -> MeshingExecutionResult<Vec<MeshingCapabilityRequirement>> {
    let version = match stage {
        MeshingStageKind::CurveMesh => &seed.resolved_request.algorithms.curve,
        MeshingStageKind::SurfaceMesh => &seed.resolved_request.algorithms.surface,
        _ => return Err(invalid("exact surface DAG received an unsupported stage")),
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

fn whole_partition(kind: MeshingPartitionKind) -> MeshingPartitionDescriptor {
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
