//! Meshing-owned geometric dependencies for exact curve, surface, and volume construction.
//!
//! This planner produces immutable host workloads and canonically aligned roots. It deliberately
//! has no tasks, attempts, workers, placement, retry, or lifecycle state; those remain owned by
//! shared execution after callers project each planned stage onto their chosen backend.

use runmat_execution::value::ValueRef;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingInputKind, MeshingInputRef, MeshingPartitionDescriptor,
    MeshingPartitionKind, MeshingStageIdentity, MeshingStageKind, MeshingWorkloadRequest,
    StableDigest, MESHING_IDENTITY_SCHEMA_VERSION, MESHING_WORKLOAD_SCHEMA_VERSION,
};
use runmat_meshing_curve::curve_partition_descriptors;
use runmat_meshing_surface::{face_partition_descriptors, MAX_EXACT_FACE_PARTITIONS};

use crate::task::{validate_input, validate_inputs};
use crate::{MeshingExecutionError, MeshingExecutionResult, MeshingHostWorkload};

mod stage;

use stage::{capabilities_for_stage, validate_seed_capabilities, whole_partition};

/// One meshing-owned stage with roots ordered exactly like its canonical prerequisites.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedMeshingStage {
    host: MeshingHostWorkload,
    input_roots: Vec<ValueRef>,
}

/// A complete deterministic edge-partition pass for the initial shared curve mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactCurvePassPlan {
    context: ExactDagContext,
    partitions: Vec<PlannedMeshingStage>,
}

impl ExactCurvePassPlan {
    pub fn partitions(&self) -> &[PlannedMeshingStage] {
        &self.partitions
    }
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
    context: ExactDagContext,
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

/// Deterministic exact-meshing DAG construction over one admitted geometry and request.
#[derive(Clone, Debug)]
pub struct ExactMeshingDagPlanner {
    seed: MeshingHostWorkload,
    geometry_root: ValueRef,
    context: ExactDagContext,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExactDagContext {
    geometry_digest: StableDigest,
    request_digest: StableDigest,
    authorization_scope: String,
    encryption_context: runmat_execution::Digest,
    capability_cohort: Option<String>,
}

impl ExactMeshingDagPlanner {
    /// Seeds exact meshing from a validated host that consumes the admitted exact geometry.
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
                "exact meshing DAG seed must have one authoritative exact-geometry input",
            ));
        };
        if seed.geometry_document.is_none() {
            return Err(invalid(
                "exact meshing DAG seed must carry its authoritative geometry document",
            ));
        }
        validate_input(&geometry_root, exact_input, &seed.artifact_access)?;
        validate_seed_capabilities(seed)?;
        let context = ExactDagContext {
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

    /// Plans the canonical edge batches that construct the first shared curve mesh.
    pub fn initial_curve_pass(
        &self,
        topology: &ExactBRepTopology,
        preferred_edges_per_partition: u32,
    ) -> MeshingExecutionResult<ExactCurvePassPlan> {
        let descriptors = curve_partition_descriptors(topology, preferred_edges_per_partition)
            .map_err(|error| invalid(error.to_string()))?;
        let partitions = descriptors
            .into_iter()
            .map(|partition| {
                self.build_stage(
                    MeshingStageKind::CurveMesh,
                    partition,
                    vec![self.geometry_root.clone()],
                )
            })
            .collect::<MeshingExecutionResult<Vec<_>>>()?;
        Ok(ExactCurvePassPlan {
            context: self.context.clone(),
            partitions,
        })
    }

    /// Plans the global shared-curve join independently of partition completion order.
    pub fn curve_join(
        &self,
        pass: &ExactCurvePassPlan,
        partition_roots: Vec<ValueRef>,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        if pass.context != self.context {
            return Err(invalid(
                "curve pass belongs to a different geometry, request, or artifact authority",
            ));
        }
        if partition_roots.len() != pass.partitions.len() || partition_roots.is_empty() {
            return Err(invalid(
                "curve barrier requires one result for every planned edge partition",
            ));
        }
        let mut roots = Vec::with_capacity(1 + partition_roots.len());
        roots.push(self.geometry_root.clone());
        roots.extend(partition_roots);
        self.build_stage(
            MeshingStageKind::CurveMesh,
            whole_partition(MeshingPartitionKind::DeterministicJoin),
            roots,
        )
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

    /// Plans the connected general-CDT stage from the final exact-surface publication. The
    /// kernel independently validates that root as the deterministic join for this geometry.
    pub fn tetrahedralization(
        &self,
        surface_root: ValueRef,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        if surface_root.logical_digest == self.geometry_root.logical_digest {
            return Err(invalid(
                "tetrahedralization requires a distinct final exact-surface artifact",
            ));
        }
        self.build_stage(
            MeshingStageKind::Tetrahedralization,
            whole_partition(MeshingPartitionKind::WholeStage),
            vec![self.geometry_root.clone(), surface_root],
        )
    }

    /// Plans the canonical Tet4/Tet10 solver projection from independently admitted terminal
    /// geometry, surface, volume, and domain-model roots.
    pub fn solver_projection(
        &self,
        surface_root: ValueRef,
        volume_root: ValueRef,
        domain_model_root: ValueRef,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        if surface_root.logical_digest == volume_root.logical_digest
            || [surface_root.logical_digest, volume_root.logical_digest]
                .contains(&domain_model_root.logical_digest)
        {
            return Err(invalid(
                "solver projection requires distinct surface, volume, and domain-model roots",
            ));
        }
        self.build_stage_with_dependencies(
            MeshingStageKind::OrderElevation,
            whole_partition(MeshingPartitionKind::WholeStage),
            vec![
                (MeshingInputKind::ExactGeometry, self.geometry_root.clone()),
                (MeshingInputKind::StageArtifact, surface_root),
                (MeshingInputKind::StageArtifact, volume_root),
                (MeshingInputKind::DomainModel, domain_model_root),
            ],
        )
    }

    pub fn solver_validation(
        &self,
        projection_root: ValueRef,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        self.build_stage_with_dependencies(
            MeshingStageKind::Validation,
            whole_partition(MeshingPartitionKind::WholeStage),
            vec![(MeshingInputKind::StageArtifact, projection_root)],
        )
    }

    pub fn solver_serialization(
        &self,
        projection_root: ValueRef,
        validation_root: ValueRef,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        if projection_root.logical_digest == validation_root.logical_digest {
            return Err(invalid(
                "solver serialization requires distinct projection and validation roots",
            ));
        }
        self.build_stage_with_dependencies(
            MeshingStageKind::Serialization,
            whole_partition(MeshingPartitionKind::WholeStage),
            vec![
                (MeshingInputKind::StageArtifact, projection_root),
                (MeshingInputKind::StageArtifact, validation_root),
            ],
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
        let dependencies = roots
            .into_iter()
            .map(|root| {
                let kind = if root.logical_digest.bytes() == self.context.geometry_digest.bytes() {
                    MeshingInputKind::ExactGeometry
                } else {
                    MeshingInputKind::StageArtifact
                };
                (kind, root)
            })
            .collect::<Vec<_>>();
        self.build_stage_with_dependencies(stage, partition, dependencies)
    }

    fn build_stage_with_dependencies(
        &self,
        stage: MeshingStageKind,
        partition: MeshingPartitionDescriptor,
        dependencies: Vec<(MeshingInputKind, ValueRef)>,
    ) -> MeshingExecutionResult<PlannedMeshingStage> {
        let mut dependencies = dependencies
            .into_iter()
            .map(|(kind, root)| {
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
        let geometry_document = if inputs
            .iter()
            .any(|input| input.kind == MeshingInputKind::ExactGeometry)
        {
            self.seed.geometry_document.clone()
        } else {
            None
        };
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
            geometry_document,
        )?;
        validate_inputs(&host.workload, &input_roots, &host.artifact_access)?;
        Ok(PlannedMeshingStage { host, input_roots })
    }
}

fn invalid(reason: impl Into<String>) -> MeshingExecutionError {
    MeshingExecutionError::Invalid(reason.into())
}
