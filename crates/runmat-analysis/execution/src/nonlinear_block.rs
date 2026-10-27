//! Portable original-history blocks use the shared host ABI and artifact authority.
use crate::*;
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind as Kind, AnalysisTransportProfiler, AnalysisWorkKind,
        ExecutionPlanIdentity, NonlinearSolidBlockArtifact, NonlinearSolidBlockRequest,
        NonlinearSolidWorkload, NumericalPlanIdentity, StudyIdentity,
    },
    planning::resource::{
        build_nonlinear_solid_block_resource_plan, nonlinear_solid_block_payload_bytes_upper_bound,
        FeaResourcePlan, LocalExecutionPlan, PreparedNonlinearSolidBlockResourceFacts,
    },
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use std::collections::BTreeSet;
mod admission;
mod output;
mod result;
pub use admission::require_nonlinear_solid_source_decode_budget;
pub use result::{import_nonlinear_solid_block, ImportedNonlinearSolidBlock};

pub struct NonlinearSolidBlockPreparation {
    pub local_execution: LocalExecutionPlan,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: NonlinearSolidWorkload,
    pub mesh: SolverMeshArtifact,
    pub request: NonlinearSolidBlockRequest,
    pub predecessor: Option<PreparedAnalysisObjects>,
    pub output_policy: AnalysisOutputPolicy,
}
pub struct PreparedNonlinearSolidBlockTask {
    pub request: NonlinearSolidBlockRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
}
fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
fn objects(
    kind: Kind,
    id: Digest,
    bytes: &[u8],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    prepare_analysis_objects(
        kind,
        id,
        bytes,
        policy.chunk_bytes(),
        if kind == Kind::NonlinearSolidBlockArtifact {
            ObjectNamespace::ResultValue
        } else {
            ObjectNamespace::InputValue
        },
        policy.inventory_limits(),
    )
}
fn predecessor(
    objects: Option<&PreparedAnalysisObjects>,
) -> AnalysisBridgeResult<Option<NonlinearSolidBlockArtifact>> {
    objects
        .map(|input| {
            crate::portable_source_admission::authenticate_source_objects(
                input,
                Kind::NonlinearSolidBlockArtifact,
            )?;
            Ok(NonlinearSolidBlockArtifact::from_canonical_bytes(
                &input.payload(),
                input.manifest.logical_identity,
            )?)
        })
        .transpose()
}
fn plan(
    workload: &NonlinearSolidWorkload,
    mesh: &SolverMeshArtifact,
    request: &NonlinearSolidBlockRequest,
    previous: Option<&NonlinearSolidBlockArtifact>,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    local_execution: LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let bytes = nonlinear_solid_block_payload_bytes_upper_bound(workload, mesh, request)?;
    Ok(build_nonlinear_solid_block_resource_plan(
        workload,
        mesh,
        request,
        previous,
        PreparedNonlinearSolidBlockResourceFacts {
            local_execution,
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: output::project(Kind::NonlinearSolidBlockArtifact, bytes, policy)?,
            receipt_output: output::project(
                Kind::NumericalReceipt,
                runmat_analysis_fea::execution::NONLINEAR_SOLID_RECEIPT_MAX_ENCODED_BYTES,
                policy,
            )?,
            output_set: projected_output_set_resource_facts(),
        },
    )?)
}
pub fn prepare_nonlinear_solid_block_task(
    p: NonlinearSolidBlockPreparation,
) -> AnalysisBridgeResult<PreparedNonlinearSolidBlockTask> {
    p.workload.validate()?;
    if let Some(input) = &p.predecessor {
        runmat_execution_artifact::object::validate_inventory(
            &input.objects,
            p.output_policy.inventory_limits(),
        )?;
        crate::portable_source_admission::authenticate_source_objects(
            input,
            Kind::NonlinearSolidBlockArtifact,
        )?;
        let facts = prepared_input_resource_facts(std::slice::from_ref(input))?;
        if facts.decoded_resident_bytes_admission_budget.0
            > p.workload.model.sequence.steps[0]
                .accuracy
                .resources
                .resident_byte_budget()
        {
            return Err(invalid(
                "nonlinear predecessor exceeds original decode ceiling",
            ));
        }
    }
    let previous = predecessor(p.predecessor.as_ref())?;
    p.request.validate_for(
        &p.workload,
        Digest::from_bytes(*p.mesh.canonical_digest.bytes()),
        previous.as_ref(),
    )?;
    let policy = p.output_policy;
    let mut inputs = vec![
        objects(
            Kind::NonlinearSolidWorkload,
            p.workload.identity()?.digest(),
            &p.workload.canonical_bytes()?,
            policy,
        )?,
        objects(
            Kind::Mesh,
            p.request.topology,
            &p.mesh.canonical_encode().map_err(invalid)?,
            policy,
        )?,
        objects(
            Kind::NonlinearSolidBlockRequest,
            p.request.identity()?,
            &p.request.canonical_bytes()?,
            policy,
        )?,
    ];
    if let Some(input) = p.predecessor {
        inputs.push(input);
    }
    let resource_plan = plan(
        &p.workload,
        &p.mesh,
        &p.request,
        previous.as_ref(),
        &inputs,
        policy,
        p.local_execution,
    )?;
    inputs.push(objects(
        Kind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        policy,
    )?);
    let execution = ExecutionPlanIdentity::new(
        NumericalPlanIdentity::from_nonlinear_solid_block(&p.request)?,
        resource_plan.identity.digest(),
    )?;
    let request = AnalysisWorkloadRequest::new(
        StudyIdentity::from_authored_digest(p.authored_study_digest, p.frozen_project_digest)?,
        p.workload.producer_identity()?,
        execution,
        AnalysisWorkKind::SolveNonlinearSolidBlock,
        inputs
            .iter()
            .map(|input| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: if input.manifest.kind == Kind::NonlinearSolidBlockArtifact {
                    ValueRefKind::ResultObject
                } else {
                    ValueRefKind::DriverObject
                },
            })
            .collect(),
        vec![Kind::NonlinearSolidBlockArtifact, Kind::NumericalReceipt],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(request, p.artifact_access, p.output_policy)?;
    let program = prepare_analysis_program(&host, p.program_revision, &inputs)?;
    Ok(PreparedNonlinearSolidBlockTask {
        request: p.request,
        host,
        program,
        inputs,
        resource_plan,
        output_policy: policy,
    })
}

/// Reconstruct the exact plan from authenticated original inputs and output policy.
pub fn rederive_nonlinear_solid_block_resource_plan(
    workload: &NonlinearSolidWorkload,
    mesh: &SolverMeshArtifact,
    request: &NonlinearSolidBlockRequest,
    previous: Option<&NonlinearSolidBlockArtifact>,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    for input in inputs {
        let (identity, bytes) = match input.manifest.kind {
            Kind::NonlinearSolidWorkload => {
                (workload.identity()?.digest(), workload.canonical_bytes()?)
            }
            Kind::Mesh => (
                Digest::from_bytes(*mesh.canonical_digest.bytes()),
                mesh.canonical_encode().map_err(invalid)?,
            ),
            Kind::NonlinearSolidBlockRequest => (request.identity()?, request.canonical_bytes()?),
            Kind::NonlinearSolidBlockArtifact => {
                let previous =
                    previous.ok_or_else(|| invalid("unexpected nonlinear predecessor"))?;
                (previous.identity, previous.canonical_bytes()?)
            }
            Kind::ResourcePlan => continue,
            _ => return Err(invalid("unexpected nonlinear block input")),
        };
        if input.manifest.kind == Kind::NonlinearSolidBlockArtifact {
            // A retained predecessor keeps its producer's original chunk policy.
            // Authenticate that closure rather than rechunking it under the next task.
            crate::portable_source_admission::authenticate_source_objects(
                input,
                Kind::NonlinearSolidBlockArtifact,
            )?;
            if input.manifest.logical_identity != identity
                || input.manifest.payload_digest != Digest::sha256(&bytes)
                || input.manifest.encoded_length != bytes.len() as u64
            {
                return Err(invalid(
                    "nonlinear predecessor differs from original canonical content",
                ));
            }
            continue;
        }
        if input != &objects(input.manifest.kind, identity, &bytes, policy)? {
            return Err(invalid(
                "nonlinear source closure differs from original canonical inputs",
            ));
        }
    }
    let carried = inputs
        .iter()
        .find(|input| input.manifest.kind == Kind::ResourcePlan)
        .ok_or_else(|| invalid("nonlinear resource input missing"))?;
    let declared = FeaResourcePlan::from_canonical_bytes(&carried.payload())?;
    let rederived = plan(
        workload,
        mesh,
        request,
        previous,
        inputs,
        policy,
        LocalExecutionPlan::from_cpu_millicores(declared.request.cpu_millicores)
            .map_err(invalid)?,
    )?;
    if carried
        != &objects(
            Kind::ResourcePlan,
            rederived.identity.digest(),
            &rederived.canonical_bytes()?,
            policy,
        )?
    {
        return Err(invalid(
            "nonlinear resource authority differs from actual inputs",
        ));
    }
    require_nonlinear_solid_source_decode_budget(inputs, &rederived, policy.inventory_limits())?;
    Ok(rederived)
}
