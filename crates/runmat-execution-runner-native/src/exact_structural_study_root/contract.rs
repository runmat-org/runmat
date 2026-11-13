mod decode;
mod preparation;
mod validation;
pub use decode::decode_exact_structural_study_root;
pub use preparation::prepare_exact_structural_study_root;
pub(crate) use validation::structural_workload_inventory_digest;
pub(super) use validation::validate_analysis_accelerator_requests;
use validation::validate_input_materialization_inventory;

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_execution::{
    prepare_analysis_host_program, AnalysisHostProgramArtifact, InputMaterializationPlan,
    StructuralStudyWorkloads,
};
use runmat_analysis_fea::planning::resource::LocalExecutionPlan;
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, OutputContract, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactAccess, ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionDescriptor, ProgramExecutionObject,
    ProgramExecutionRequest, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_meshing_core::MeshingRequest;
use runmat_meshing_execution::MeshingHostProgram;
use serde::{Deserialize, Serialize};

use super::{
    protocol, read, CONTRACT_LIMITS, EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE,
    EXACT_STRUCTURAL_STUDY_ROOT_TARGET_PROFILE, ROOT_DOMAIN, ROOT_MEDIA_TYPE, ROOT_PREFIX,
    ROOT_PROGRAM_BYTES, ROOT_VALUE_SCHEMA,
};
use crate::input_materialization_contract::InputMaterializationPlanRequest;
use crate::{decode_exact_meshing_root, NativeExecutionResult, PreparedExactMeshingRoot};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactStructuralStudyRootRequest {
    schema_version: u16,
    authored_study_digest: Digest,
    frozen_project_digest: Option<Digest>,
    geometry_source_digest: Digest,
    workloads: StructuralStudyWorkloads,
    local_execution: LocalExecutionPlan,
    analysis_accelerator_requests: Vec<AcceleratorRequest>,
    input_materializations: Vec<InputMaterializationPlanRequest>,
    exact_meshing: ProgramExecutionDescriptor,
    exact_argument_count: u32,
    materialization_argument_count: u32,
}

impl ExactStructuralStudyRootRequest {
    fn validate(&self) -> NativeExecutionResult<()> {
        self.workloads.validate().map_err(protocol)?;
        validate_analysis_accelerator_requests(
            &self.workloads,
            &self.analysis_accelerator_requests,
        )?;
        self.exact_meshing.validate().map_err(protocol)?;
        let prepared_inputs = self
            .input_materializations
            .iter()
            .map(InputMaterializationPlanRequest::to_plan)
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        validate_input_materialization_inventory(
            &self.workloads,
            &prepared_inputs,
            &self.exact_meshing.recipe.program_revision,
        )?;
        let materialization_reference_count = prepared_inputs
            .iter()
            .flat_map(|prepared| prepared.input_references().iter())
            .map(|reference| reference.id)
            .collect::<BTreeSet<_>>()
            .len();
        if self.schema_version != 4
            || self.authored_study_digest.bytes() == &[0; 32]
            || self.geometry_source_digest.bytes() == &[0; 32]
            || self
                .frozen_project_digest
                .is_some_and(|digest| digest.bytes() == &[0; 32])
            || self.exact_argument_count < 2
            || usize::try_from(self.materialization_argument_count).map_err(protocol)?
                != materialization_reference_count
            || self.exact_meshing.recipe.execution_mode != crate::EXACT_MESHING_ROOT_EXECUTION_MODE
            || self.exact_meshing.artifact.form != ExecutableForm::MeshingWorkload
            || self.exact_meshing.requested_outputs != 1
        {
            return Err(protocol("exact structural study root is invalid"));
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> NativeExecutionResult<Vec<u8>> {
        self.validate()?;
        encode_contract(ROOT_PREFIX, ROOT_DOMAIN, self, CONTRACT_LIMITS).map_err(protocol)
    }

    fn decode(bytes: &[u8]) -> NativeExecutionResult<Self> {
        let root =
            decode_contract(ROOT_PREFIX, ROOT_DOMAIN, bytes, CONTRACT_LIMITS).map_err(protocol)?;
        Self::validate(&root)?;
        Ok(root)
    }
}

pub struct ExactStructuralStudyRootPreparation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub geometry_source_digest: Digest,
    pub workloads: StructuralStudyWorkloads,
    pub local_execution: LocalExecutionPlan,
    pub analysis_accelerator_requests: Vec<AcceleratorRequest>,
    pub input_materializations: Vec<InputMaterializationPlan>,
    pub exact_meshing: PreparedExactMeshingRoot,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequiredStudyProgram {
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedExactStructuralStudyRoot {
    request: ProgramExecutionRequest,
    objects: Vec<LogicalObject>,
    input_value_objects: Vec<ProgramExecutionObject>,
    root: ExactStructuralStudyRootRequest,
    artifact_access: ArtifactAccess,
    required_programs: Vec<RequiredStudyProgram>,
    expected_meshing_request: MeshingRequest,
    input_expectations: Vec<runmat_analysis_execution::InputMaterializationPlanExpectation>,
}

impl PreparedExactStructuralStudyRoot {
    pub const fn request(&self) -> &ProgramExecutionRequest {
        &self.request
    }

    pub fn objects(&self) -> &[LogicalObject] {
        &self.objects
    }

    pub fn transport_objects(&self) -> NativeExecutionResult<Vec<ProgramExecutionObject>> {
        let references = object_references(&self.request.arguments)?;
        let mut transported = Vec::with_capacity(self.objects.len());
        for object in &self.objects {
            let reference = references
                .get(&object.descriptor.digest)
                .ok_or_else(|| protocol("study root contains an unbound input object"))?
                .clone();
            let object = ProgramExecutionObject {
                reference,
                bytes: object.bytes.clone(),
            };
            object.validate().map_err(protocol)?;
            transported.push(object);
        }
        for object in &self.input_value_objects {
            let reference = references
                .get(&object.reference.logical_digest)
                .ok_or_else(|| protocol("study root omits a signal input value reference"))?;
            if reference != &object.reference {
                return Err(protocol(
                    "study root signal input value differs from its request reference",
                ));
            }
            object.validate().map_err(protocol)?;
            transported.push(object.clone());
        }
        if transported.len() != references.len() {
            return Err(protocol(
                "study root input references differ from its object inventory",
            ));
        }
        Ok(transported)
    }

    pub fn required_programs(&self) -> NativeExecutionResult<Vec<RequiredStudyProgram>> {
        Ok(self.required_programs.clone())
    }

    pub fn write_to(&self, store: &mut impl CacheExport) -> NativeExecutionResult<()> {
        for object in &self.objects {
            store.write_verified(object).map_err(protocol)?;
        }
        Ok(())
    }

    pub fn result_expectation(&self) -> super::publication::ExactStructuralStudyResultExpectation {
        super::publication::ExactStructuralStudyResultExpectation {
            program_revision: self.request.recipe.program_revision.clone(),
            artifact_access: self.artifact_access.clone(),
            authored_study_digest: self.root.authored_study_digest,
            frozen_project_digest: self.root.frozen_project_digest,
            geometry_source_digest: self.root.geometry_source_digest,
            workloads: self.root.workloads.clone(),
            local_execution: self.root.local_execution,
            analysis_accelerator_requests: self.root.analysis_accelerator_requests.clone(),
            input_materializations: self.input_expectations.clone(),
            expected_meshing_request: self.expected_meshing_request.clone(),
        }
    }

    pub fn into_request(self) -> ProgramExecutionRequest {
        self.request
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedExactStructuralStudyRoot {
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub geometry_source_digest: Digest,
    pub workloads: StructuralStudyWorkloads,
    pub local_execution: LocalExecutionPlan,
    /// The execution policy's `memory_per_worker` from the typed study root, when authored.
    pub memory_per_worker_bytes: Option<u64>,
    pub analysis_accelerator_requests: Vec<AcceleratorRequest>,
    pub input_materializations: Vec<InputMaterializationPlan>,
    pub exact_meshing: ProgramExecutionRequest,
    pub meshing_request: MeshingRequest,
    pub artifact_access: ArtifactAccess,
    /// Present when the study was compiled from source-neutral geometry after meshing.
    /// Pre-meshed low-level roots intentionally have no authored compilation product.
    pub compiled_study: Option<runmat_analysis_fea::study::CompiledAnalysisStudy>,
    pub compiled_geometry_output: Option<runmat_analysis_execution::GeometryMaterializationOutput>,
}

fn descriptor(request: &ProgramExecutionRequest) -> ProgramExecutionDescriptor {
    ProgramExecutionDescriptor {
        console: request.console,
        environment: request.environment.clone(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe: request.recipe.clone(),
        artifact: request.artifact.clone(),
        callable: request.callable.clone(),
        requested_outputs: request.requested_outputs,
    }
}

fn object_references(
    arguments: &[ValuePayload],
) -> NativeExecutionResult<BTreeMap<Digest, ValueRef>> {
    arguments
        .iter()
        .map(|argument| match argument {
            ValuePayload::Object(reference) => {
                Ok((reference.logical_digest, reference.as_ref().clone()))
            }
            _ => Err(protocol("study root inputs must be externalized")),
        })
        .collect()
}
