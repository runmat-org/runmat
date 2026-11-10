//! Whole-study contract that carries a typed analysis definition through geometry and meshing.

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_analysis_execution::{
    prepare_analysis_host_program, InputMaterializationPlan, InputMaterializationPlanExpectation,
};
use runmat_analysis_fea::planning::resource::LocalExecutionPlan;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution::{Digest, OutputContract, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactAccess, ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionDescriptor, ProgramExecutionObject,
    ProgramExecutionRequest, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};

use crate::input_materialization_contract::{
    validate_input_materialization_plan_inventory, validate_input_plan_expectation_inventory,
    InputMaterializationPlanRequest,
};
use crate::{
    decode_geometry_meshing_root, NativeExecutionError, NativeExecutionResult,
    PreparedGeometryMeshingRoot, RequiredGeometryMeshingProgram,
};

pub const STRUCTURAL_STUDY_ROOT_EXECUTION_MODE: &str = "structural-study-root";
pub const STRUCTURAL_STUDY_ROOT_TARGET_PROFILE: &str = "portable-structural-study-root";
const ROOT_PROGRAM_BYTES: &[u8] = b"runmat-structural-study-root\0";
const ROOT_PREFIX: &[u8] = b"runmat-structural-study-root\0";
const ROOT_DOMAIN: &str = "structural-study-root";
const ROOT_MEDIA_TYPE: &str = "application/vnd.runmat.structural-study-root+cbor";
const ROOT_VALUE_SCHEMA: &str = "runmat.analysis.structural-study-root";
const CONTRACT_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(65_536, 96);

mod contract;
mod initial_fields;
mod preparation;
pub use contract::{
    NonlinearSolidStudyPlanning, StructuralStudyMaterialExpectation,
    StructuralStudyResultExpectation, StructuralStudyRootPreparation, StudyRootPlanning,
};
use contract::{StructuralStudyMaterial, StructuralStudyRootRequest};
pub use preparation::prepare_structural_study_root;

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedStructuralStudyRoot {
    request: ProgramExecutionRequest,
    objects: Vec<LogicalObject>,
    input_value_objects: Vec<ProgramExecutionObject>,
    required_programs: Vec<RequiredStructuralStudyProgram>,
    authored_study_digest: Digest,
    expectation: StructuralStudyResultExpectation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequiredStructuralStudyProgram {
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
}

impl PreparedStructuralStudyRoot {
    pub const fn request(&self) -> &ProgramExecutionRequest {
        &self.request
    }

    pub fn objects(&self) -> &[LogicalObject] {
        &self.objects
    }

    pub fn required_programs(&self) -> &[RequiredStructuralStudyProgram] {
        &self.required_programs
    }

    pub const fn authored_study_digest(&self) -> Digest {
        self.authored_study_digest
    }

    pub fn result_expectation(&self) -> StructuralStudyResultExpectation {
        self.expectation.clone()
    }

    /// Rebinds transport authority without recompiling study-definition or executable identity.
    pub fn rebind_artifact_access(
        mut self,
        artifact_access: ArtifactAccess,
    ) -> NativeExecutionResult<Self> {
        artifact_access.validate().map_err(protocol)?;
        for argument in &mut self.request.arguments {
            let ValuePayload::Object(reference) = argument else {
                return Err(protocol(
                    "structural study root inputs must be externalized",
                ));
            };
            rebind_reference(reference, &artifact_access)?;
        }
        for object in &mut self.input_value_objects {
            rebind_reference(&mut object.reference, &artifact_access)?;
        }
        self.expectation.artifact_access = artifact_access;
        self.expectation.validate()?;
        self.request
            .validate_for_portable_host()
            .map_err(protocol)?;
        self.transport_objects()?;
        Ok(self)
    }

    pub fn write_to(&self, store: &mut impl CacheExport) -> NativeExecutionResult<()> {
        for object in &self.objects {
            store.write_verified(object).map_err(protocol)?;
        }
        for object in &self.input_value_objects {
            let logical = LogicalObject::new(
                ObjectNamespace::InputValue,
                format!(
                    "analysis/generator-inputs/{}",
                    object.reference.logical_digest
                ),
                object.reference.media_type.clone(),
                object.bytes.clone(),
            )
            .map_err(protocol)?;
            if logical.descriptor.digest != object.reference.logical_digest {
                return Err(protocol(
                    "generator input object differs from its logical reference",
                ));
            }
            store.write_verified(&logical).map_err(protocol)?;
        }
        Ok(())
    }

    pub fn transport_objects(&self) -> NativeExecutionResult<Vec<ProgramExecutionObject>> {
        let references = self
            .request
            .arguments
            .iter()
            .map(|argument| match argument {
                ValuePayload::Object(reference) => Ok((reference.logical_digest, reference)),
                _ => Err(protocol(
                    "structural study root inputs must be externalized",
                )),
            })
            .collect::<NativeExecutionResult<BTreeMap<_, _>>>()?;
        let mut objects = self
            .objects
            .iter()
            .map(|object| {
                let reference = references
                    .get(&object.descriptor.digest)
                    .ok_or_else(|| protocol("structural study root contains an unbound object"))?;
                let object = ProgramExecutionObject {
                    reference: (*reference).as_ref().clone(),
                    bytes: object.bytes.clone(),
                };
                object.validate().map_err(protocol)?;
                Ok(object)
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        for object in &self.input_value_objects {
            let reference = references
                .get(&object.reference.logical_digest)
                .ok_or_else(|| protocol("structural study omits a generator input reference"))?;
            if reference.as_ref() != &object.reference {
                return Err(protocol(
                    "structural study generator input differs from its request reference",
                ));
            }
            object.validate().map_err(protocol)?;
            objects.push(object.clone());
        }
        if objects.len() != references.len() {
            return Err(protocol(
                "structural study root object inventory differs from its inputs",
            ));
        }
        Ok(objects)
    }

    pub fn into_request(self) -> ProgramExecutionRequest {
        self.request
    }
}

fn rebind_reference(
    reference: &mut runmat_execution::value::ValueRef,
    access: &ArtifactAccess,
) -> NativeExecutionResult<()> {
    reference.id = access.value_id(reference.logical_digest);
    reference.authorization_scope = access.authorization_scope.clone();
    reference.encryption_context = access.encryption_context;
    reference.resident_fence = None;
    ValuePayload::Object(Box::new(reference.clone()))
        .validate(runmat_execution::value::ValueLimits::default())
        .map_err(protocol)
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedStructuralStudyRoot {
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub study: RealizedStudy,
    pub materials: BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub planning: StudyRootPlanning,
    pub local_execution: LocalExecutionPlan,
    /// The execution policy's `memory_per_worker`, when authored.
    pub memory_per_worker_bytes: Option<u64>,
    /// The execution policy's accelerator admission, so the worker admits placement itself.
    pub accelerator_admission: runmat_analysis_core::AcceleratorAdmissionPolicy,
    pub analysis_accelerator_requests: Vec<AcceleratorRequest>,
    pub input_materializations: Vec<InputMaterializationPlan>,
    pub initial_fields: Vec<runmat_analysis_fea::execution::InitialFieldBinding>,
    pub geometry_meshing: ProgramExecutionRequest,
    pub artifact_access: ArtifactAccess,
}

impl DecodedStructuralStudyRoot {
    /// Admits the study's accelerator placement on the worker under the root's policy, before
    /// any meshing or solve: a requested accelerator that this worker cannot provide (no active
    /// provider, or a family without a device path) fails a strict study here instead of after a
    /// full remote run. Under the fallback policy the solve-time admission decides as usual.
    pub fn admit_accelerator_placement(
        &self,
    ) -> Result<(), runmat_analysis_fea::accelerator_admission::AcceleratorRequiredError> {
        if self.accelerator_admission != runmat_analysis_core::AcceleratorAdmissionPolicy::Required
        {
            return Ok(());
        }
        runmat_analysis_fea::accelerator_admission::admit_study_accelerator(
            &self.study.sequence.steps,
            self.planning.backend(),
            self.accelerator_admission,
            || match runmat_analysis_fea::accelerator_admission::device_unavailable_reason() {
                Some(reason) => Err(reason.to_string()),
                None => Ok(()),
            },
        )
        .map(|_| ())
    }
}

pub fn decode_structural_study_root(
    program: &ProgramExecutionRequest,
    store: &impl CacheImport,
) -> NativeExecutionResult<DecodedStructuralStudyRoot> {
    program.validate_for_portable_host().map_err(protocol)?;
    if program.artifact.form != ExecutableForm::AnalysisWorkload
        || program.recipe.execution_mode != STRUCTURAL_STUDY_ROOT_EXECUTION_MODE
        || program.recipe.target != ProgramTarget::portable(STRUCTURAL_STUDY_ROOT_TARGET_PROFILE)
        || program.artifact.executable_bytes != ROOT_PROGRAM_BYTES
        || !program.recipe.compile_options.is_empty()
    {
        return Err(protocol("program is not a structural study root"));
    }
    let ValuePayload::Object(root_reference) = program
        .arguments
        .first()
        .ok_or_else(|| protocol("structural study root descriptor is absent"))?
    else {
        return Err(protocol(
            "structural study root descriptor must be externalized",
        ));
    };
    let artifact_access = ArtifactAccess {
        authorization_scope: root_reference.authorization_scope.clone(),
        encryption_context: root_reference.encryption_context,
    };
    artifact_access
        .validate_reference(
            root_reference,
            ValueRefKind::DriverObject,
            ROOT_MEDIA_TYPE,
            ROOT_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let bytes = store
        .read_verified(root_reference.logical_digest)
        .map_err(protocol)?
        .ok_or_else(|| protocol("structural study root descriptor is unavailable"))?;
    if bytes.len() as u64 != root_reference.encoded_length {
        return Err(protocol(
            "structural study root descriptor length differs from its reference",
        ));
    }
    let root = StructuralStudyRootRequest::decode(&bytes)?;
    if root.geometry_meshing.recipe.program_revision != program.recipe.program_revision {
        return Err(protocol(
            "geometry-meshing and structural-study program revisions differ",
        ));
    }
    let geometry_count = usize::try_from(root.geometry_argument_count).map_err(protocol)?;
    let materialization_count =
        usize::try_from(root.materialization_argument_count).map_err(protocol)?;
    let source_count = usize::try_from(root.initial_field_argument_count).map_err(protocol)?;
    let component_count = geometry_count
        .checked_add(source_count)
        .and_then(|count| count.checked_add(materialization_count))
        .ok_or_else(|| protocol("structural study component count overflows"))?;
    let arguments = program
        .arguments
        .get(1..)
        .filter(|arguments| arguments.len() == component_count)
        .ok_or_else(|| protocol("structural study component count differs"))?;
    let (geometry_arguments, remaining_arguments) = arguments.split_at(geometry_count);
    let (materialization_arguments, source_arguments) =
        remaining_arguments.split_at(materialization_count);
    let initial_fields = initial_fields::decode(
        &root.study,
        &root.initial_fields,
        source_arguments,
        &artifact_access,
        store,
    )?;
    let geometry_meshing = ProgramExecutionRequest {
        console: root.geometry_meshing.console,
        environment: root.geometry_meshing.environment,
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe: root.geometry_meshing.recipe,
        artifact: root.geometry_meshing.artifact,
        callable: root.geometry_meshing.callable.clone(),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: geometry_arguments.to_vec(),
        requested_outputs: root.geometry_meshing.requested_outputs,
    };
    decode_geometry_meshing_root(&geometry_meshing, store)?;
    let input_materializations = root
        .input_materializations
        .iter()
        .map(InputMaterializationPlanRequest::to_plan)
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let expected_materialization_references = input_materializations
        .iter()
        .flat_map(|prepared| prepared.input_references().iter())
        .map(|reference| (reference.id, reference))
        .collect::<BTreeMap<_, _>>();
    let transported_materialization_references = materialization_arguments
        .iter()
        .map(|argument| match argument {
            ValuePayload::Object(reference) => Ok((reference.id, reference.as_ref())),
            _ => Err(protocol("generator input value is not externalized")),
        })
        .collect::<NativeExecutionResult<BTreeMap<_, _>>>()?;
    if expected_materialization_references != transported_materialization_references {
        return Err(protocol(
            "structural study generator references differ from its materialization plans",
        ));
    }
    Ok(DecodedStructuralStudyRoot {
        authored_study_digest: root.authored_study_digest,
        frozen_project_digest: root.frozen_project_digest,
        study: root.study,
        materials: root
            .materials
            .into_iter()
            .map(|entry| (entry.reference, entry.material))
            .collect(),
        planning: root.planning,
        local_execution: root.local_execution,
        memory_per_worker_bytes: root.memory_per_worker_bytes,
        accelerator_admission: root.accelerator_admission,
        analysis_accelerator_requests: root.analysis_accelerator_requests,
        input_materializations,
        initial_fields,
        geometry_meshing,
        artifact_access,
    })
}

fn required_program(program: &RequiredGeometryMeshingProgram) -> RequiredStructuralStudyProgram {
    RequiredStructuralStudyProgram {
        recipe: program.recipe.clone(),
        artifact: program.artifact.clone(),
    }
}

fn descriptor_from_request(request: &ProgramExecutionRequest) -> ProgramExecutionDescriptor {
    ProgramExecutionDescriptor {
        console: request.console,
        environment: request.environment.clone(),
        schema_version: request.schema_version,
        recipe: request.recipe.clone(),
        artifact: request.artifact.clone(),
        callable: request.callable.clone(),
        requested_outputs: request.requested_outputs,
    }
}

fn validate_accelerator_requests(
    backend: runmat_analysis_fea::ComputeBackend,
    requests: &[AcceleratorRequest],
) -> NativeExecutionResult<()> {
    if requests.len() > 16 {
        return Err(protocol(
            "structural study exceeds the accelerator-request limit",
        ));
    }
    for request in requests {
        request.validate().map_err(protocol)?;
    }
    match backend {
        runmat_analysis_fea::ComputeBackend::Cpu if !requests.is_empty() => Err(protocol(
            "CPU study planning cannot require analysis accelerators",
        )),
        runmat_analysis_fea::ComputeBackend::Gpu if requests.is_empty() => Err(protocol(
            "accelerator study planning requires an exact allocation request",
        )),
        _ => Ok(()),
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
#[path = "structural_study_root_tests.rs"]
mod tests;
#[cfg(test)]
pub(crate) use tests::initial_fields::verify_original_field_transport;
