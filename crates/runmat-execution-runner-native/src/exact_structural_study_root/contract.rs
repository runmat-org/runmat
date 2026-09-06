use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_execution::{
    prepare_analysis_host_program, AnalysisHostProgramArtifact, PreparedSignalGeneratorChunk,
    PreparedSignalGeneratorProgram, PreparedSignalMaterialization, PreparedStructuralLinearProgram,
};
use runmat_analysis_fea::execution::StructuralLinearWorkload;
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, OutputContract, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionDescriptor, ProgramExecutionObject,
    ProgramExecutionRequest, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_meshing_core::MeshingRequest;
use runmat_meshing_execution::MeshingHostProgram;
use serde::{Deserialize, Serialize};

use super::{
    protocol, read, CONTRACT_LIMITS, EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE,
    EXACT_STRUCTURAL_STUDY_ROOT_TARGET_PROFILE, ROOT_DOMAIN, ROOT_MEDIA_TYPE, ROOT_PREFIX,
    ROOT_PROGRAM_BYTES, ROOT_VALUE_SCHEMA,
};
use crate::{decode_exact_meshing_root, NativeExecutionResult, PreparedExactMeshingRoot};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactStructuralStudyRootRequest {
    schema_version: u16,
    authored_study_digest: Digest,
    frozen_project_digest: Option<Digest>,
    geometry_source_digest: Digest,
    workload_template: StructuralLinearWorkload,
    signal_materializations: Vec<ExactSignalMaterializationRequest>,
    exact_meshing: ProgramExecutionDescriptor,
    exact_argument_count: u32,
}

impl ExactStructuralStudyRootRequest {
    fn validate(&self) -> NativeExecutionResult<()> {
        self.workload_template.validate().map_err(protocol)?;
        self.exact_meshing.validate().map_err(protocol)?;
        let prepared_signals = self
            .signal_materializations
            .iter()
            .map(ExactSignalMaterializationRequest::to_prepared)
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        validate_signal_inventory(
            &self.workload_template,
            &prepared_signals,
            &self.exact_meshing.recipe.program_revision,
        )?;
        if self.schema_version != 1
            || self.authored_study_digest.bytes() == &[0; 32]
            || self.geometry_source_digest.bytes() == &[0; 32]
            || self
                .frozen_project_digest
                .is_some_and(|digest| digest.bytes() == &[0; 32])
            || self.exact_argument_count < 2
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactSignalMaterializationRequest {
    authored: runmat_analysis_core::PhysicalSignal,
    admitted: runmat_analysis_execution::AdmittedSignalGenerator,
    descriptor: ProgramExecutionDescriptor,
    chunks: Vec<runmat_analysis_execution::SignalGeneratorChunk>,
}

impl ExactSignalMaterializationRequest {
    fn from_prepared(prepared: &PreparedSignalMaterialization) -> NativeExecutionResult<Self> {
        prepared.validate().map_err(protocol)?;
        Ok(Self {
            authored: prepared.authored.clone(),
            admitted: prepared.admitted.clone(),
            descriptor: prepared.program.descriptor.clone(),
            chunks: prepared
                .program
                .chunks
                .iter()
                .map(|chunk| chunk.plan.clone())
                .collect(),
        })
    }

    fn to_prepared(&self) -> NativeExecutionResult<PreparedSignalMaterialization> {
        let chunks = self
            .chunks
            .iter()
            .cloned()
            .map(|plan| PreparedSignalGeneratorChunk {
                request: ProgramExecutionRequest {
                    schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
                    recipe: self.descriptor.recipe.clone(),
                    artifact: self.descriptor.artifact.clone(),
                    function: self.descriptor.function,
                    arguments: plan.arguments.clone(),
                    requested_outputs: self.descriptor.requested_outputs,
                },
                plan,
            })
            .collect();
        let prepared = PreparedSignalMaterialization {
            authored: self.authored.clone(),
            admitted: self.admitted.clone(),
            program: PreparedSignalGeneratorProgram {
                descriptor: self.descriptor.clone(),
                chunks,
            },
        };
        prepared.validate().map_err(protocol)?;
        Ok(prepared)
    }
}

pub struct ExactStructuralStudyRootPreparation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub geometry_source_digest: Digest,
    pub workload_template: StructuralLinearWorkload,
    pub signal_materializations: Vec<PreparedSignalMaterialization>,
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
    root: ExactStructuralStudyRootRequest,
    artifact_access: ArtifactAccess,
    required_programs: Vec<RequiredStudyProgram>,
    expected_meshing_request: MeshingRequest,
    signal_expectations: Vec<runmat_analysis_execution::SignalMaterializationExpectation>,
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
            workload_template: self.root.workload_template.clone(),
            signal_materializations: self.signal_expectations.clone(),
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
    pub workload_template: StructuralLinearWorkload,
    pub signal_materializations: Vec<PreparedSignalMaterialization>,
    pub exact_meshing: ProgramExecutionRequest,
    pub meshing_request: MeshingRequest,
    pub artifact_access: ArtifactAccess,
}

pub fn prepare_exact_structural_study_root(
    preparation: ExactStructuralStudyRootPreparation,
) -> NativeExecutionResult<PreparedExactStructuralStudyRoot> {
    let ExactStructuralStudyRootPreparation {
        program_revision,
        artifact_access,
        authored_study_digest,
        frozen_project_digest,
        geometry_source_digest,
        workload_template,
        signal_materializations,
        exact_meshing,
    } = preparation;
    artifact_access.validate().map_err(protocol)?;
    let exact_request = exact_meshing.request().clone();
    let expected_meshing_request = exact_meshing.meshing_request().clone();
    let exact_programs = exact_meshing.required_programs()?;
    validate_signal_inventory(
        &workload_template,
        &signal_materializations,
        &program_revision,
    )?;
    let signal_expectations = signal_materializations
        .iter()
        .map(PreparedSignalMaterialization::expectation)
        .collect::<Result<Vec<_>, _>>()
        .map_err(protocol)?;
    if exact_request.recipe.program_revision != program_revision {
        return Err(protocol("study and exact-meshing program revisions differ"));
    }
    for reference in object_references(&exact_request.arguments)?.values() {
        if reference.authorization_scope != artifact_access.authorization_scope
            || reference.encryption_context != artifact_access.encryption_context
        {
            return Err(protocol(
                "study and exact-meshing artifact authorities differ",
            ));
        }
    }
    let root = ExactStructuralStudyRootRequest {
        schema_version: 1,
        authored_study_digest,
        frozen_project_digest,
        geometry_source_digest,
        workload_template,
        signal_materializations: signal_materializations
            .iter()
            .map(ExactSignalMaterializationRequest::from_prepared)
            .collect::<NativeExecutionResult<Vec<_>>>()?,
        exact_meshing: descriptor(&exact_request),
        exact_argument_count: u32::try_from(exact_request.arguments.len()).map_err(protocol)?,
    };
    let root_object = LogicalObject::new(
        ObjectNamespace::InputValue,
        "analysis/exact-structural-study-roots",
        ROOT_MEDIA_TYPE,
        root.canonical_bytes()?,
    )
    .map_err(protocol)?;
    let root_reference = artifact_access
        .reference(&root_object, ValueRefKind::DriverObject, ROOT_VALUE_SCHEMA)
        .map_err(protocol)?;
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision,
        entrypoint: "analysis_workload".into(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(EXACT_STRUCTURAL_STUDY_ROOT_TARGET_PROFILE),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::AnalysisWorkload,
        ROOT_PROGRAM_BYTES.to_vec(),
    )
    .map_err(protocol)?;
    let mut arguments = vec![ValuePayload::Object(Box::new(root_reference))];
    arguments.extend(exact_request.arguments);
    let request = ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe,
        artifact,
        function: 0,
        arguments,
        requested_outputs: 1,
    };
    request.validate().map_err(protocol)?;
    let mut objects = exact_meshing.objects().to_vec();
    objects.push(root_object);
    let mut required_programs = vec![RequiredStudyProgram {
        recipe: request.recipe.clone(),
        artifact: request.artifact.clone(),
    }];
    for MeshingHostProgram { recipe, artifact } in exact_programs {
        required_programs.push(RequiredStudyProgram { recipe, artifact });
    }
    for materialization in &signal_materializations {
        required_programs.push(RequiredStudyProgram {
            recipe: materialization.program.descriptor.recipe.clone(),
            artifact: materialization.program.descriptor.artifact.clone(),
        });
    }
    let AnalysisHostProgramArtifact { recipe, artifact } =
        prepare_analysis_host_program(request.recipe.program_revision.clone(), 1)
            .map_err(protocol)?;
    required_programs.push(RequiredStudyProgram { recipe, artifact });
    let mut seen = BTreeSet::new();
    required_programs.retain(|program| seen.insert(program.artifact.id));
    Ok(PreparedExactStructuralStudyRoot {
        request,
        objects,
        root,
        artifact_access,
        required_programs,
        expected_meshing_request,
        signal_expectations,
    })
}

pub fn decode_exact_structural_study_root(
    program: &ProgramExecutionRequest,
    store: &impl CacheImport,
) -> NativeExecutionResult<DecodedExactStructuralStudyRoot> {
    program.validate_for_portable_host().map_err(protocol)?;
    if program.artifact.form != ExecutableForm::AnalysisWorkload
        || program.recipe.execution_mode != EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE
        || program.recipe.target
            != ProgramTarget::portable(EXACT_STRUCTURAL_STUDY_ROOT_TARGET_PROFILE)
        || program.artifact.executable_bytes != ROOT_PROGRAM_BYTES
        || !program.recipe.compile_options.is_empty()
    {
        return Err(protocol("program is not an exact structural study root"));
    }
    let [ValuePayload::Object(root_reference), exact_arguments @ ..] = program.arguments.as_slice()
    else {
        return Err(protocol("exact structural study root inputs are invalid"));
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
    let bytes = read(store, root_reference)?;
    let root = ExactStructuralStudyRootRequest::decode(&bytes)?;
    if usize::try_from(root.exact_argument_count).map_err(protocol)? != exact_arguments.len()
        || root.exact_meshing.recipe.program_revision != program.recipe.program_revision
    {
        return Err(protocol(
            "exact structural study root component contract differs from its request",
        ));
    }
    let exact_meshing = ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe: root.exact_meshing.recipe.clone(),
        artifact: root.exact_meshing.artifact.clone(),
        function: root.exact_meshing.function,
        arguments: exact_arguments.to_vec(),
        requested_outputs: root.exact_meshing.requested_outputs,
    };
    let decoded_exact = decode_exact_meshing_root(&exact_meshing, store)?;
    if decoded_exact.artifact_access != artifact_access
        || Digest::sha256(&decoded_exact.source_bytes) != root.geometry_source_digest
    {
        return Err(protocol(
            "exact structural study geometry differs from its semantic intent",
        ));
    }
    let signal_materializations = root
        .signal_materializations
        .iter()
        .map(ExactSignalMaterializationRequest::to_prepared)
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    Ok(DecodedExactStructuralStudyRoot {
        authored_study_digest: root.authored_study_digest,
        frozen_project_digest: root.frozen_project_digest,
        geometry_source_digest: root.geometry_source_digest,
        workload_template: root.workload_template,
        signal_materializations,
        exact_meshing,
        meshing_request: decoded_exact.request.meshing_request,
        artifact_access,
    })
}

fn validate_signal_inventory(
    workload: &StructuralLinearWorkload,
    materializations: &[PreparedSignalMaterialization],
    program_revision: &ProgramRevision,
) -> NativeExecutionResult<()> {
    let authored = workload
        .model
        .signals
        .iter()
        .filter(|signal| signal.generator().is_some())
        .collect::<Vec<_>>();
    if authored.len() != materializations.len()
        || authored
            .iter()
            .zip(materializations)
            .any(|(authored, prepared)| *authored != &prepared.authored)
    {
        return Err(protocol(
            "study signal materializations differ from its authored generator inventory",
        ));
    }
    let mut identities = BTreeSet::new();
    for prepared in materializations {
        prepared.validate().map_err(protocol)?;
        if &prepared.admitted.program_revision != program_revision
            || !identities.insert(prepared.admitted.materialization_identity)
        {
            return Err(protocol(
                "study signal materialization inventory is inconsistent or duplicated",
            ));
        }
    }
    Ok(())
}

pub(crate) fn prepare_structural_study_analysis_program(
    authored_study_digest: Digest,
    frozen_project_digest: Option<Digest>,
    program_revision: ProgramRevision,
    artifact_access: ArtifactAccess,
    workload: StructuralLinearWorkload,
    mesh: runmat_meshing_core::SolverMeshArtifact,
) -> NativeExecutionResult<PreparedStructuralLinearProgram> {
    runmat_analysis_execution::prepare_structural_linear_program(
        runmat_analysis_execution::StructuralLinearProgramPreparation {
            authored_study_digest,
            frozen_project_digest,
            program_revision,
            artifact_access,
            workload,
            mesh,
            chunk_bytes: runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
        },
    )
    .map_err(protocol)
}

fn descriptor(request: &ProgramExecutionRequest) -> ProgramExecutionDescriptor {
    ProgramExecutionDescriptor {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe: request.recipe.clone(),
        artifact: request.artifact.clone(),
        function: request.function,
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
            ValuePayload::Inline(_) => Err(protocol("study root inputs must be externalized")),
        })
        .collect()
}
