use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_fea::execution::{StudyIdentity, SweepDefinition, SweepIdentity};
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{OutputContract, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::{
    ArtifactAccess, ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionDescriptor, ProgramExecutionObject,
    ProgramExecutionRequest, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};

use super::{
    protocol, CONTRACT_LIMITS, EXACT_STRUCTURAL_SWEEP_ROOT_EXECUTION_MODE,
    EXACT_STRUCTURAL_SWEEP_ROOT_TARGET_PROFILE, ROOT_DOMAIN, ROOT_MEDIA_TYPE, ROOT_PREFIX,
    ROOT_PROGRAM_BYTES, ROOT_VALUE_SCHEMA,
};
use crate::{
    ExactStructuralStudyResultExpectation, NativeExecutionResult, PreparedExactStructuralStudyRoot,
    RequiredStudyProgram,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactStructuralSweepEntry {
    study: StudyIdentity,
    descriptor: ProgramExecutionDescriptor,
    argument_count: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactStructuralSweepRootRequest {
    schema_version: u16,
    definition: SweepDefinition,
    entries: Vec<ExactStructuralSweepEntry>,
}

impl ExactStructuralSweepRootRequest {
    fn validate(&self) -> NativeExecutionResult<()> {
        if self.schema_version != 1 || self.entries.is_empty() {
            return Err(protocol("exact structural sweep root is invalid"));
        }
        let entry_studies = self
            .entries
            .iter()
            .map(|entry| entry.study)
            .collect::<Vec<_>>();
        if self.definition.studies() != entry_studies || self.definition.identity().is_err() {
            return Err(protocol(
                "exact structural sweep definition differs from its ordered studies",
            ));
        }
        for entry in &self.entries {
            entry.descriptor.validate().map_err(protocol)?;
            if entry.argument_count == 0
                || !matches!(
                    entry.descriptor.recipe.execution_mode.as_str(),
                    crate::EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE
                        | crate::STRUCTURAL_STUDY_ROOT_EXECUTION_MODE
                )
                || entry.descriptor.artifact.form != ExecutableForm::AnalysisWorkload
                || entry.descriptor.requested_outputs != 1
            {
                return Err(protocol("exact structural sweep entry is invalid"));
            }
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> NativeExecutionResult<Vec<u8>> {
        self.validate()?;
        encode_contract(ROOT_PREFIX, ROOT_DOMAIN, self, CONTRACT_LIMITS).map_err(protocol)
    }

    fn decode(bytes: &[u8]) -> NativeExecutionResult<Self> {
        let request =
            decode_contract(ROOT_PREFIX, ROOT_DOMAIN, bytes, CONTRACT_LIMITS).map_err(protocol)?;
        Self::validate(&request)?;
        Ok(request)
    }
}

pub struct ExactStructuralSweepRootPreparation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub definition: SweepDefinition,
    pub studies: Vec<PreparedExactStructuralStudyRoot>,
}

pub struct StructuralSweepRootPreparation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub definition: SweepDefinition,
    pub studies: Vec<crate::PreparedStructuralStudyRoot>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralSweepResultExpectation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub definition: SweepDefinition,
    pub studies: Vec<crate::StructuralStudyResultExpectation>,
}

impl StructuralSweepResultExpectation {
    pub fn identity(&self) -> NativeExecutionResult<SweepIdentity> {
        self.definition.identity().map_err(protocol)
    }

    pub fn validate(&self) -> NativeExecutionResult<()> {
        self.artifact_access.validate().map_err(protocol)?;
        if self.studies.is_empty() {
            return Err(protocol("structural sweep expectation is empty"));
        }
        for study in &self.studies {
            study.validate()?;
            if study.program_revision != self.program_revision
                || study.artifact_access != self.artifact_access
            {
                return Err(protocol(
                    "structural sweep study differs from its revision or authority",
                ));
            }
        }
        let studies = self
            .studies
            .iter()
            .map(|study| StudyIdentity::from_authored_digest(study.authored_study_digest, None))
            .collect::<Result<Vec<_>, _>>()
            .map_err(protocol)?;
        if self.definition.studies() != studies {
            return Err(protocol(
                "structural sweep definition differs from its ordered studies",
            ));
        }
        self.identity()?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedExactStructuralSweepRoot {
    request: ProgramExecutionRequest,
    objects: Vec<ProgramExecutionObject>,
    required_programs: Vec<RequiredStudyProgram>,
    expectation: super::ExactStructuralSweepResultExpectation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedStructuralSweepRoot {
    request: ProgramExecutionRequest,
    objects: Vec<ProgramExecutionObject>,
    required_programs: Vec<RequiredStudyProgram>,
    expectation: StructuralSweepResultExpectation,
}

impl PreparedStructuralSweepRoot {
    pub const fn request(&self) -> &ProgramExecutionRequest {
        &self.request
    }

    pub fn transport_objects(&self) -> Vec<ProgramExecutionObject> {
        self.objects.clone()
    }

    pub fn required_programs(&self) -> Vec<RequiredStudyProgram> {
        self.required_programs.clone()
    }

    pub fn result_expectation(&self) -> StructuralSweepResultExpectation {
        self.expectation.clone()
    }

    pub fn rebind_artifact_access(
        mut self,
        artifact_access: ArtifactAccess,
    ) -> NativeExecutionResult<Self> {
        artifact_access.validate().map_err(protocol)?;
        for argument in &mut self.request.arguments {
            let ValuePayload::Object(reference) = argument else {
                return Err(protocol("structural sweep inputs must be externalized"));
            };
            rebind_reference(reference, &artifact_access)?;
        }
        for object in &mut self.objects {
            rebind_reference(&mut object.reference, &artifact_access)?;
            object.validate().map_err(protocol)?;
        }
        self.expectation.artifact_access = artifact_access.clone();
        for study in &mut self.expectation.studies {
            study.artifact_access = artifact_access.clone();
        }
        self.expectation.validate()?;
        self.request
            .validate_for_portable_host()
            .map_err(protocol)?;
        Ok(self)
    }

    pub fn into_request(self) -> ProgramExecutionRequest {
        self.request
    }
}

fn rebind_reference(
    reference: &mut ValueRef,
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

impl PreparedExactStructuralSweepRoot {
    pub const fn request(&self) -> &ProgramExecutionRequest {
        &self.request
    }

    pub fn transport_objects(&self) -> Vec<ProgramExecutionObject> {
        self.objects.clone()
    }

    pub fn required_programs(&self) -> Vec<RequiredStudyProgram> {
        self.required_programs.clone()
    }

    pub fn result_expectation(&self) -> super::ExactStructuralSweepResultExpectation {
        self.expectation.clone()
    }

    pub fn into_request(self) -> ProgramExecutionRequest {
        self.request
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedExactStructuralSweepRoot {
    pub identity: SweepIdentity,
    pub composition: runmat_analysis_fea::execution::SweepCompositionPolicy,
    pub studies: Vec<ProgramExecutionRequest>,
    pub artifact_access: ArtifactAccess,
}

pub fn prepare_exact_structural_sweep_root(
    preparation: ExactStructuralSweepRootPreparation,
) -> NativeExecutionResult<PreparedExactStructuralSweepRoot> {
    let ExactStructuralSweepRootPreparation {
        program_revision,
        artifact_access,
        definition,
        studies,
    } = preparation;
    artifact_access.validate().map_err(protocol)?;
    if studies.is_empty() {
        return Err(protocol(
            "exact structural sweep requires at least one study",
        ));
    }

    let mut entries = Vec::with_capacity(studies.len());
    let mut arguments = Vec::new();
    let mut object_inventory = BTreeMap::new();
    let mut required_programs = Vec::new();
    let mut expectations = Vec::with_capacity(studies.len());
    for study in studies {
        let request = study.request().clone();
        if request.recipe.program_revision != program_revision {
            return Err(protocol("sweep and study program revisions differ"));
        }
        let expectation = study.result_expectation();
        entries.push(ExactStructuralSweepEntry {
            study: expectation.study_identity()?,
            descriptor: descriptor(&request),
            argument_count: u32::try_from(request.arguments.len()).map_err(protocol)?,
        });
        arguments.extend(request.arguments);
        for object in study.transport_objects()? {
            insert_object(&mut object_inventory, object)?;
        }
        required_programs.extend(study.required_programs()?);
        expectations.push(expectation);
    }

    let prepared_studies = expectations
        .iter()
        .map(ExactStructuralStudyResultExpectation::study_identity)
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    if definition.studies() != prepared_studies {
        return Err(protocol(
            "sweep definition differs from its prepared study identities",
        ));
    }
    definition.identity().map_err(protocol)?;

    let root = ExactStructuralSweepRootRequest {
        schema_version: 1,
        definition: definition.clone(),
        entries,
    };
    let root_object = LogicalObject::new(
        ObjectNamespace::InputValue,
        "analysis/exact-structural-sweep-roots",
        ROOT_MEDIA_TYPE,
        root.canonical_bytes()?,
    )
    .map_err(protocol)?;
    let root_reference = artifact_access
        .reference(&root_object, ValueRefKind::DriverObject, ROOT_VALUE_SCHEMA)
        .map_err(protocol)?;
    insert_object(
        &mut object_inventory,
        ProgramExecutionObject {
            reference: root_reference.clone(),
            bytes: root_object.bytes,
        },
    )?;

    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: program_revision.clone(),
        entrypoint: "analysis_workload".into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: EXACT_STRUCTURAL_SWEEP_ROOT_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(EXACT_STRUCTURAL_SWEEP_ROOT_TARGET_PROFILE),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::AnalysisWorkload,
        ROOT_PROGRAM_BYTES.to_vec(),
    )
    .map_err(protocol)?;
    required_programs.push(RequiredStudyProgram {
        recipe: recipe.clone(),
        artifact: artifact.clone(),
    });
    let mut seen_programs = BTreeSet::new();
    required_programs.retain(|program| seen_programs.insert(program.artifact.id));
    let mut request_arguments = vec![ValuePayload::Object(Box::new(root_reference))];
    request_arguments.extend(arguments);
    let request = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: runmat_execution::ProgramCallable::semantic(
            runmat_execution::ProgramFunctionId(0),
            None,
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: request_arguments,
        requested_outputs: 1,
    };
    request.validate().map_err(protocol)?;
    let expectation = super::ExactStructuralSweepResultExpectation {
        program_revision,
        artifact_access,
        definition,
        studies: expectations,
    };
    expectation.validate()?;
    Ok(PreparedExactStructuralSweepRoot {
        request,
        objects: object_inventory.into_values().collect(),
        required_programs,
        expectation,
    })
}

pub fn prepare_structural_sweep_root(
    preparation: StructuralSweepRootPreparation,
) -> NativeExecutionResult<PreparedStructuralSweepRoot> {
    let StructuralSweepRootPreparation {
        program_revision,
        artifact_access,
        definition,
        studies,
    } = preparation;
    artifact_access.validate().map_err(protocol)?;
    if studies.is_empty() {
        return Err(protocol("structural sweep requires at least one study"));
    }

    let mut entries = Vec::with_capacity(studies.len());
    let mut arguments = Vec::new();
    let mut object_inventory = BTreeMap::new();
    let mut required_programs = Vec::new();
    let mut identities = Vec::with_capacity(studies.len());
    let mut expectations = Vec::with_capacity(studies.len());
    for study in studies {
        let request = study.request().clone();
        if request.recipe.program_revision != program_revision {
            return Err(protocol("sweep and study program revisions differ"));
        }
        let identity = StudyIdentity::from_authored_digest(study.authored_study_digest(), None)
            .map_err(protocol)?;
        entries.push(ExactStructuralSweepEntry {
            study: identity,
            descriptor: descriptor(&request),
            argument_count: u32::try_from(request.arguments.len()).map_err(protocol)?,
        });
        arguments.extend(request.arguments);
        for object in study.transport_objects()? {
            insert_object(&mut object_inventory, object)?;
        }
        required_programs.extend(study.required_programs().iter().map(|program| {
            RequiredStudyProgram {
                recipe: program.recipe.clone(),
                artifact: program.artifact.clone(),
            }
        }));
        identities.push(identity);
        expectations.push(study.result_expectation());
    }
    if definition.studies() != identities {
        return Err(protocol(
            "sweep definition differs from its prepared study identities",
        ));
    }
    definition.identity().map_err(protocol)?;

    let expectation_definition = definition.clone();
    let root = ExactStructuralSweepRootRequest {
        schema_version: 1,
        definition,
        entries,
    };
    let root_object = LogicalObject::new(
        ObjectNamespace::InputValue,
        "analysis/structural-sweep-roots",
        ROOT_MEDIA_TYPE,
        root.canonical_bytes()?,
    )
    .map_err(protocol)?;
    let root_reference = artifact_access
        .reference(&root_object, ValueRefKind::DriverObject, ROOT_VALUE_SCHEMA)
        .map_err(protocol)?;
    insert_object(
        &mut object_inventory,
        ProgramExecutionObject {
            reference: root_reference.clone(),
            bytes: root_object.bytes,
        },
    )?;

    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision,
        entrypoint: "analysis_workload".into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: EXACT_STRUCTURAL_SWEEP_ROOT_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(EXACT_STRUCTURAL_SWEEP_ROOT_TARGET_PROFILE),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::AnalysisWorkload,
        ROOT_PROGRAM_BYTES.to_vec(),
    )
    .map_err(protocol)?;
    required_programs.push(RequiredStudyProgram {
        recipe: recipe.clone(),
        artifact: artifact.clone(),
    });
    let mut seen_programs = BTreeSet::new();
    required_programs.retain(|program| seen_programs.insert(program.artifact.id));
    let mut request_arguments = vec![ValuePayload::Object(Box::new(root_reference))];
    request_arguments.extend(arguments);
    let request = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: runmat_execution::ProgramCallable::semantic(
            runmat_execution::ProgramFunctionId(0),
            None,
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: request_arguments,
        requested_outputs: 1,
    };
    request.validate().map_err(protocol)?;
    let expectation = StructuralSweepResultExpectation {
        program_revision: request.recipe.program_revision.clone(),
        artifact_access,
        definition: expectation_definition,
        studies: expectations,
    };
    expectation.validate()?;
    Ok(PreparedStructuralSweepRoot {
        request,
        objects: object_inventory.into_values().collect(),
        required_programs,
        expectation,
    })
}

pub fn decode_exact_structural_sweep_root(
    program: &ProgramExecutionRequest,
    store: &impl CacheImport,
) -> NativeExecutionResult<DecodedExactStructuralSweepRoot> {
    program.validate_for_portable_host().map_err(protocol)?;
    if program.artifact.form != ExecutableForm::AnalysisWorkload
        || program.recipe.execution_mode != EXACT_STRUCTURAL_SWEEP_ROOT_EXECUTION_MODE
        || program.recipe.target
            != ProgramTarget::portable(EXACT_STRUCTURAL_SWEEP_ROOT_TARGET_PROFILE)
        || program.artifact.executable_bytes != ROOT_PROGRAM_BYTES
        || !program.recipe.compile_options.is_empty()
    {
        return Err(protocol("program is not an exact structural sweep root"));
    }
    let [ValuePayload::Object(root_reference), child_arguments @ ..] = program.arguments.as_slice()
    else {
        return Err(protocol("exact structural sweep root inputs are invalid"));
    };
    if root_reference.kind != ValueRefKind::DriverObject {
        return Err(protocol("exact structural sweep root reference is invalid"));
    }
    let bytes = store
        .read_verified(root_reference.logical_digest)
        .map_err(protocol)?
        .ok_or_else(|| protocol("exact structural sweep root object is unavailable"))?;
    if bytes.len() as u64 != root_reference.encoded_length {
        return Err(protocol("exact structural sweep root length changed"));
    }
    let root = ExactStructuralSweepRootRequest::decode(&bytes)?;
    let artifact_access = ArtifactAccess {
        authorization_scope: root_reference.authorization_scope.clone(),
        encryption_context: root_reference.encryption_context,
    };
    let expected_argument_count = root.entries.iter().try_fold(0usize, |count, entry| {
        count.checked_add(entry.argument_count as usize)
    });
    if expected_argument_count != Some(child_arguments.len()) {
        return Err(protocol(
            "exact structural sweep argument cardinality changed",
        ));
    }
    let mut offset = 0usize;
    let mut studies = Vec::with_capacity(root.entries.len());
    for entry in root.entries {
        let end = offset + entry.argument_count as usize;
        let request = ProgramExecutionRequest {
            console: entry.descriptor.console,
            environment: entry.descriptor.environment,
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            recipe: entry.descriptor.recipe,
            artifact: entry.descriptor.artifact,
            callable: entry.descriptor.callable.clone(),
            context: runmat_execution::ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments: child_arguments[offset..end].to_vec(),
            requested_outputs: entry.descriptor.requested_outputs,
        };
        match request.recipe.execution_mode.as_str() {
            crate::EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE => {
                crate::decode_exact_structural_study_root(&request, store)?;
            }
            crate::STRUCTURAL_STUDY_ROOT_EXECUTION_MODE => {
                crate::decode_structural_study_root(&request, store)?;
            }
            _ => return Err(protocol("structural sweep entry mode is unsupported")),
        }
        studies.push(request);
        offset = end;
    }
    Ok(DecodedExactStructuralSweepRoot {
        identity: root.definition.identity().map_err(protocol)?,
        composition: root.definition.composition(),
        studies,
        artifact_access,
    })
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

fn insert_object(
    objects: &mut BTreeMap<runmat_execution::Digest, ProgramExecutionObject>,
    object: ProgramExecutionObject,
) -> NativeExecutionResult<()> {
    object.validate().map_err(protocol)?;
    match objects.get(&object.reference.logical_digest) {
        Some(existing) if existing != &object => Err(protocol(
            "sweep input object digest has conflicting metadata",
        )),
        Some(_) => Ok(()),
        None => {
            objects.insert(object.reference.logical_digest, object);
            Ok(())
        }
    }
}

#[cfg(test)]
mod environment_tests;
