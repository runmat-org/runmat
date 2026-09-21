//! Whole-study contract that carries a typed analysis definition through geometry and meshing.

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_analysis_execution::{
    prepare_analysis_host_program, InputMaterializationExpectation, PreparedInputMaterialization,
};
use runmat_analysis_fea::study::StructuralStudyPlanningPolicy;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution::{Digest, OutputContract, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactAccess, ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionDescriptor, ProgramExecutionObject,
    ProgramExecutionRequest, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use serde::{Deserialize, Serialize};

use crate::input_materialization_contract::InputMaterializationRequest;
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
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 8_000_000, 65_536, 96);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralStudyRootRequest {
    schema_version: u16,
    authored_study_digest: Digest,
    frozen_project_digest: Option<Digest>,
    study: RealizedStudy,
    materials: Vec<StructuralStudyMaterial>,
    planning: StructuralStudyPlanningPolicy,
    analysis_accelerator_requests: Vec<AcceleratorRequest>,
    input_materializations: Vec<InputMaterializationRequest>,
    geometry_meshing: ProgramExecutionDescriptor,
    geometry_argument_count: u32,
    materialization_argument_count: u32,
}

impl StructuralStudyRootRequest {
    fn validate(&self) -> NativeExecutionResult<()> {
        self.planning.numerics.validate().map_err(protocol)?;
        self.geometry_meshing
            .validate_for_portable_host()
            .map_err(protocol)?;
        if self.schema_version != 1
            || self.authored_study_digest.bytes() == &[0; 32]
            || self
                .frozen_project_digest
                .is_some_and(|digest| digest.bytes() == &[0; 32])
            || self.study.module_path.trim().is_empty()
            || self.study.study_id.trim().is_empty()
            || self.geometry_argument_count == 0
            || self.geometry_meshing.artifact.form != ExecutableForm::MeshingWorkload
            || self.geometry_meshing.recipe.execution_mode
                != crate::GEOMETRY_MESHING_ROOT_EXECUTION_MODE
            || self.geometry_meshing.requested_outputs != 1
        {
            return Err(protocol("structural study root request is invalid"));
        }
        let mut identities = BTreeSet::new();
        if self
            .materials
            .iter()
            .any(|entry| !identities.insert(entry.reference.clone()))
        {
            return Err(protocol(
                "structural study material identities are not unique",
            ));
        }
        let materials = self
            .materials
            .iter()
            .map(|entry| entry.material.clone())
            .collect::<Vec<_>>();
        runmat_analysis_core::validate_material_definitions(&materials).map_err(protocol)?;
        validate_accelerator_requests(self.planning.backend, &self.analysis_accelerator_requests)?;
        let materializations = self
            .input_materializations
            .iter()
            .map(InputMaterializationRequest::to_prepared)
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        validate_input_materialization_inventory(
            &self.study,
            &materializations,
            &self.geometry_meshing.recipe.program_revision,
        )?;
        let reference_count = materializations
            .iter()
            .flat_map(|prepared| prepared.program().input_references.iter())
            .map(|reference| reference.id)
            .collect::<BTreeSet<_>>()
            .len();
        if usize::try_from(self.materialization_argument_count).map_err(protocol)?
            != reference_count
        {
            return Err(protocol(
                "structural study input-reference count differs from its materialization plans",
            ));
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralStudyMaterial {
    reference: AuthoredMaterialRef,
    material: MaterialModel,
}

pub struct StructuralStudyRootPreparation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub study: RealizedStudy,
    pub materials: BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub planning: StructuralStudyPlanningPolicy,
    pub analysis_accelerator_requests: Vec<AcceleratorRequest>,
    pub input_materializations: Vec<PreparedInputMaterialization>,
    pub geometry_meshing: PreparedGeometryMeshingRoot,
}

/// Client-held admission contract for a geometry-derived structural study result.
///
/// Workloads are intentionally absent: exact topology is not available until the remote geometry
/// and meshing stages finish. The importer admits the published compiled study against these
/// immutable inputs, derives the workloads locally, and then applies the ordinary exact-result
/// validation path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralStudyResultExpectation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub study: RealizedStudy,
    pub materials: Vec<StructuralStudyMaterialExpectation>,
    pub planning: StructuralStudyPlanningPolicy,
    pub analysis_accelerator_requests: Vec<AcceleratorRequest>,
    pub input_materializations: Vec<InputMaterializationExpectation>,
    pub geometry_request: runmat_analysis_execution::GeometryMaterializationRequest,
    pub expected_meshing_request: runmat_meshing_core::MeshingRequest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralStudyMaterialExpectation {
    pub reference: AuthoredMaterialRef,
    pub material: MaterialModel,
}

impl StructuralStudyResultExpectation {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        self.artifact_access.validate().map_err(protocol)?;
        self.planning.numerics.validate().map_err(protocol)?;
        self.expected_meshing_request.validate().map_err(protocol)?;
        self.geometry_request.validate().map_err(protocol)?;
        if self.authored_study_digest.bytes() == &[0; 32]
            || self
                .frozen_project_digest
                .is_some_and(|digest| digest.bytes() == &[0; 32])
        {
            return Err(protocol("structural study result expectation is invalid"));
        }
        if self.study.module_path.trim().is_empty()
            || self.study.study_id.trim().is_empty()
            || self.materials.iter().enumerate().any(|(index, material)| {
                self.materials[..index]
                    .iter()
                    .any(|existing| existing.reference == material.reference)
            })
        {
            return Err(protocol(
                "structural study result source contract is invalid",
            ));
        }
        runmat_analysis_core::validate_material_definitions(
            &self
                .materials
                .iter()
                .map(|entry| entry.material.clone())
                .collect::<Vec<_>>(),
        )
        .map_err(protocol)?;
        let materials = self
            .materials
            .iter()
            .map(|entry| (entry.reference.clone(), entry.material.clone()))
            .collect::<BTreeMap<_, _>>();
        let expected_source_digest =
            runmat_analysis_fea::study::structural_study_source_digest(&self.study, &materials)
                .map_err(protocol)?;
        let geometry_definition = runmat_geometry_core::GeometryDefinitionRef {
            module_path: self.geometry_request.definition.module_path.clone(),
            geometry_id: self.geometry_request.definition.geometry_id.clone(),
        };
        if expected_source_digest != self.authored_study_digest
            || self.study.geometry.definition != geometry_definition
            || self.study.geometry.bindings != self.geometry_request.bindings
        {
            return Err(protocol(
                "structural study result source identity differs from its geometry or materials",
            ));
        }
        validate_accelerator_requests(self.planning.backend, &self.analysis_accelerator_requests)?;
        validate_input_expectation_inventory(
            &self.study,
            &self.input_materializations,
            &self.program_revision,
        )?;
        Ok(())
    }
}

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
    pub planning: StructuralStudyPlanningPolicy,
    pub analysis_accelerator_requests: Vec<AcceleratorRequest>,
    pub input_materializations: Vec<PreparedInputMaterialization>,
    pub geometry_meshing: ProgramExecutionRequest,
    pub artifact_access: ArtifactAccess,
}

pub fn prepare_structural_study_root(
    preparation: StructuralStudyRootPreparation,
) -> NativeExecutionResult<PreparedStructuralStudyRoot> {
    let StructuralStudyRootPreparation {
        program_revision,
        artifact_access,
        authored_study_digest,
        frozen_project_digest,
        study,
        materials,
        planning,
        analysis_accelerator_requests,
        input_materializations,
        geometry_meshing,
    } = preparation;
    artifact_access.validate().map_err(protocol)?;
    if geometry_meshing.request().recipe.program_revision != program_revision {
        return Err(protocol(
            "geometry-meshing and structural-study program revisions differ",
        ));
    }
    let expected_meshing_request = geometry_meshing.meshing_request().clone();
    let geometry_request = geometry_meshing.geometry_request().clone();
    let input_materializations = input_materializations
        .into_iter()
        .map(|materialization| materialization.rebind(&artifact_access).map_err(protocol))
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    validate_input_materialization_inventory(&study, &input_materializations, &program_revision)?;
    let mut input_value_objects = BTreeMap::new();
    for materialization in &input_materializations {
        materialization.validate_closed().map_err(protocol)?;
        for object in materialization.input_objects() {
            match input_value_objects.get(&object.reference.id) {
                Some(existing) if existing != object => {
                    return Err(protocol(
                        "generator input value id collides with different immutable content",
                    ));
                }
                Some(_) => {}
                None => {
                    input_value_objects.insert(object.reference.id, object.clone());
                }
            }
        }
    }
    let input_expectations = input_materializations
        .iter()
        .map(PreparedInputMaterialization::expectation)
        .collect::<Result<Vec<_>, _>>()
        .map_err(protocol)?;
    let root = StructuralStudyRootRequest {
        schema_version: 1,
        authored_study_digest,
        frozen_project_digest,
        study,
        materials: materials
            .into_iter()
            .map(|(reference, material)| StructuralStudyMaterial {
                reference,
                material,
            })
            .collect(),
        planning,
        analysis_accelerator_requests,
        input_materializations: input_materializations
            .iter()
            .map(InputMaterializationRequest::from_prepared)
            .collect::<NativeExecutionResult<Vec<_>>>()?,
        geometry_meshing: descriptor_from_request(geometry_meshing.request()),
        geometry_argument_count: u32::try_from(geometry_meshing.request().arguments.len())
            .map_err(protocol)?,
        materialization_argument_count: u32::try_from(input_value_objects.len())
            .map_err(protocol)?,
    };
    let root_object = LogicalObject::new(
        ObjectNamespace::InputValue,
        "analysis/structural-study-roots",
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
        execution_mode: STRUCTURAL_STUDY_ROOT_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(STRUCTURAL_STUDY_ROOT_TARGET_PROFILE),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
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
    arguments.extend(geometry_meshing.request().arguments.clone());
    arguments.extend(
        input_value_objects
            .values()
            .map(|object| ValuePayload::Object(Box::new(object.reference.clone()))),
    );
    let request = ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe: recipe.clone(),
        artifact: artifact.clone(),
        function: 0,
        arguments,
        requested_outputs: 1,
    };
    request.validate_for_portable_host().map_err(protocol)?;
    let mut objects = geometry_meshing.objects().to_vec();
    objects.push(root_object);
    let mut required_programs = vec![RequiredStructuralStudyProgram { recipe, artifact }];
    required_programs.extend(
        geometry_meshing
            .required_programs()
            .iter()
            .map(required_program),
    );
    for materialization in &input_materializations {
        required_programs.push(RequiredStructuralStudyProgram {
            recipe: materialization.program().descriptor.recipe.clone(),
            artifact: materialization.program().descriptor.artifact.clone(),
        });
    }
    let analysis_host =
        prepare_analysis_host_program(request.recipe.program_revision.clone()).map_err(protocol)?;
    required_programs.push(RequiredStructuralStudyProgram {
        recipe: analysis_host.recipe,
        artifact: analysis_host.artifact,
    });
    let mut seen = BTreeSet::new();
    required_programs.retain(|program| seen.insert(program.artifact.id));
    let expectation = StructuralStudyResultExpectation {
        program_revision: request.recipe.program_revision.clone(),
        artifact_access,
        authored_study_digest: root.authored_study_digest,
        frozen_project_digest: root.frozen_project_digest,
        study: root.study.clone(),
        materials: root
            .materials
            .iter()
            .map(|entry| StructuralStudyMaterialExpectation {
                reference: entry.reference.clone(),
                material: entry.material.clone(),
            })
            .collect(),
        planning: root.planning,
        analysis_accelerator_requests: root.analysis_accelerator_requests.clone(),
        input_materializations: input_expectations,
        geometry_request,
        expected_meshing_request,
    };
    expectation.validate()?;
    Ok(PreparedStructuralStudyRoot {
        request,
        objects,
        input_value_objects: input_value_objects.into_values().collect(),
        required_programs,
        authored_study_digest: root.authored_study_digest,
        expectation,
    })
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
    let component_count = geometry_count
        .checked_add(materialization_count)
        .ok_or_else(|| protocol("structural study component count overflows"))?;
    let arguments = program
        .arguments
        .get(1..)
        .filter(|arguments| arguments.len() == component_count)
        .ok_or_else(|| protocol("structural study component count differs"))?;
    let (geometry_arguments, materialization_arguments) = arguments.split_at(geometry_count);
    let geometry_meshing = ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe: root.geometry_meshing.recipe,
        artifact: root.geometry_meshing.artifact,
        function: root.geometry_meshing.function,
        arguments: geometry_arguments.to_vec(),
        requested_outputs: root.geometry_meshing.requested_outputs,
    };
    decode_geometry_meshing_root(&geometry_meshing, store)?;
    let input_materializations = root
        .input_materializations
        .iter()
        .map(InputMaterializationRequest::to_prepared)
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let expected_materialization_references = input_materializations
        .iter()
        .flat_map(|prepared| prepared.program().input_references.iter())
        .map(|reference| (reference.id, reference))
        .collect::<BTreeMap<_, _>>();
    let transported_materialization_references = materialization_arguments
        .iter()
        .map(|argument| match argument {
            ValuePayload::Object(reference) => Ok((reference.id, reference.as_ref())),
            ValuePayload::Inline(_) => Err(protocol("generator input value is not externalized")),
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
        analysis_accelerator_requests: root.analysis_accelerator_requests,
        input_materializations,
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
        schema_version: request.schema_version,
        recipe: request.recipe.clone(),
        artifact: request.artifact.clone(),
        function: request.function,
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
            "CPU structural planning cannot require analysis accelerators",
        )),
        runmat_analysis_fea::ComputeBackend::Gpu if requests.is_empty() => Err(protocol(
            "accelerator structural planning requires an exact allocation request",
        )),
        _ => Ok(()),
    }
}

fn validate_input_materialization_inventory(
    study: &RealizedStudy,
    materializations: &[PreparedInputMaterialization],
    revision: &ProgramRevision,
) -> NativeExecutionResult<()> {
    let models = study
        .included_models
        .iter()
        .chain(std::iter::once(&study.model))
        .collect::<Vec<_>>();
    let mut signals = models
        .iter()
        .flat_map(|model| &model.signals)
        .filter(|signal| signal.generator().is_some());
    let mut responses = models
        .iter()
        .flat_map(|model| &model.response_spectra)
        .filter(|spectrum| spectrum.ordinates.generator().is_some());
    let mut random = models
        .iter()
        .flat_map(|model| &model.random_excitations)
        .filter(|spectrum| spectrum.spectral_density.generator().is_some());
    let expected_count =
        signals.clone().count() + responses.clone().count() + random.clone().count();
    if expected_count != materializations.len() {
        return Err(protocol(
            "study input materializations differ from its authored generator inventory",
        ));
    }
    let mut identities = BTreeSet::new();
    for prepared in materializations {
        prepared.validate().map_err(protocol)?;
        let (materialization_revision, identity, authored_matches) = match prepared {
            PreparedInputMaterialization::Signal(prepared) => (
                &prepared.admitted.program_revision,
                prepared.admitted.materialization_identity.digest(),
                signals.next() == Some(&prepared.authored),
            ),
            PreparedInputMaterialization::Numerical(prepared) => {
                let authored_matches = match &prepared.authored {
                    runmat_analysis_execution::NumericalGeneratorSource::ResponseSpectrum(
                        spectrum,
                    ) => responses.next() == Some(spectrum.as_ref()),
                    runmat_analysis_execution::NumericalGeneratorSource::RandomSpectrum(
                        spectrum,
                    ) => random.next() == Some(spectrum.as_ref()),
                };
                (
                    &prepared.admitted.invocation.program_revision,
                    prepared.admitted.materialization_identity.digest(),
                    authored_matches,
                )
            }
        };
        if materialization_revision != revision || !authored_matches || !identities.insert(identity)
        {
            return Err(protocol(
                "study input materialization inventory is inconsistent or duplicated",
            ));
        }
    }
    Ok(())
}

fn validate_input_expectation_inventory(
    study: &RealizedStudy,
    expectations: &[InputMaterializationExpectation],
    revision: &ProgramRevision,
) -> NativeExecutionResult<()> {
    for expectation in expectations {
        expectation.validate().map_err(protocol)?;
    }
    let models = study
        .included_models
        .iter()
        .chain(std::iter::once(&study.model))
        .collect::<Vec<_>>();
    let expected = models
        .iter()
        .flat_map(|model| &model.signals)
        .filter(|signal| signal.generator().is_some())
        .map(InputExpectationSource::Signal)
        .chain(
            models
                .iter()
                .flat_map(|model| &model.response_spectra)
                .filter(|spectrum| spectrum.ordinates.generator().is_some())
                .map(InputExpectationSource::Response),
        )
        .chain(
            models
                .iter()
                .flat_map(|model| &model.random_excitations)
                .filter(|spectrum| spectrum.spectral_density.generator().is_some())
                .map(InputExpectationSource::Random),
        )
        .collect::<Vec<_>>();
    let actual = expectations
        .iter()
        .map(|expectation| match expectation {
            InputMaterializationExpectation::Signal(value) => (
                &value.admitted.program_revision,
                InputExpectationSource::Signal(&value.authored),
            ),
            InputMaterializationExpectation::Numerical(value) => {
                let source = match &value.authored {
                    runmat_analysis_execution::NumericalGeneratorSource::ResponseSpectrum(
                        value,
                    ) => InputExpectationSource::Response(value),
                    runmat_analysis_execution::NumericalGeneratorSource::RandomSpectrum(value) => {
                        InputExpectationSource::Random(value)
                    }
                };
                (&value.admitted.invocation.program_revision, source)
            }
        })
        .collect::<Vec<_>>();
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|((actual_revision, actual), expected)| {
                *actual_revision != revision || *actual != expected
            })
    {
        return Err(protocol(
            "study input expectations differ from its authored generator inventory",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum InputExpectationSource<'a> {
    Signal(&'a runmat_analysis_core::PhysicalSignal),
    Response(&'a runmat_analysis_core::ResponseSpectrum),
    Random(&'a runmat_analysis_core::RandomExcitationSpectrum),
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(test)]
#[path = "structural_study_root_tests.rs"]
mod tests;
