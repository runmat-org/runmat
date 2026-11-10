//! Portable result closure of a remotely executed family study.
//!
//! The executing driver publishes one canonical wire naming the exact geometry receipt, the
//! meshing result closure, any generated-input closures and the family's canonical completion
//! root. The client admits each component against its own source contract and replays the
//! completion from the downloaded objects; no executing host state is trusted.

use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_execution::GeometryMaterializationOutput;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, LogicalObject, ObjectNamespace, ProgramExecutionObject,
    ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_meshing_core::CanonicalMeshingContract;
use runmat_meshing_execution::ExactMeshingDagRunResult;
use serde::{Deserialize, Serialize};

use crate::{
    CompletedFamilyStudy, CompletedFeaStudyResult, DecodedStructuralStudyRoot,
    NativeExecutionError, NativeExecutionResult, StructuralStudyResultExpectation, StudyFamilyKind,
};

const PREFIX: &[u8] = b"runmat-family-study-publication/v1\0";
const DOMAIN: &str = "family-study-publication-v1";
const MEDIA_TYPE: &str = "application/vnd.runmat.family-study-publication.v1+cbor";
const VALUE_SCHEMA: &str = "runmat.analysis.family-study-publication.v1";
const LIMITS: CanonicalLimits = CanonicalLimits::platform_content(65_536, 64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyStudyPublication {
    schema_version: u16,
    family: StudyFamilyKind,
    program_revision: ProgramRevision,
    authored_study_digest: Digest,
    compiled_geometry_output: GeometryMaterializationOutput,
    meshing_request_digest: Digest,
    meshing_root: ValueRef,
    meshing_objects: Vec<ValueRef>,
    input_objects: Vec<ValueRef>,
    completion_root: ValueRef,
    completion_objects: Vec<ValueRef>,
}

impl FamilyStudyPublication {
    fn validate(&self, access: &ArtifactAccess) -> NativeExecutionResult<()> {
        if self.schema_version != 1 || self.authored_study_digest.bytes() == &[0; 32] {
            return Err(protocol("family study publication is invalid"));
        }
        for reference in self
            .meshing_objects
            .iter()
            .chain(&self.input_objects)
            .chain(&self.completion_objects)
            .chain([&self.meshing_root, &self.completion_root])
        {
            if reference.authorization_scope != access.authorization_scope
                || reference.encryption_context != access.encryption_context
            {
                return Err(protocol(
                    "family study publication references foreign artifact authority",
                ));
            }
        }
        if !self.meshing_objects.contains(&self.meshing_root)
            || !self.completion_objects.contains(&self.completion_root)
        {
            return Err(protocol(
                "family study publication roots are absent from their closures",
            ));
        }
        Ok(())
    }

    fn canonical_bytes(&self, access: &ArtifactAccess) -> NativeExecutionResult<Vec<u8>> {
        self.validate(access)?;
        encode_contract(PREFIX, DOMAIN, self, LIMITS).map_err(protocol)
    }

    fn decode(bytes: &[u8], access: &ArtifactAccess) -> NativeExecutionResult<Self> {
        let publication: Self = decode_contract(PREFIX, DOMAIN, bytes, LIMITS).map_err(protocol)?;
        publication.validate(access)?;
        Ok(publication)
    }
}

/// Package a completed family for the client: one root output plus its closed inventory.
pub(crate) fn package_family_study_result(
    request: &ProgramExecutionRequest,
    decoded: &DecodedStructuralStudyRoot,
    geometry_output: &GeometryMaterializationOutput,
    meshing_request: &runmat_meshing_core::MeshingRequest,
    meshing: &ExactMeshingDagRunResult,
    inputs: &[crate::input_materialization::CompletedInputMaterialization],
    completed: &CompletedFamilyStudy,
) -> NativeExecutionResult<(ProgramExecutionResponse, Vec<ProgramExecutionObject>)> {
    let access = &decoded.artifact_access;
    let completion = completed.prepare_root()?;
    let input_objects = unique(
        inputs
            .iter()
            .map(|input| {
                input
                    .objects
                    .references(access, ValueRefKind::ResultObject)
                    .map_err(protocol)
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?
            .into_iter()
            .flatten(),
    )?;
    let meshing_request_digest = Digest::from_bytes(
        *meshing_request
            .canonical_digest()
            .map_err(protocol)?
            .bytes(),
    );
    let publication = FamilyStudyPublication {
        schema_version: 1,
        family: completed.kind(),
        program_revision: request.recipe.program_revision.clone(),
        authored_study_digest: decoded.authored_study_digest,
        compiled_geometry_output: geometry_output.clone(),
        meshing_request_digest,
        meshing_root: meshing.root.clone(),
        meshing_objects: meshing.result_objects.clone(),
        input_objects: input_objects.clone(),
        completion_root: completion.root.clone(),
        completion_objects: completion.catalog.objects.clone(),
    };
    let publication_object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "analysis/family-study-publications",
        MEDIA_TYPE,
        publication.canonical_bytes(access)?,
    )
    .map_err(protocol)?;
    let publication_root = access
        .reference(
            &publication_object,
            ValueRefKind::ResultObject,
            VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let inventory = unique(
        meshing
            .result_objects
            .iter()
            .chain(&input_objects)
            .chain(&completion.catalog.objects)
            .chain(std::iter::once(&publication_root))
            .cloned(),
    )?;
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(publication_root.clone()))],
        result_objects: inventory,
    };
    response.validate_against(request).map_err(protocol)?;
    let mut new_objects = vec![ProgramExecutionObject {
        reference: publication_root,
        bytes: publication_object.bytes,
    }];
    for input in inputs {
        let references = input
            .objects
            .references(access, ValueRefKind::ResultObject)
            .map_err(protocol)?;
        for (object, reference) in input.objects.objects.iter().zip(references) {
            new_objects.push(ProgramExecutionObject {
                reference,
                bytes: object.bytes.clone(),
            });
        }
    }
    let catalog = completion
        .catalog
        .objects
        .iter()
        .map(|reference| (reference.logical_digest, reference))
        .collect::<BTreeMap<_, _>>();
    for object in completion.objects {
        let reference = catalog
            .get(&object.descriptor.digest)
            .ok_or_else(|| protocol("family completion object is missing from its catalog"))?;
        new_objects.push(ProgramExecutionObject {
            reference: (*reference).clone(),
            bytes: object.bytes,
        });
    }
    for object in &new_objects {
        object.validate().map_err(protocol)?;
    }
    Ok((response, new_objects))
}

/// Admit a remote family result against the client-held source contract and replay it.
pub fn import_family_study_result(
    source: impl CacheImport + Clone + Send + Sync + 'static,
    response: ProgramExecutionResponse,
    expected: &StructuralStudyResultExpectation,
    limits: ObjectInventoryLimits,
) -> NativeExecutionResult<CompletedFeaStudyResult> {
    expected.validate()?;
    let family = StudyFamilyKind::of(&expected.planning)
        .ok_or_else(|| protocol("structural expectation cannot import a family result"))?;
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = &response
    else {
        return Err(protocol(
            "family study execution did not return a result closure",
        ));
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err(protocol("family study result root is invalid"));
    };
    let access = &expected.artifact_access;
    access
        .validate_reference(root, ValueRefKind::ResultObject, MEDIA_TYPE, VALUE_SCHEMA)
        .map_err(protocol)?;
    let publication = FamilyStudyPublication::decode(&read(&source, root)?, access)?;
    if publication.family != family
        || publication.program_revision != expected.program_revision
        || publication.authored_study_digest != expected.authored_study_digest
    {
        return Err(protocol(
            "family study result differs from its submitted source identity",
        ));
    }
    let declared = publication
        .meshing_objects
        .iter()
        .chain(&publication.input_objects)
        .chain(&publication.completion_objects)
        .chain(std::iter::once(root.as_ref()))
        .map(|reference| reference.logical_digest)
        .collect::<BTreeSet<_>>();
    let returned = result_objects
        .iter()
        .map(|reference| reference.logical_digest)
        .collect::<BTreeSet<_>>();
    if declared != returned {
        return Err(protocol(
            "family study result inventory differs from its publication closure",
        ));
    }
    if publication.meshing_request_digest
        != Digest::from_bytes(
            *expected
                .expected_meshing_request
                .canonical_digest()
                .map_err(protocol)?
                .bytes(),
        )
    {
        return Err(protocol(
            "family study result was meshed under a different request",
        ));
    }
    publication
        .compiled_geometry_output
        .validate_against(&expected.geometry_request)
        .map_err(protocol)?;
    let mesh = runmat_meshing_execution::import_exact_meshing_result(
        &source,
        publication.meshing_root.clone(),
        publication.meshing_objects.clone(),
        access.clone(),
        limits,
    )
    .map_err(protocol)?;
    let materials = expected
        .materials
        .iter()
        .map(|entry| (entry.reference.clone(), entry.material.clone()))
        .collect::<BTreeMap<_, _>>();
    let recompiled = runmat_analysis_fea::study::compile_analysis_study(
        runmat_analysis_fea::study::AnalysisStudyCompilationInput {
            study: &expected.study,
            geometry: &publication.compiled_geometry_output.document,
            regions: &publication.compiled_geometry_output.regions,
            materials: &materials,
            mesh: &mesh.artifact,
        },
    )
    .map_err(protocol)?;
    let maximum_resident_bytes = recompiled
        .model
        .sequence
        .steps
        .first()
        .map(|step| step.accuracy.resources.resident_byte_budget())
        .ok_or_else(|| protocol("family study has no authored step"))?;
    let result = crate::imported_family_result(
        family,
        source,
        &publication.completion_root,
        access,
        crate::FamilyResultImportLimits {
            objects: limits,
            maximum_resident_bytes,
        },
        mesh.artifact.clone(),
        mesh.evidence.clone(),
    )
    .map_err(protocol)?;
    let compiled = result
        .family_compiled_study()
        .ok_or_else(|| protocol("imported family result has no compiled study"))?;
    // Generated inputs rewrite execution signals only; every other compiled authority must
    // equal the client's own compilation over the admitted geometry and mesh.
    let admitted = if expected.input_materializations.is_empty() {
        compiled == &recompiled
    } else {
        compiled.study_id == recompiled.study_id && compiled.properties == recompiled.properties
    };
    if !admitted {
        return Err(protocol(
            "family study result differs from the independently compiled source contract",
        ));
    }
    Ok(result)
}

fn unique(references: impl IntoIterator<Item = ValueRef>) -> NativeExecutionResult<Vec<ValueRef>> {
    let mut by_digest = BTreeMap::new();
    for reference in references {
        if let Some(existing) = by_digest.insert(reference.logical_digest, reference.clone()) {
            if existing != reference {
                return Err(protocol(
                    "family study closure contains conflicting references",
                ));
            }
        }
    }
    Ok(by_digest.into_values().collect())
}

fn read(source: &impl CacheImport, reference: &ValueRef) -> NativeExecutionResult<Vec<u8>> {
    let bytes = source
        .read_verified(reference.logical_digest)
        .map_err(protocol)?
        .ok_or_else(|| protocol("family study publication is unavailable"))?;
    if reference.logical_digest != Digest::sha256(&bytes)
        || reference.encoded_length != bytes.len() as u64
    {
        return Err(protocol(
            "family study publication differs from its reference",
        ));
    }
    Ok(bytes)
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
