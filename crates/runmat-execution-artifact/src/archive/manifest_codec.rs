use std::collections::BTreeSet;

use minicbor::{Decoder, Encoder};
use runmat_execution::resource::Capability;
use runmat_execution::{Digest, OutputContract, ProgramRevision};

use crate::{
    ArtifactError, ArtifactResult, BuildResourceDeclaration, BundleCallable, BundleManifest,
    ExecutableForm, ObjectDescriptor, ObjectNamespace, ProgramArtifact, ProgramArtifactId,
    ProgramBuildRecipe, ProgramRecipeId, ProjectRevisionRecord,
};

const MAX_ITEMS: usize = 100_000;
const MAX_TEXT: usize = 4096;
const MAX_CODE_CLOSURE_BYTES: usize = 16 * 1024 * 1024;

pub(super) fn encode_manifest(manifest: &BundleManifest) -> ArtifactResult<Vec<u8>> {
    let code_closure = serde_json::to_vec(&manifest.code_closure)
        .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    if code_closure.len() > MAX_CODE_CLOSURE_BYTES {
        return Err(ArtifactError::Limit(
            "bundle code closure is too large".to_string(),
        ));
    }
    let mut bytes = b"runmat-execution-bundle-manifest-v3\0".to_vec();
    let mut encoder = Encoder::new(&mut bytes);
    encoder
        .array(11)
        .and_then(|encoder| encoder.u16(manifest.schema_version))
        .and_then(|encoder| {
            encoder.bytes(
                &manifest
                    .program_revision
                    .canonical_bytes()
                    .map_err(|_| minicbor::encode::Error::message("invalid program revision"))?,
            )
        })
        .and_then(|encoder| encoder.array(2))
        .and_then(|encoder| encoder.bytes(manifest.project_revision.graph_digest.bytes()))
        .and_then(|encoder| encoder.bytes(manifest.project_revision.source_digest.bytes()))
        .and_then(|encoder| encoder.bytes(&code_closure))
        .and_then(|encoder| encoder.array(manifest.sources.len() as u64))
        .map_err(encode_error)?;
    for source in &manifest.sources {
        encode_descriptor_to(&mut encoder, source)?;
    }
    encoder
        .array(manifest.callables.len() as u64)
        .map_err(encode_error)?;
    for callable in &manifest.callables {
        encoder
            .array(3)
            .and_then(|encoder| encoder.str(&callable.owner_identity))
            .and_then(|encoder| encoder.str(&callable.qualified_name))
            .and_then(|encoder| encoder.bytes(callable.source_digest.bytes()))
            .map_err(encode_error)?;
    }
    encoder
        .array(manifest.recipes.len() as u64)
        .map_err(encode_error)?;
    for recipe in &manifest.recipes {
        encode_recipe(&mut encoder, recipe)?;
    }
    encoder
        .array(manifest.artifacts.len() as u64)
        .map_err(encode_error)?;
    for artifact in &manifest.artifacts {
        encoder
            .array(6)
            .and_then(|encoder| encoder.u16(artifact.schema_version))
            .and_then(|encoder| encoder.bytes(artifact.id.0.bytes()))
            .and_then(|encoder| encoder.bytes(artifact.recipe_id.0.bytes()))
            .and_then(|encoder| {
                encoder.bytes(
                    &artifact
                        .target
                        .canonical_bytes()
                        .map_err(|_| minicbor::encode::Error::message("invalid program target"))?,
                )
            })
            .and_then(|encoder| encoder.u8(artifact.form as u8))
            .and_then(|encoder| encoder.bytes(&artifact.executable_bytes))
            .map_err(encode_error)?;
    }
    encoder
        .array(manifest.requested_capabilities.len() as u64)
        .map_err(encode_error)?;
    for capability in &manifest.requested_capabilities {
        encode_capability(&mut encoder, capability)?;
    }
    encoder
        .array(3)
        .and_then(|encoder| encoder.u32(manifest.resources.cpu_millicores))
        .and_then(|encoder| encoder.u64(manifest.resources.memory_bytes))
        .and_then(|encoder| encoder.u64(manifest.resources.scratch_bytes))
        .and_then(|encoder| encoder.array(manifest.portable_environment.len() as u64))
        .map_err(encode_error)?;
    for (name, value) in &manifest.portable_environment {
        encoder
            .array(2)
            .and_then(|encoder| encoder.str(name))
            .and_then(|encoder| encoder.str(value))
            .map_err(encode_error)?;
    }
    Ok(bytes)
}

pub(super) fn decode_manifest(bytes: &[u8]) -> ArtifactResult<BundleManifest> {
    let payload = bytes
        .strip_prefix(b"runmat-execution-bundle-manifest-v3\0")
        .ok_or_else(|| ArtifactError::Invalid("invalid bundle manifest domain".into()))?;
    let mut decoder = Decoder::new(payload);
    require_len(decoder.array(), 11, "bundle manifest")?;
    let schema_version = decoder.u16().map_err(decode_error)?;
    let revision = decoder.bytes().map_err(decode_error)?;
    let program_revision = ProgramRevision::from_canonical_bytes(revision)
        .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    require_len(decoder.array(), 2, "project revision")?;
    let project_revision = ProjectRevisionRecord {
        graph_digest: decode_digest(&mut decoder)?,
        source_digest: decode_digest(&mut decoder)?,
    };
    let code_closure_bytes = decoder.bytes().map_err(decode_error)?;
    if code_closure_bytes.len() > MAX_CODE_CLOSURE_BYTES {
        return Err(ArtifactError::Limit(
            "bundle code closure is too large".to_string(),
        ));
    }
    let code_closure = serde_json::from_slice(code_closure_bytes)
        .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    let source_count = bounded_len(decoder.array(), "sources")?;
    let mut sources = Vec::with_capacity(source_count);
    for _ in 0..source_count {
        sources.push(decode_descriptor_from(&mut decoder)?);
    }
    let callable_count = bounded_len(decoder.array(), "callables")?;
    let mut callables = Vec::with_capacity(callable_count);
    for _ in 0..callable_count {
        require_len(decoder.array(), 3, "callable")?;
        callables.push(BundleCallable {
            owner_identity: decode_text(&mut decoder)?,
            qualified_name: decode_text(&mut decoder)?,
            source_digest: decode_digest(&mut decoder)?,
        });
    }
    let recipe_count = bounded_len(decoder.array(), "recipes")?;
    let mut recipes = Vec::with_capacity(recipe_count);
    for _ in 0..recipe_count {
        recipes.push(decode_recipe(&mut decoder)?);
    }
    let artifact_count = bounded_len(decoder.array(), "artifacts")?;
    let mut artifacts = Vec::with_capacity(artifact_count);
    for _ in 0..artifact_count {
        require_len(decoder.array(), 6, "program artifact")?;
        artifacts.push(ProgramArtifact {
            schema_version: decoder.u16().map_err(decode_error)?,
            id: ProgramArtifactId(decode_digest(&mut decoder)?),
            recipe_id: ProgramRecipeId(decode_digest(&mut decoder)?),
            target: crate::ProgramTarget::from_canonical_bytes(
                decoder.bytes().map_err(decode_error)?,
            )?,
            form: decode_form(decoder.u8().map_err(decode_error)?)?,
            executable_bytes: decoder.bytes().map_err(decode_error)?.to_vec(),
        });
    }
    let capability_count = bounded_len(decoder.array(), "capabilities")?;
    let mut requested_capabilities = BTreeSet::new();
    for _ in 0..capability_count {
        if !requested_capabilities.insert(decode_capability(&mut decoder)?) {
            return Err(ArtifactError::Invalid("duplicate capability".into()));
        }
    }
    require_len(decoder.array(), 3, "resources")?;
    let resources = BuildResourceDeclaration {
        cpu_millicores: decoder.u32().map_err(decode_error)?,
        memory_bytes: decoder.u64().map_err(decode_error)?,
        scratch_bytes: decoder.u64().map_err(decode_error)?,
    };
    let environment_count = bounded_len(decoder.array(), "portable environment")?;
    let mut portable_environment = Vec::with_capacity(environment_count);
    for _ in 0..environment_count {
        require_len(decoder.array(), 2, "environment entry")?;
        portable_environment.push((decode_text(&mut decoder)?, decode_text(&mut decoder)?));
    }
    if decoder.position() != payload.len() {
        return Err(ArtifactError::Invalid(
            "bundle manifest contains trailing CBOR data".into(),
        ));
    }
    Ok(BundleManifest {
        schema_version,
        program_revision,
        project_revision,
        code_closure,
        sources,
        callables,
        recipes,
        artifacts,
        requested_capabilities,
        resources,
        portable_environment,
    })
}

pub(super) fn encode_descriptor(descriptor: &ObjectDescriptor) -> ArtifactResult<Vec<u8>> {
    let mut bytes = Vec::new();
    encode_descriptor_to(&mut Encoder::new(&mut bytes), descriptor)?;
    Ok(bytes)
}

pub(super) fn decode_descriptor(bytes: &[u8]) -> ArtifactResult<ObjectDescriptor> {
    let mut decoder = Decoder::new(bytes);
    let descriptor = decode_descriptor_from(&mut decoder)?;
    if decoder.position() != bytes.len() {
        return Err(ArtifactError::Invalid(
            "object descriptor contains trailing CBOR data".into(),
        ));
    }
    Ok(descriptor)
}

fn encode_descriptor_to(
    encoder: &mut Encoder<&mut Vec<u8>>,
    descriptor: &ObjectDescriptor,
) -> ArtifactResult<()> {
    encoder
        .array(5)
        .and_then(|encoder| encoder.u8(descriptor.namespace as u8))
        .and_then(|encoder| encoder.str(&descriptor.logical_name))
        .and_then(|encoder| encoder.bytes(descriptor.digest.bytes()))
        .and_then(|encoder| encoder.u64(descriptor.encoded_length))
        .and_then(|encoder| encoder.str(&descriptor.media_type))
        .map(|_| ())
        .map_err(encode_error)
}

fn decode_descriptor_from(decoder: &mut Decoder<'_>) -> ArtifactResult<ObjectDescriptor> {
    require_len(decoder.array(), 5, "object descriptor")?;
    let namespace = decode_namespace(decoder.u8().map_err(decode_error)?)?;
    let descriptor = ObjectDescriptor {
        namespace,
        logical_name: decode_text(decoder)?,
        digest: decode_digest(decoder)?,
        encoded_length: decoder.u64().map_err(decode_error)?,
        media_type: decode_text(decoder)?,
    };
    descriptor.validate()?;
    Ok(descriptor)
}

fn encode_recipe(
    encoder: &mut Encoder<&mut Vec<u8>>,
    recipe: &ProgramBuildRecipe,
) -> ArtifactResult<()> {
    encoder
        .array(10)
        .and_then(|encoder| encoder.u16(recipe.schema_version))
        .and_then(|encoder| {
            encoder.bytes(
                &recipe
                    .program_revision
                    .canonical_bytes()
                    .map_err(|_| minicbor::encode::Error::message("invalid program revision"))?,
            )
        })
        .and_then(|encoder| encoder.str(&recipe.entrypoint))
        .and_then(|encoder| encoder.u16(recipe.outputs.requested_outputs))
        .and_then(|encoder| encoder.str(&recipe.execution_mode))
        .and_then(|encoder| {
            encoder.bytes(
                &recipe
                    .target
                    .canonical_bytes()
                    .map_err(|_| minicbor::encode::Error::message("invalid program target"))?,
            )
        })
        .and_then(|encoder| encoder.array(recipe.features.len() as u64))
        .map_err(encode_error)?;
    for feature in &recipe.features {
        encoder.str(feature).map_err(encode_error)?;
    }
    encoder
        .array(recipe.compile_options.len() as u64)
        .map_err(encode_error)?;
    for option in &recipe.compile_options {
        encoder.str(option).map_err(encode_error)?;
    }
    encoder
        .array(recipe.source_objects.len() as u64)
        .map_err(encode_error)?;
    for source in &recipe.source_objects {
        encode_descriptor_to(encoder, source)?;
    }
    match recipe.expected_artifact_id {
        Some(id) => encoder.bytes(id.0.bytes()).map_err(encode_error)?,
        None => encoder.null().map_err(encode_error)?,
    };
    Ok(())
}

fn decode_recipe(decoder: &mut Decoder<'_>) -> ArtifactResult<ProgramBuildRecipe> {
    require_len(decoder.array(), 10, "program recipe")?;
    let schema_version = decoder.u16().map_err(decode_error)?;
    let program_revision =
        ProgramRevision::from_canonical_bytes(decoder.bytes().map_err(decode_error)?)
            .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    let entrypoint = decode_text(decoder)?;
    let requested_outputs = decoder.u16().map_err(decode_error)?;
    let execution_mode = decode_text(decoder)?;
    let target =
        crate::ProgramTarget::from_canonical_bytes(decoder.bytes().map_err(decode_error)?)?;
    let feature_count = bounded_len(decoder.array(), "recipe features")?;
    let mut features = BTreeSet::new();
    for _ in 0..feature_count {
        if !features.insert(decode_text(decoder)?) {
            return Err(ArtifactError::Invalid("duplicate recipe feature".into()));
        }
    }
    let option_count = bounded_len(decoder.array(), "compile options")?;
    let mut compile_options = BTreeSet::new();
    for _ in 0..option_count {
        if !compile_options.insert(decode_text(decoder)?) {
            return Err(ArtifactError::Invalid("duplicate compile option".into()));
        }
    }
    let source_count = bounded_len(decoder.array(), "recipe sources")?;
    let mut source_objects = Vec::with_capacity(source_count);
    for _ in 0..source_count {
        source_objects.push(decode_descriptor_from(decoder)?);
    }
    let expected_artifact_id = match decoder.datatype().map_err(decode_error)? {
        minicbor::data::Type::Null => {
            decoder.null().map_err(decode_error)?;
            None
        }
        minicbor::data::Type::Bytes => Some(ProgramArtifactId(decode_digest(decoder)?)),
        _ => {
            return Err(ArtifactError::Invalid(
                "invalid expected program artifact identity".into(),
            ))
        }
    };
    let recipe = ProgramBuildRecipe {
        schema_version,
        program_revision,
        entrypoint,
        outputs: OutputContract { requested_outputs },
        execution_mode,
        target,
        features,
        compile_options,
        source_objects,
        expected_artifact_id,
    };
    recipe.validate()?;
    Ok(recipe)
}

fn encode_capability(
    encoder: &mut Encoder<&mut Vec<u8>>,
    capability: &Capability,
) -> ArtifactResult<()> {
    match capability {
        Capability::ProcessIsolation => encoder.array(1).and_then(|encoder| encoder.u8(0)),
        Capability::BrowserWorker => encoder.array(1).and_then(|encoder| encoder.u8(1)),
        Capability::NetworkDenied => encoder.array(1).and_then(|encoder| encoder.u8(2)),
        Capability::Accelerator(value) => encoder
            .array(2)
            .and_then(|encoder| encoder.u8(3))
            .and_then(|encoder| encoder.str(value)),
        Capability::Custom(value) => encoder
            .array(2)
            .and_then(|encoder| encoder.u8(4))
            .and_then(|encoder| encoder.str(value)),
    }
    .map(|_| ())
    .map_err(encode_error)
}

fn decode_capability(decoder: &mut Decoder<'_>) -> ArtifactResult<Capability> {
    let length = definite_len(decoder.array(), "capability")?;
    let tag = decoder.u8().map_err(decode_error)?;
    match (tag, length) {
        (0, 1) => Ok(Capability::ProcessIsolation),
        (1, 1) => Ok(Capability::BrowserWorker),
        (2, 1) => Ok(Capability::NetworkDenied),
        (3, 2) => Ok(Capability::Accelerator(decode_text(decoder)?)),
        (4, 2) => Ok(Capability::Custom(decode_text(decoder)?)),
        _ => Err(ArtifactError::Invalid(
            "invalid execution capability".into(),
        )),
    }
}

fn decode_namespace(value: u8) -> ArtifactResult<ObjectNamespace> {
    match value {
        0 => Ok(ObjectNamespace::ProgramSource),
        1 => Ok(ObjectNamespace::PackageRelease),
        2 => Ok(ObjectNamespace::ProgramArtifact),
        3 => Ok(ObjectNamespace::InputValue),
        4 => Ok(ObjectNamespace::ResultValue),
        5 => Ok(ObjectNamespace::DetailedEvent),
        6 => Ok(ObjectNamespace::Log),
        7 => Ok(ObjectNamespace::Checkpoint),
        _ => Err(ArtifactError::Invalid("invalid object namespace".into())),
    }
}

fn decode_form(value: u8) -> ArtifactResult<ExecutableForm> {
    match value {
        0 => Ok(ExecutableForm::InterpreterBytecodeV1),
        1 => Ok(ExecutableForm::InterpreterScriptV1),
        2 => Ok(ExecutableForm::TestAttemptV1),
        3 => Ok(ExecutableForm::ExecutableUnitV3),
        4 => Ok(ExecutableForm::NativeObjectV1),
        _ => Err(ArtifactError::Invalid("invalid executable form".into())),
    }
}

fn decode_digest(decoder: &mut Decoder<'_>) -> ArtifactResult<Digest> {
    let bytes: [u8; 32] = decoder
        .bytes()
        .map_err(decode_error)?
        .try_into()
        .map_err(|_| ArtifactError::Invalid("digest must contain exactly 32 bytes".into()))?;
    Ok(Digest::from_bytes(bytes))
}

fn decode_text(decoder: &mut Decoder<'_>) -> ArtifactResult<String> {
    let value = decoder.str().map_err(decode_error)?;
    if value.len() > MAX_TEXT {
        return Err(ArtifactError::Limit("manifest text is too long".into()));
    }
    Ok(value.to_owned())
}

fn bounded_len(
    length: Result<Option<u64>, minicbor::decode::Error>,
    field: &'static str,
) -> ArtifactResult<usize> {
    let length = definite_len(length, field)?;
    if length > MAX_ITEMS {
        return Err(ArtifactError::Limit(format!("too many {field}")));
    }
    Ok(length)
}

fn require_len(
    length: Result<Option<u64>, minicbor::decode::Error>,
    expected: usize,
    field: &'static str,
) -> ArtifactResult<()> {
    let actual = definite_len(length, field)?;
    if actual != expected {
        return Err(ArtifactError::Invalid(format!(
            "{field} has {actual} fields, expected {expected}"
        )));
    }
    Ok(())
}

fn definite_len(
    length: Result<Option<u64>, minicbor::decode::Error>,
    field: &'static str,
) -> ArtifactResult<usize> {
    match length.map_err(decode_error)? {
        Some(length) => usize::try_from(length)
            .map_err(|_| ArtifactError::Limit(format!("{field} cannot fit in memory"))),
        None => Err(ArtifactError::Invalid(format!(
            "{field} uses indefinite-length CBOR"
        ))),
    }
}

fn encode_error(error: minicbor::encode::Error<std::convert::Infallible>) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}

fn decode_error(error: minicbor::decode::Error) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}
