use std::collections::BTreeSet;

use minicbor::Encoder;
use runmat_execution::resource::Capability;
use runmat_execution::Digest;

use super::{BundleManifest, ExecutionBundle, EXECUTION_BUNDLE_SCHEMA_VERSION};
use crate::object::{validate_inventory, ObjectInventoryLimits};
use crate::{ArtifactError, ArtifactResult, BundleCodeClosure, ObjectNamespace};

pub(super) fn validate(bundle: &ExecutionBundle) -> ArtifactResult<()> {
    if bundle.manifest.schema_version != EXECUTION_BUNDLE_SCHEMA_VERSION {
        return Err(ArtifactError::Invalid(
            "unsupported execution bundle schema".into(),
        ));
    }
    if bundle.manifest.recipes.is_empty() {
        return Err(ArtifactError::Invalid(
            "execution bundle must contain at least one build recipe".into(),
        ));
    }
    bundle
        .manifest
        .program_revision
        .validate()
        .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
    let exact_project_sources = bundle.manifest.program_revision.source_digest()
        == &bundle.manifest.project_revision.source_digest;
    let exact_test_overlay = bundle
        .manifest
        .program_revision
        .domain_contribution("runmat.test.config")
        .is_some();
    if bundle.manifest.program_revision.graph_digest()
        != &bundle.manifest.project_revision.graph_digest
        || (!exact_project_sources && !exact_test_overlay)
        || bundle.manifest.resources.cpu_millicores == 0
        || bundle.manifest.resources.memory_bytes == 0
    {
        return Err(ArtifactError::Identity(
            "bundle revisions or resources are inconsistent".into(),
        ));
    }
    validate_code_closure(bundle)?;
    validate_inventory(&bundle.objects, ObjectInventoryLimits::default())?;
    let descriptors = bundle
        .objects
        .iter()
        .map(|object| object.descriptor.clone())
        .collect::<Vec<_>>();
    if descriptors != bundle.manifest.sources
        || bundle
            .manifest
            .sources
            .iter()
            .any(|source| source.namespace != ObjectNamespace::ProgramSource)
        || bundle
            .manifest
            .sources
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || bundle
            .manifest
            .callables
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || bundle
            .manifest
            .portable_environment
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
    {
        return Err(ArtifactError::Invalid(
            "bundle inventories are not canonical".into(),
        ));
    }
    let recipes = bundle
        .manifest
        .recipes
        .iter()
        .map(|recipe| {
            if recipe.program_revision != bundle.manifest.program_revision
                || recipe.source_objects != bundle.manifest.sources
            {
                return Err(ArtifactError::Identity(
                    "bundle recipe does not name its exact bundle revision and source closure"
                        .into(),
                ));
            }
            recipe.id()
        })
        .collect::<ArtifactResult<Vec<_>>>()?;
    if recipes.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(ArtifactError::Invalid(
            "bundle recipes are not unique and sorted".into(),
        ));
    }
    if bundle
        .manifest
        .artifacts
        .windows(2)
        .any(|pair| pair[0].id >= pair[1].id)
    {
        return Err(ArtifactError::Invalid(
            "bundle program artifacts are not unique and sorted".into(),
        ));
    }
    let recipe_set = recipes.into_iter().collect::<BTreeSet<_>>();
    for artifact in &bundle.manifest.artifacts {
        let recipe = bundle
            .manifest
            .recipes
            .iter()
            .find(|recipe| recipe.id().ok() == Some(artifact.recipe_id))
            .ok_or_else(|| {
                ArtifactError::Invalid("program artifact has no bundle recipe".into())
            })?;
        artifact.validate_against(recipe)?;
        if !recipe_set.contains(&artifact.recipe_id) {
            return Err(ArtifactError::Invalid(
                "program artifact recipe is absent".into(),
            ));
        }
    }
    let source_digests = bundle
        .manifest
        .sources
        .iter()
        .map(|source| source.digest)
        .collect::<BTreeSet<_>>();
    if bundle
        .manifest
        .callables
        .iter()
        .any(|callable| !source_digests.contains(&callable.source_digest))
    {
        return Err(ArtifactError::Identity(
            "callable inventory references a source outside the bundle".into(),
        ));
    }
    for (name, value) in &bundle.manifest.portable_environment {
        let upper = name.to_ascii_uppercase();
        if name.is_empty()
            || name.len() > 128
            || value.len() > 4096
            || !name.is_ascii()
            || !value.is_ascii()
            || name.chars().any(char::is_control)
            || value.chars().any(char::is_control)
            || ["TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH"]
                .iter()
                .any(|marker| upper.contains(marker))
        {
            return Err(ArtifactError::Invalid(
                "portable environment contains an invalid or secret-bearing entry".into(),
            ));
        }
    }
    Ok(())
}

fn validate_code_closure(bundle: &ExecutionBundle) -> ArtifactResult<()> {
    match &bundle.manifest.code_closure {
        BundleCodeClosure::SourceProject { handoff } => validate_project_handoff(bundle, handoff),
        BundleCodeClosure::Compiled { package } => {
            package.validate()?;
            if package.graph_digest != bundle.manifest.project_revision.graph_digest
                || package.source_digest != bundle.manifest.project_revision.source_digest
                || !bundle.manifest.sources.is_empty()
                || !bundle.manifest.callables.is_empty()
                || !bundle.objects.is_empty()
                || bundle.manifest.artifacts.is_empty()
                || bundle.manifest.artifacts.iter().any(|artifact| {
                    !matches!(
                        artifact.form,
                        crate::ExecutableForm::ExecutableUnitV3
                            | crate::ExecutableForm::NativeObjectV1
                    )
                })
            {
                return Err(ArtifactError::Identity(
                    "compiled package closure differs from its bundle revision, contains source payloads, or has a non-compiled artifact".into(),
                ));
            }
            Ok(())
        }
    }
}

fn validate_project_handoff(
    bundle: &ExecutionBundle,
    handoff: &runmat_package::FrozenProjectHandoff,
) -> ArtifactResult<()> {
    handoff
        .validate()
        .map_err(|error| ArtifactError::Invalid(format!("invalid project handoff: {error}")))?;
    let revision = handoff.revision();
    if revision.graph_digest.bytes() != bundle.manifest.project_revision.graph_digest.bytes()
        || revision.source_revision.bytes()
            != bundle.manifest.project_revision.source_digest.bytes()
        || handoff.project.workspace_root != std::path::Path::new(".")
        || handoff.project.manifest_path != std::path::Path::new("runmat.toml")
    {
        return Err(ArtifactError::Identity(
            "bundle project handoff does not match its portable revision or roots".into(),
        ));
    }

    let mut expected_sources = Vec::new();
    let mut expected_callables = Vec::new();
    for package in handoff.project.sources.packages.values() {
        for source in &package.sources {
            let logical_name =
                format!("{}/{}", package.mount.logical_root, source.id.relative_path);
            let access_path = handoff
                .project
                .access_paths
                .get(&source.id)
                .ok_or_else(|| {
                    ArtifactError::Invalid(format!(
                        "bundle source {} has no portable access path",
                        source.id.relative_path
                    ))
                })?;
            if access_path != std::path::Path::new(&logical_name) {
                return Err(ArtifactError::Invalid(format!(
                    "bundle source {} is bound to a physical or inconsistent path",
                    source.id.relative_path
                )));
            }
            let object = bundle
                .objects
                .iter()
                .find(|object| object.descriptor.logical_name == logical_name)
                .ok_or_else(|| {
                    ArtifactError::Invalid(format!(
                        "bundle source {} has no logical object",
                        source.id.relative_path
                    ))
                })?;
            if object.descriptor.namespace != ObjectNamespace::ProgramSource
                || object.descriptor.digest.bytes() != source.id.content_digest.bytes()
            {
                return Err(ArtifactError::Identity(format!(
                    "bundle source {} differs from the frozen source identity",
                    source.id.relative_path
                )));
            }
            expected_sources.push(object.descriptor.clone());
            expected_callables.push(crate::BundleCallable {
                owner_identity: package.package_instance.to_string(),
                qualified_name: source.qualified_name.clone(),
                source_digest: object.descriptor.digest,
            });
        }
    }
    expected_sources.sort();
    expected_callables.sort();
    if expected_sources != bundle.manifest.sources
        || expected_callables != bundle.manifest.callables
    {
        return Err(ArtifactError::Identity(
            "bundle source/callable inventories differ from the frozen project".into(),
        ));
    }
    Ok(())
}

pub(super) fn identity(manifest: &BundleManifest) -> ArtifactResult<Digest> {
    let mut bytes = b"runmat-execution-bundle-v3\0".to_vec();
    let revision = manifest
        .program_revision
        .canonical_bytes()
        .map_err(|error| ArtifactError::Encoding(error.to_string()))?;
    let mut encoder = Encoder::new(&mut bytes);
    encoder
        .array(12)
        .and_then(|encoder| encoder.u16(manifest.schema_version))
        .and_then(|encoder| encoder.bytes(&revision))
        .and_then(|encoder| encoder.bytes(manifest.project_revision.graph_digest.bytes()))
        .and_then(|encoder| encoder.bytes(manifest.project_revision.source_digest.bytes()))
        .and_then(|encoder| {
            let closure = serde_json::to_vec(&manifest.code_closure)
                .map_err(|_| minicbor::encode::Error::message("invalid code closure"))?;
            encoder.bytes(&closure)
        })
        .and_then(|encoder| encoder.array(manifest.sources.len() as u64))
        .map_err(encoding)?;
    for source in &manifest.sources {
        encoder
            .array(5)
            .and_then(|encoder| encoder.u8(source.namespace as u8))
            .and_then(|encoder| encoder.str(&source.logical_name))
            .and_then(|encoder| encoder.bytes(source.digest.bytes()))
            .and_then(|encoder| encoder.u64(source.encoded_length))
            .and_then(|encoder| encoder.str(&source.media_type))
            .map_err(encoding)?;
    }
    encoder
        .array(manifest.callables.len() as u64)
        .map_err(encoding)?;
    for callable in &manifest.callables {
        encoder
            .array(3)
            .and_then(|encoder| encoder.str(&callable.owner_identity))
            .and_then(|encoder| encoder.str(&callable.qualified_name))
            .and_then(|encoder| encoder.bytes(callable.source_digest.bytes()))
            .map_err(encoding)?;
    }
    encoder
        .array(manifest.recipes.len() as u64)
        .map_err(encoding)?;
    for recipe in &manifest.recipes {
        encoder.bytes(recipe.id()?.0.bytes()).map_err(encoding)?;
    }
    encoder
        .array(manifest.artifacts.len() as u64)
        .map_err(encoding)?;
    for artifact in &manifest.artifacts {
        encoder.bytes(artifact.id.0.bytes()).map_err(encoding)?;
    }
    encoder
        .array(manifest.requested_capabilities.len() as u64)
        .map_err(encoding)?;
    for capability in &manifest.requested_capabilities {
        encode_capability(&mut encoder, capability)?;
    }
    encoder
        .array(3)
        .and_then(|encoder| encoder.u32(manifest.resources.cpu_millicores))
        .and_then(|encoder| encoder.u64(manifest.resources.memory_bytes))
        .and_then(|encoder| encoder.u64(manifest.resources.scratch_bytes))
        .map_err(encoding)?;
    encoder
        .array(manifest.portable_environment.len() as u64)
        .map_err(encoding)?;
    for (name, value) in &manifest.portable_environment {
        encoder
            .array(2)
            .and_then(|encoder| encoder.str(name))
            .and_then(|encoder| encoder.str(value))
            .map_err(encoding)?;
    }
    Ok(Digest::sha256(bytes))
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
    .map_err(encoding)
}

fn encoding(error: minicbor::encode::Error<std::convert::Infallible>) -> ArtifactError {
    ArtifactError::Encoding(error.to_string())
}
