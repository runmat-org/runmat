use runmat_execution::host::NativeObjectFormat as ExecutionObjectFormat;
use runmat_execution_artifact::{
    ExecutableForm, NativeObjectPayload, ProgramArtifact, ProgramBuildRecipe, ProgramTarget,
};
use runmat_native_codegen::aot::{
    NativeObjectFormat, NativeObjectManifest, RelocatableNativeObject, NATIVE_OBJECT_SCHEMA_VERSION,
};

use crate::{AotError, AotResult};

#[derive(serde::Deserialize)]
struct NativeObjectManifestAdmission {
    schema_version: u16,
}

pub fn materialize_native_object_artifact(
    mut recipe: ProgramBuildRecipe,
    object: &RelocatableNativeObject,
) -> AotResult<(ProgramBuildRecipe, ProgramArtifact)> {
    object
        .validate()
        .map_err(|error| AotError::contract("aot.artifact.object", error.to_string()))?;
    let environment = recipe.program_revision.environment();
    if object.manifest.runtime_fingerprint != environment.runtime_fingerprint
        || object.manifest.catalog_fingerprint != environment.catalog_fingerprint
        || recipe.entrypoint != object.manifest.entrypoint.0.to_string()
    {
        return Err(AotError::contract(
            "aot.artifact.identity",
            "native object does not match the recipe program environment or entrypoint",
        ));
    }
    let target = &object.manifest.target;
    recipe.target = ProgramTarget::native(
        "native-object-v1",
        target
            .execution_identity()
            .map_err(|error| AotError::contract("aot.artifact.target", error.to_string()))?,
    );
    let metadata = canonical_native_manifest(&object.manifest)?;
    let payload = NativeObjectPayload::new(
        execution_object_format(object.manifest.object_format.token())?,
        metadata,
        object.bytes.clone(),
    )
    .map_err(|error| AotError::contract("aot.artifact.payload", error.to_string()))?;
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::NativeObjectV1,
        payload
            .to_canonical_bytes()
            .map_err(|error| AotError::contract("aot.artifact.payload", error.to_string()))?,
    )
    .map_err(|error| AotError::contract("aot.artifact.program", error.to_string()))?;
    read_native_object_artifact(&recipe, &artifact)?;
    Ok((recipe, artifact))
}

/// Admit a native object only after both artifact envelopes and the nested
/// native manifest have been bound to the requested program and target.
pub fn read_native_object_artifact(
    recipe: &ProgramBuildRecipe,
    artifact: &ProgramArtifact,
) -> AotResult<RelocatableNativeObject> {
    artifact
        .validate_against(recipe)
        .map_err(|error| AotError::contract("aot.artifact.program", error.to_string()))?;
    recipe
        .program_revision
        .validate_current_compiler()
        .map_err(|error| {
            AotError::contract("aot.artifact.compiler_compatibility", error.to_string())
        })?;
    let payload = artifact
        .native_object()
        .map_err(|error| AotError::contract("aot.artifact.payload", error.to_string()))?
        .ok_or_else(|| {
            AotError::contract("aot.artifact.form", "artifact is not a native object")
        })?;
    let admission: NativeObjectManifestAdmission = serde_json::from_slice(&payload.metadata)
        .map_err(|error| AotError::contract("aot.artifact.metadata", error.to_string()))?;
    if admission.schema_version != NATIVE_OBJECT_SCHEMA_VERSION {
        return Err(revision_mismatch(
            "aot.artifact.native_schema",
            "native object schema",
            admission.schema_version,
            NATIVE_OBJECT_SCHEMA_VERSION,
        ));
    }
    let manifest: NativeObjectManifest = serde_json::from_slice(&payload.metadata)
        .map_err(|error| AotError::contract("aot.artifact.metadata", error.to_string()))?;
    if canonical_native_manifest(&manifest)? != payload.metadata {
        return Err(AotError::contract(
            "aot.artifact.metadata",
            "native object manifest encoding is not canonical",
        ));
    }
    let target = manifest
        .target
        .execution_identity()
        .map_err(|error| AotError::contract("aot.artifact.target", error.to_string()))?;
    let expected_target = recipe.target.native.as_ref().ok_or_else(|| {
        AotError::contract(
            "aot.artifact.target",
            "native recipe has no target identity",
        )
    })?;
    if &target != expected_target {
        return Err(AotError::contract(
            "aot.artifact.target",
            format!("native target actual {target:?}, expected {expected_target:?}"),
        ));
    }
    let expected_format = NativeObjectFormat::for_target(&manifest.target)
        .map_err(|error| AotError::contract("aot.artifact.format", error.to_string()))?;
    if manifest.object_format != expected_format {
        return Err(AotError::contract(
            "aot.artifact.format",
            format!(
                "native manifest object format actual {}, expected {}",
                manifest.object_format.token(),
                expected_format.token()
            ),
        ));
    }
    let expected_payload_format = execution_object_format(expected_format.token())?;
    if payload.object_format != expected_payload_format {
        return Err(AotError::contract(
            "aot.artifact.payload_format",
            format!(
                "native payload object format actual {}, expected {}",
                payload.object_format.token(),
                expected_payload_format.token()
            ),
        ));
    }
    let environment = recipe.program_revision.environment();
    if manifest.runtime_fingerprint != environment.runtime_fingerprint {
        return Err(AotError::contract(
            "aot.artifact.runtime_fingerprint",
            format!(
                "runtime fingerprint actual {}, expected {}",
                manifest.runtime_fingerprint, environment.runtime_fingerprint
            ),
        ));
    }
    if manifest.catalog_fingerprint != environment.catalog_fingerprint {
        return Err(AotError::contract(
            "aot.artifact.catalog_fingerprint",
            format!(
                "catalog fingerprint actual {}, expected {}",
                manifest.catalog_fingerprint, environment.catalog_fingerprint
            ),
        ));
    }
    if manifest.entrypoint.0.to_string() != recipe.entrypoint {
        return Err(AotError::contract(
            "aot.artifact.entrypoint",
            format!(
                "native entrypoint actual {}, expected {}",
                manifest.entrypoint.0, recipe.entrypoint
            ),
        ));
    }
    let expected_native_key = manifest
        .target
        .cache_key(&manifest.executable_cache_key)
        .map_err(|error| AotError::contract("aot.artifact.cache", error.to_string()))?;
    if manifest.native_cache_key != expected_native_key {
        return Err(AotError::contract(
            "aot.artifact.cache",
            format!(
                "native cache key actual {}, expected {}",
                manifest.native_cache_key, expected_native_key
            ),
        ));
    }
    let object = RelocatableNativeObject {
        manifest,
        bytes: payload.object.clone(),
    };
    object
        .validate()
        .map_err(|error| AotError::contract("aot.artifact.object", error.to_string()))?;
    Ok(object)
}

fn revision_mismatch(
    code: &'static str,
    field: &'static str,
    actual: u16,
    expected: u16,
) -> AotError {
    AotError::contract(
        code,
        format!("{field} actual {actual}, expected {expected}"),
    )
}

fn execution_object_format(value: &str) -> AotResult<ExecutionObjectFormat> {
    ExecutionObjectFormat::from_token(value)
        .map_err(|error| AotError::contract("aot.artifact.target", error.to_string()))
}

fn canonical_native_manifest(manifest: &NativeObjectManifest) -> AotResult<Vec<u8>> {
    let value = serde_json::to_value(manifest)
        .map_err(|error| AotError::contract("aot.artifact.metadata", error.to_string()))?;
    serde_json::to_vec(&value)
        .map_err(|error| AotError::contract("aot.artifact.metadata", error.to_string()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};
    use runmat_execution_artifact::{
        ExecutableForm, NativeObjectPayload, ProgramArtifact, ProgramBuildRecipe,
        ProgramTargetCohort, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    };
    use runmat_native_codegen::aot::{
        AotRuntimeBindingMode, NativeObjectFormat, NativeObjectFunction, NativeObjectManifest,
        NativeOptimization, RelocatableNativeObject, AOT_ENTRY_SYMBOL,
        NATIVE_OBJECT_SCHEMA_VERSION,
    };
    use runmat_native_codegen::NativeTarget;
    use runmat_types::ProgramFunctionId;

    use super::{
        execution_object_format, materialize_native_object_artifact, read_native_object_artifact,
        ExecutionObjectFormat,
    };

    fn artifact_with_manifest(
        recipe: &ProgramBuildRecipe,
        manifest: &NativeObjectManifest,
        bytes: &[u8],
    ) -> ProgramArtifact {
        let payload = NativeObjectPayload::new(
            execution_object_format(manifest.object_format.token()).unwrap(),
            super::canonical_native_manifest(manifest).unwrap(),
            bytes.to_vec(),
        )
        .unwrap();
        ProgramArtifact::materialize(
            recipe,
            ExecutableForm::NativeObjectV1,
            payload.to_canonical_bytes().unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn native_artifact_target_is_derived_from_the_verified_object() {
        let bytes = b"native-object".to_vec();
        let target = NativeTarget::current();
        let runtime = Digest::sha256(b"runtime");
        let catalog = Digest::sha256(b"catalog");
        let function = ProgramFunctionId(7);
        let executable_cache_key = Digest::sha256(b"executable");
        let native_cache_key = target.cache_key(&executable_cache_key).unwrap();
        let object = RelocatableNativeObject {
            manifest: NativeObjectManifest {
                schema_version: NATIVE_OBJECT_SCHEMA_VERSION,
                object_format: NativeObjectFormat::for_target(&target).unwrap(),
                target,
                executable_cache_key,
                native_cache_key,
                runtime_fingerprint: runtime,
                catalog_fingerprint: catalog,
                optimization: NativeOptimization::Speed,
                runtime_binding_mode: AotRuntimeBindingMode::Dynamic,
                object_digest: Digest::sha256(&bytes),
                object_bytes: bytes.len() as u64,
                entrypoint: function,
                functions: vec![NativeObjectFunction {
                    function,
                    symbol: AOT_ENTRY_SYMBOL.into(),
                }],
                data: Vec::new(),
            },
            bytes,
        };
        let revision = ProgramRevision::new(
            Digest::sha256(b"graph"),
            Digest::sha256(b"source"),
            ProgramEnvironment::new(
                runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
                runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
                runtime,
                catalog,
                "matlab",
            )
            .unwrap(),
        )
        .unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: function.0.to_string(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "native".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("unbound"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };

        let (recipe, artifact) = materialize_native_object_artifact(recipe, &object).unwrap();
        assert_eq!(recipe.target.cohort, ProgramTargetCohort::Native);
        artifact.validate_against(&recipe).unwrap();
        assert_eq!(
            read_native_object_artifact(&recipe, &artifact).unwrap(),
            object
        );
        let payload = artifact.native_object().unwrap().unwrap();
        assert_eq!(
            payload.object_format,
            execution_object_format(
                NativeObjectFormat::for_target(&NativeTarget::current())
                    .unwrap()
                    .token(),
            )
            .unwrap()
        );

        let wrong_payload_format = match payload.object_format {
            ExecutionObjectFormat::MachO => ExecutionObjectFormat::Elf,
            ExecutionObjectFormat::Elf | ExecutionObjectFormat::Coff => {
                ExecutionObjectFormat::MachO
            }
        };
        let wrong_payload = NativeObjectPayload::new(
            wrong_payload_format,
            payload.metadata.clone(),
            payload.object.clone(),
        )
        .unwrap();
        let wrong_format_artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::NativeObjectV1,
            wrong_payload.to_canonical_bytes().unwrap(),
        )
        .unwrap();
        assert!(read_native_object_artifact(&recipe, &wrong_format_artifact)
            .unwrap_err()
            .to_string()
            .contains("aot.artifact.payload_format"));

        let mut wrong_manifest_format = object.manifest.clone();
        wrong_manifest_format.object_format = match wrong_manifest_format.object_format {
            NativeObjectFormat::MachO => NativeObjectFormat::Elf,
            NativeObjectFormat::Elf | NativeObjectFormat::Coff => NativeObjectFormat::MachO,
        };
        let wrong_manifest_format_artifact =
            artifact_with_manifest(&recipe, &wrong_manifest_format, &object.bytes);
        assert!(
            read_native_object_artifact(&recipe, &wrong_manifest_format_artifact)
                .unwrap_err()
                .to_string()
                .contains("aot.artifact.format")
        );

        let stale_schema: NativeObjectManifest = serde_json::from_slice(include_bytes!(
            "../tests/fixtures/native-object-manifest-4.json"
        ))
        .expect("frozen native-object manifest 4 remains structurally decodable");
        let stale_artifact = artifact_with_manifest(&recipe, &stale_schema, &object.bytes);
        assert!(read_native_object_artifact(&recipe, &stale_artifact)
            .unwrap_err()
            .to_string()
            .contains("aot.artifact.native_schema"));

        let changed_schema_metadata =
            br#"{"functions":"changed representation","schema_version":4}"#.to_vec();
        let changed_schema_payload = NativeObjectPayload::new(
            execution_object_format(object.manifest.object_format.token()).unwrap(),
            changed_schema_metadata,
            object.bytes.clone(),
        )
        .unwrap();
        let changed_schema_artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::NativeObjectV1,
            changed_schema_payload.to_canonical_bytes().unwrap(),
        )
        .unwrap();
        let changed_schema_error =
            read_native_object_artifact(&recipe, &changed_schema_artifact).unwrap_err();
        assert!(changed_schema_error
            .to_string()
            .contains("aot.artifact.native_schema"));
        assert!(changed_schema_error
            .to_string()
            .contains("actual 4, expected 5"));

        for (mutate, expected_code) in [
            (
                (|manifest: &mut NativeObjectManifest| {
                    manifest.runtime_fingerprint = Digest::sha256(b"other-runtime")
                }) as fn(&mut NativeObjectManifest),
                "aot.artifact.runtime_fingerprint",
            ),
            (
                |manifest: &mut NativeObjectManifest| {
                    manifest.catalog_fingerprint = Digest::sha256(b"other-catalog")
                },
                "aot.artifact.catalog_fingerprint",
            ),
            (
                |manifest: &mut NativeObjectManifest| {
                    manifest.native_cache_key = Digest::sha256(b"other-cache")
                },
                "aot.artifact.cache",
            ),
        ] {
            let mut mismatched = object.manifest.clone();
            mutate(&mut mismatched);
            let mismatched_artifact = artifact_with_manifest(&recipe, &mismatched, &object.bytes);
            assert!(read_native_object_artifact(&recipe, &mismatched_artifact)
                .unwrap_err()
                .to_string()
                .contains(expected_code));
        }

        let mut wrong_entrypoint = recipe.clone();
        wrong_entrypoint.entrypoint = "99".into();
        let wrong_entrypoint_artifact =
            artifact_with_manifest(&wrong_entrypoint, &object.manifest, &object.bytes);
        assert!(
            read_native_object_artifact(&wrong_entrypoint, &wrong_entrypoint_artifact)
                .unwrap_err()
                .to_string()
                .contains("aot.artifact.entrypoint")
        );

        let mut wrong_target = recipe.clone();
        wrong_target.target.native.as_mut().unwrap().pointer_width = 32;
        let wrong_target_artifact =
            artifact_with_manifest(&wrong_target, &object.manifest, &object.bytes);
        assert!(
            read_native_object_artifact(&wrong_target, &wrong_target_artifact)
                .unwrap_err()
                .to_string()
                .contains("aot.artifact.target")
        );
    }
}
