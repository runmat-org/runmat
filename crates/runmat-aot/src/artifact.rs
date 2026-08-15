use runmat_execution_artifact::{
    ExecutableForm, NativeObjectPayload, NativeTargetIdentity, ProgramArtifact, ProgramBuildRecipe,
    ProgramTarget,
};
use runmat_native_codegen::aot::RelocatableNativeObject;

use crate::{AotError, AotResult};

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
    let abi = &target.abi;
    recipe.target = ProgramTarget::native(
        "native-object-v1",
        NativeTargetIdentity {
            architecture: target.architecture.clone(),
            operating_system: target.operating_system.clone(),
            pointer_width: target.pointer_width,
            abi: format!(
                "{}:{}:{}:{}",
                abi.schema_version,
                abi.encoded_version,
                abi.contract_fingerprint,
                abi.layout_fingerprint
            ),
            object_format: object.manifest.object_format.token().into(),
        },
    );
    let metadata = serde_json::to_vec(&object.manifest)
        .map_err(|error| AotError::contract("aot.artifact.metadata", error.to_string()))?;
    let payload = NativeObjectPayload::new(
        object.manifest.object_format.token(),
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
    Ok((recipe, artifact))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};
    use runmat_execution_artifact::{
        ProgramBuildRecipe, ProgramTargetCohort, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    };
    use runmat_native_codegen::aot::{
        NativeObjectFormat, NativeObjectFunction, NativeObjectManifest, NativeOptimization,
        RelocatableNativeObject, AOT_ENTRY_SYMBOL, NATIVE_OBJECT_SCHEMA_VERSION,
    };
    use runmat_native_codegen::NativeTarget;
    use runmat_types::ProgramFunctionId;

    use super::materialize_native_object_artifact;

    #[test]
    fn native_artifact_target_is_derived_from_the_verified_object() {
        let bytes = b"native-object".to_vec();
        let target = NativeTarget::current();
        let runtime = Digest::sha256(b"runtime");
        let catalog = Digest::sha256(b"catalog");
        let function = ProgramFunctionId(7);
        let object = RelocatableNativeObject {
            manifest: NativeObjectManifest {
                schema_version: NATIVE_OBJECT_SCHEMA_VERSION,
                object_format: NativeObjectFormat::for_target(&target).unwrap(),
                target,
                executable_cache_key: Digest::sha256(b"executable"),
                native_cache_key: Digest::sha256(b"native"),
                runtime_fingerprint: runtime,
                catalog_fingerprint: catalog,
                optimization: NativeOptimization::Speed,
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
            ProgramEnvironment::new(1, 1, runtime, catalog, "matlab").unwrap(),
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
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };

        let (recipe, artifact) = materialize_native_object_artifact(recipe, &object).unwrap();
        assert_eq!(recipe.target.cohort, ProgramTargetCohort::Native);
        artifact.validate_against(&recipe).unwrap();
        let payload = artifact.native_object().unwrap().unwrap();
        assert_eq!(
            payload.object_format,
            NativeObjectFormat::for_target(&NativeTarget::current())
                .unwrap()
                .token()
        );
    }
}
