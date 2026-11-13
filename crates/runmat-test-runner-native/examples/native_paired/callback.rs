use std::path::Path;

use anyhow::{ensure, Context, Result};
use runmat_execution::Digest;
use runmat_native_ffi::{
    CallingConvention, NativeInterfaceArtifactBundle, NativeInterfaceArtifactManifest,
    NativeScalar, NativeType, ParameterDirection, PointerOwnership,
};
use runmat_types::InteropManifest;

pub struct CallbackArtifact {
    pub bundle: NativeInterfaceArtifactBundle,
    pub interop: InteropManifest,
    pub policy: runmat_execution_artifact::LogicalObject,
}

pub fn load(
    bundle_path: &Path,
    expected_bundle_digest: &str,
    policy_path: &Path,
    expected_policy_digest: &str,
    interface_name: &str,
    artifact_identity: &str,
    library_digest: &str,
) -> Result<CallbackArtifact> {
    let bytes = runmat_aot::selected_admission::read_bounded(
        bundle_path,
        runmat_execution_artifact::target_pack::MAX_COMPONENT_BYTES,
    )
    .with_context(|| {
        format!(
            "read bounded native callback bundle `{}`",
            bundle_path.display()
        )
    })?;
    ensure!(
        Digest::sha256(&bytes).to_string() == expected_bundle_digest,
        "native callback bundle differs from the independent operator digest"
    );
    let bundle = NativeInterfaceArtifactBundle::from_canonical_bytes(&bytes)
        .context("decode canonical native callback bundle")?;
    let manifests = bundle
        .manifests()
        .context("decode native callback manifests")?;
    let [manifest] = manifests.as_slice() else {
        anyhow::bail!("native callback witness requires exactly one prepared interface");
    };
    validate_manifest(manifest, interface_name, artifact_identity, library_digest)?;
    let policy_bytes = runmat_aot::selected_admission::read_bounded(
        policy_path,
        runmat_execution_artifact::MAX_FOREIGN_POLICY_BYTES,
    )
    .with_context(|| {
        format!(
            "read bounded native callback policy `{}`",
            policy_path.display()
        )
    })?;
    ensure!(
        Digest::sha256(&policy_bytes).to_string() == expected_policy_digest,
        "native callback policy differs from the independent operator digest"
    );
    let policy = runmat_execution_artifact::foreign_policy_object(policy_bytes)
        .context("admit canonical native callback policy")?;
    let config = runmat_core::foreign_artifacts::decode_admitted_policy(&policy)
        .map_err(anyhow::Error::msg)
        .context("decode typed native callback policy")?;
    ensure!(
        config.native.isolation == runmat_config::runtime::NativeFfiIsolation::Process,
        "hosted-callback qualification requires the isolated native callback route"
    );
    Ok(CallbackArtifact {
        interop: manifest.interop_manifest(),
        bundle,
        policy,
    })
}

fn validate_manifest(
    manifest: &NativeInterfaceArtifactManifest,
    interface_name: &str,
    artifact_identity: &str,
    library_digest: &str,
) -> Result<()> {
    ensure!(
        manifest.interface_name == interface_name
            && manifest.identity.as_str() == artifact_identity
            && manifest.library_digest == library_digest,
        "native callback interface differs from its independent identity declaration"
    );
    let library = manifest
        .metadata
        .libraries
        .first()
        .context("native callback interface has no library")?;
    let symbol = library
        .symbols
        .iter()
        .find(|symbol| symbol.name == "fixture_apply")
        .context("native callback interface lacks fixture_apply")?;
    ensure!(
        symbol.exported_name == "fixture_apply"
            && symbol.calling_convention == CallingConvention::C
            && symbol.return_type
                == NativeType::Scalar {
                    scalar: NativeScalar::I32
                }
            && !symbol.return_nullable
            && symbol.return_ownership.is_none()
            && !symbol.variadic
            && symbol.parameters.len() == 2,
        "fixture_apply has an incompatible native signature"
    );
    let value = &symbol.parameters[0];
    let callback = &symbol.parameters[1];
    ensure!(
        value.ty
            == NativeType::Scalar {
                scalar: NativeScalar::I32
            }
            && value.direction == ParameterDirection::Input
            && value.ownership == PointerOwnership::Borrowed
            && !value.nullable,
        "fixture_apply value parameter is incompatible"
    );
    let NativeType::Callback {
        calling_convention,
        return_type,
        parameters,
    } = &callback.ty
    else {
        anyhow::bail!("fixture_apply callback parameter is not callable");
    };
    ensure!(
        callback.direction == ParameterDirection::Input
            && callback.ownership == PointerOwnership::Borrowed
            && !callback.nullable
            && *calling_convention == CallingConvention::C
            && **return_type
                == NativeType::Scalar {
                    scalar: NativeScalar::I32
                }
            && parameters.len() == 1
            && parameters[0].ty
                == NativeType::Scalar {
                    scalar: NativeScalar::I32
                }
            && parameters[0].direction == ParameterDirection::Input
            && parameters[0].ownership == PointerOwnership::Borrowed
            && !parameters[0].nullable,
        "fixture_apply callback ABI is incompatible"
    );
    Ok(())
}
