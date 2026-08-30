use std::rc::Rc;

use runmat_mex::{MexArtifactBundle, MexArtifactManifest};

pub(crate) struct InstalledMexArtifacts {
    _root: tempfile::TempDir,
}

pub(crate) fn install(
    bytes: &[u8],
    session: &Rc<runmat_runtime::foreign::MexRuntimeSession>,
) -> Result<InstalledMexArtifacts, String> {
    let bundle = MexArtifactBundle::from_canonical_bytes(bytes)
        .map_err(|error| format!("standalone MEX artifact bundle is invalid: {error}"))?;
    ensure_required_providers(&bundle)?;
    let root = tempfile::Builder::new()
        .prefix("runmat-aot-mex-")
        .tempdir()
        .map_err(|error| format!("create standalone MEX artifact root: {error}"))?;
    for (index, entry) in bundle.artifacts.iter().enumerate() {
        let manifest = MexArtifactManifest::from_canonical_bytes(&entry.manifest)
            .map_err(|error| format!("decode standalone MEX artifact: {error}"))?;
        manifest
            .validate_current_module(&entry.module)
            .map_err(|error| format!("validate standalone MEX artifact: {error}"))?;
        let directory = root.path().join(index.to_string());
        std::fs::create_dir(&directory)
            .map_err(|error| format!("create standalone MEX directory: {error}"))?;
        let module_path = directory.join(format!(
            "{}.{}",
            manifest.module_name, manifest.target.suffix
        ));
        let manifest_path = MexArtifactManifest::path_for_module(&module_path);
        crate::materialize::write_private_read_only(&module_path, &entry.module)?;
        crate::materialize::write_private_read_only(&manifest_path, &entry.manifest)?;
        session
            .install_artifact(&module_path)
            .map_err(|error| format!("install standalone MEX artifact: {error}"))?;
    }
    Ok(InstalledMexArtifacts { _root: root })
}

fn ensure_required_providers(bundle: &MexArtifactBundle) -> Result<(), String> {
    let requires_cuda = bundle.artifacts.iter().try_fold(false, |required, entry| {
        MexArtifactManifest::from_canonical_bytes(&entry.manifest)
            .map(|manifest| {
                required || manifest.source_language == runmat_mex::MexSourceLanguage::Cuda
            })
            .map_err(|error| format!("decode standalone MEX artifact: {error}"))
    })?;
    if requires_cuda
        && runmat_accelerate_api::provider_for_native_device(
            runmat_accelerate_api::NativeDeviceApi::Cuda,
        )
        .is_none()
    {
        let provider = runmat_accelerate::backend::cuda::register_cuda_provider()
            .map_err(|error| format!("initialize standalone CUDA MEX provider: {error}"))?;
        if provider.is_none() {
            return Err(
                "standalone CUDA MEX artifact requires an available CUDA driver and device".into(),
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_mex::{MexArtifactBundleEntry, MexBuild};
    use runmat_runtime::foreign::ForeignAdapter as _;

    #[tokio::test(flavor = "current_thread")]
    async fn installs_and_executes_exact_embedded_module_from_private_storage() {
        let source_root = tempfile::tempdir().unwrap();
        let source = source_root.path().join("embedded_fixture.c");
        std::fs::write(
            &source,
            r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs > 0) plhs[0] = mxCreateDoubleScalar(42.0);
}
"#,
        )
        .unwrap();
        let artifact = MexBuild::new(&source, source_root.path())
            .compile()
            .unwrap();
        let manifest = std::fs::read(&artifact.manifest).unwrap();
        let module = std::fs::read(&artifact.module).unwrap();
        let identity = MexArtifactManifest::from_canonical_bytes(&manifest)
            .unwrap()
            .identity
            .to_string();
        let bundle = MexArtifactBundle::new(vec![MexArtifactBundleEntry { manifest, module }])
            .unwrap()
            .canonical_bytes()
            .unwrap();
        let session = Rc::new(runmat_runtime::foreign::MexRuntimeSession::new());
        let installed = install(&bundle, &session).unwrap();
        assert!(session
            .descriptor()
            .artifact_identities
            .contains(identity.as_str()));
        std::fs::remove_file(&artifact.manifest).unwrap();
        std::fs::remove_file(&artifact.module).unwrap();
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let value = session
            .load_and_call("embedded_fixture", Vec::new(), 1, runtime)
            .await
            .expect("installed artifact resolves")
            .expect("embedded artifact executes");
        assert_eq!(value, runmat_value::Value::Num(42.0));
        drop(installed);
    }
}
