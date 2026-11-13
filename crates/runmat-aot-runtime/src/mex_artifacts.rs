use std::rc::Rc;

use runmat_execution_artifact::PrivateArtifactRoot;
use runmat_mex::{MexArtifactBundle, MexArtifactManifest};

pub(crate) struct InstalledMexArtifacts {
    _root: PrivateArtifactRoot,
}

#[cfg(test)]
pub(crate) fn install(
    bytes: &[u8],
    session: &Rc<runmat_runtime::foreign::MexRuntimeSession>,
) -> Result<InstalledMexArtifacts, String> {
    let bundle = MexArtifactBundle::from_canonical_bytes(bytes)
        .map_err(|error| format!("standalone MEX artifact bundle is invalid: {error}"))?;
    install_admitted(&bundle, session)
}

pub(crate) fn install_admitted(
    bundle: &MexArtifactBundle,
    session: &Rc<runmat_runtime::foreign::MexRuntimeSession>,
) -> Result<InstalledMexArtifacts, String> {
    ensure_required_providers(bundle)?;
    let mut root = PrivateArtifactRoot::new("runmat-aot-mex-")?;
    for (index, entry) in bundle.artifacts.iter().enumerate() {
        let manifest = MexArtifactManifest::from_canonical_bytes(&entry.manifest)
            .map_err(|error| format!("decode standalone MEX artifact: {error}"))?;
        manifest
            .validate_current_module(&entry.module)
            .map_err(|error| format!("validate standalone MEX artifact: {error}"))?;
        let module_relative = format!(
            "{index}/{}.{}",
            manifest.module_name, manifest.target.suffix
        );
        let module_path = root.write_verified(&module_relative, &entry.module)?;
        // Object keys remain portable; the private root resolves native paths.
        let manifest_relative = format!("{module_relative}.runmat.json");
        root.write_verified(&manifest_relative, &entry.manifest)?;
        session
            .install_artifact(module_path.as_path())
            .map_err(|error| format!("install standalone MEX artifact: {error}"))?;
    }
    Ok(InstalledMexArtifacts { _root: root })
}

pub(crate) fn ensure_required_providers(bundle: &MexArtifactBundle) -> Result<(), String> {
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
        #[cfg(not(feature = "cuda"))]
        return Err(
            "standalone CUDA MEX artifact requires CUDA support in this runtime build".into(),
        );
        #[cfg(feature = "cuda")]
        {
            let provider = runmat_accelerate::backend::cuda::register_cuda_provider()
                .map_err(|error| format!("initialize standalone CUDA MEX provider: {error}"))?;
            if provider.is_none() {
                return Err(
                    "standalone CUDA MEX artifact requires an available CUDA driver and device"
                        .into(),
                );
            }
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
        let mut runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let local_tasks = Rc::new(runmat_runtime::context::RuntimeLocalTaskSupervisor::new());
        let ports = runtime
            .service_ports()
            .clone()
            .with_local_tasks(local_tasks.submission_port(runtime.local_identity()))
            .with_call(Rc::new(runmat_runtime::context::RuntimeCallRouter));
        runtime = runtime.with_service_ports(ports);
        let lifecycle_host =
            runmat_runtime::context::RuntimeLifecycleHost::new(&runtime, local_tasks.clone())
                .unwrap();
        let mut guard =
            crate::mex::install(&runtime, session.clone(), local_tasks.clone()).unwrap();
        local_tasks
            .drive_until(runtime.clone().scope(async {
                let arguments = runmat_runtime::call::RootedFunctionArguments::ordinary(
                    runmat_runtime::execution::RootedValues::in_context(Vec::new(), &runtime)
                        .expect("root embedded MEX arguments in their original runtime"),
                );
                let value = session
                    .load_and_call("embedded_fixture", arguments, 1, runtime.clone())
                    .await
                    .break_value()
                    .expect("installed artifact resolves")
                    .expect("embedded artifact executes");
                value.with_owned_sequence(|value| {
                    assert_eq!(value.as_slice(), &[runmat_value::Value::Num(42.0)]);
                });
                match guard.shutdown_gracefully_retirement().await {
                    runmat_runtime::foreign::MexSessionRetirement::Retired(result) => {
                        result.expect("embedded module retires before private storage")
                    }
                    runmat_runtime::foreign::MexSessionRetirement::Retained(error) => {
                        panic!("embedded module retained after shutdown: {error}")
                    }
                }
            }))
            .await
            .expect("scoped MEX lifecycle completes");
        lifecycle_host.force_close();
        drop(installed);
    }
}
