use std::rc::Rc;

use runmat_execution_artifact::PrivateArtifactRoot;
use runmat_java::JavaArtifactBundle;

pub(crate) struct InstalledJavaArtifacts {
    _root: PrivateArtifactRoot,
}

#[cfg(test)]
pub(crate) fn install(
    bytes: &[u8],
    adapter: &Rc<runmat_runtime::foreign::JavaAdapter>,
) -> Result<InstalledJavaArtifacts, String> {
    let bundle = JavaArtifactBundle::from_canonical_bytes(bytes)
        .map_err(|error| format!("standalone Java artifact bundle is invalid: {error}"))?;
    install_admitted(&bundle, adapter)
}

pub(crate) fn install_admitted(
    bundle: &JavaArtifactBundle,
    adapter: &Rc<runmat_runtime::foreign::JavaAdapter>,
) -> Result<InstalledJavaArtifacts, String> {
    let mut root = PrivateArtifactRoot::new("runmat-aot-java-")?;
    adapter
        .attach_selected_byte_lease(root.lease())
        .map_err(|error| format!("attach standalone Java byte owner: {error}"))?;
    let mut installed = Vec::with_capacity(bundle.artifacts.len());
    for (index, artifact) in bundle.artifacts.iter().enumerate() {
        artifact
            .identity
            .validate_bytes(&artifact.bytes)
            .map_err(|error| format!("validate standalone Java artifact: {error}"))?;
        let path = root.write_verified(&format!("{index}.jar"), &artifact.bytes)?;
        installed.push((artifact.identity.clone(), path.as_path().to_path_buf()));
    }
    adapter
        .install_project_artifacts(&installed)
        .map_err(|error| format!("install standalone Java artifacts: {error}"))?;
    Ok(InstalledJavaArtifacts { _root: root })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_java::{JavaArtifactBundleEntry, JavaArtifactIdentity};
    use runmat_types::{
        CapabilityRequirement, CapabilitySet, ExecutionStackRequirement, ForeignAdapterId,
        ForeignAdapterRequirement, ForeignArtifactIdentity, InteropManifest,
        INTEROP_MANIFEST_SCHEMA_VERSION,
    };
    use std::collections::BTreeSet;

    #[test]
    fn installs_exact_aot_artifacts_into_java_adapter() {
        unsafe extern "C" fn unused_resolver(_: u32) -> *const std::ffi::c_void {
            std::ptr::null()
        }

        let bytes = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0".to_vec();
        let identity = JavaArtifactIdentity::for_bytes(&bytes);
        let bundle = JavaArtifactBundle::new(vec![JavaArtifactBundleEntry {
            logical_name: "fixture".into(),
            identity: identity.clone(),
            bytes: bytes.clone(),
        }])
        .unwrap();
        let native_id = ForeignAdapterId::new(runmat_native_ffi::NATIVE_FFI_ADAPTER_ID).unwrap();
        let mex_id = ForeignAdapterId::new(runmat_mex::MEX_ADAPTER_ID).unwrap();
        let java_id = ForeignAdapterId::new(runmat_java::JAVA_ADAPTER_ID).unwrap();
        let python_id = ForeignAdapterId::new(runmat_python::PYTHON_ADAPTER_ID).unwrap();
        let manifest = InteropManifest {
            schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: vec![ForeignAdapterRequirement {
                adapter: java_id.clone(),
                minimum_version: runmat_java::JAVA_ADAPTER_VERSION,
                capabilities: CapabilitySet(BTreeSet::from([
                    CapabilityRequirement::ForeignRuntime,
                ])),
                execution_stack: ExecutionStackRequirement::Process,
                artifact_identities: vec![ForeignArtifactIdentity::new(identity.as_str()).unwrap()],
            }],
            adapter_contracts: Vec::new(),
        };
        let policy = runmat_execution_artifact::capture_typed_foreign_policy(
            &runmat_config::runtime::ForeignConfig::default(),
        )
        .unwrap();
        let mut input = crate::AotProcessInput {
            function_resolver: unused_resolver,
            builtin_resolver: unused_resolver,
            native_ir: Vec::new(),
            program: Vec::new(),
            resume_points: Vec::new(),
            native_interfaces: runmat_native_ffi::NativeInterfaceArtifactBundle::empty()
                .canonical_bytes()
                .unwrap(),
            mex_artifacts: runmat_mex::MexArtifactBundle::empty()
                .canonical_bytes()
                .unwrap(),
            java_artifacts: bundle.canonical_bytes().unwrap(),
            python_artifacts: runmat_python::PythonArtifactBundle::empty()
                .canonical_bytes()
                .unwrap(),
        };
        crate::foreign_policy::AdmittedForeignPolicy::admit(
            Some(&policy),
            &manifest,
            &input,
            &native_id,
            &mex_id,
            &java_id,
            &python_id,
        )
        .expect("the exact archive satisfies its Java grant");

        let mut extra_bytes = bytes;
        extra_bytes.push(1);
        let extra_bundle = JavaArtifactBundle::new(vec![
            bundle.artifacts[0].clone(),
            JavaArtifactBundleEntry {
                logical_name: "ungranted".into(),
                identity: JavaArtifactIdentity::for_bytes(&extra_bytes),
                bytes: extra_bytes,
            },
        ])
        .unwrap();
        input.java_artifacts = extra_bundle.canonical_bytes().unwrap();
        let extra_error = crate::foreign_policy::AdmittedForeignPolicy::admit(
            Some(&policy),
            &manifest,
            &input,
            &native_id,
            &mex_id,
            &java_id,
            &python_id,
        )
        .err()
        .expect("an extra same-adapter archive must be rejected");
        assert!(extra_error.contains("java archived artifacts differ from its exact grant"));

        input.java_artifacts = bundle.canonical_bytes().unwrap();
        let environment = runmat_python::PythonEnvironmentIdentity {
            implementation: "cpython".into(),
            version: runmat_python::PythonVersion {
                major: 3,
                minor: 12,
                patch: 0,
            },
            abi_tag: "cp312".into(),
            platform_tag: "test".into(),
            execution_mode: runmat_python::PythonExecutionMode::InProcess,
        };
        let mut python_required = manifest;
        python_required.adapters.push(ForeignAdapterRequirement {
            adapter: python_id.clone(),
            minimum_version: runmat_python::PYTHON_ADAPTER_VERSION,
            capabilities: CapabilitySet(BTreeSet::from([CapabilityRequirement::ForeignRuntime])),
            execution_stack: ExecutionStackRequirement::Process,
            artifact_identities: vec![ForeignArtifactIdentity::new(
                environment.artifact_identity(),
            )
            .unwrap()],
        });
        let missing_error = crate::foreign_policy::AdmittedForeignPolicy::admit(
            Some(&policy),
            &python_required,
            &input,
            &native_id,
            &mex_id,
            &java_id,
            &python_id,
        )
        .err()
        .expect("a required archived Python environment must be present");
        assert!(missing_error.contains("python archived artifacts differ from its exact grant"));

        let foreign = runmat_runtime::foreign::ForeignRuntime::new(
            runmat_runtime::foreign::ForeignPlatform::Native,
        );
        let adapter = runmat_runtime::foreign::JavaAdapter::new(foreign.handles().clone()).unwrap();
        let installed = install(&bundle.canonical_bytes().unwrap(), &adapter).unwrap();
        let descriptor = runmat_runtime::foreign::ForeignAdapter::descriptor(adapter.as_ref());
        assert!(descriptor.artifact_identities.contains(identity.as_str()));
        drop(installed);
    }
}
