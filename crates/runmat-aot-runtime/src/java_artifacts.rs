use std::io::Write as _;
use std::path::PathBuf;
use std::rc::Rc;

use runmat_java::JavaArtifactBundle;

pub(crate) struct InstalledJavaArtifacts {
    _root: tempfile::TempDir,
}

pub(crate) fn install(
    bytes: &[u8],
    adapter: &Rc<runmat_runtime::foreign::JavaAdapter>,
) -> Result<InstalledJavaArtifacts, String> {
    let bundle = JavaArtifactBundle::from_canonical_bytes(bytes)
        .map_err(|error| format!("standalone Java artifact bundle is invalid: {error}"))?;
    let root = tempfile::Builder::new()
        .prefix("runmat-aot-java-")
        .tempdir()
        .map_err(|error| format!("create standalone Java artifact root: {error}"))?;
    let mut installed = Vec::with_capacity(bundle.artifacts.len());
    for (index, artifact) in bundle.artifacts.iter().enumerate() {
        artifact
            .identity
            .validate_bytes(&artifact.bytes)
            .map_err(|error| format!("validate standalone Java artifact: {error}"))?;
        let path = root.path().join(format!("{index}.jar"));
        write_private_read_only(&path, &artifact.bytes)?;
        installed.push((artifact.identity.clone(), path));
    }
    adapter
        .install_project_artifacts(&installed)
        .map_err(|error| format!("install standalone Java artifacts: {error}"))?;
    Ok(InstalledJavaArtifacts { _root: root })
}

fn write_private_read_only(path: &PathBuf, bytes: &[u8]) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create Java artifact {}: {error}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("write Java artifact {}: {error}", path.display()))?;
    let mut permissions = file
        .metadata()
        .map_err(|error| format!("inspect Java artifact {}: {error}", path.display()))?
        .permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(path, permissions)
        .map_err(|error| format!("seal Java artifact {}: {error}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_java::{JavaArtifactBundleEntry, JavaArtifactIdentity};

    #[test]
    fn installs_exact_aot_artifacts_into_java_adapter() {
        let bytes = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0".to_vec();
        let identity = JavaArtifactIdentity::for_bytes(&bytes);
        let bundle = JavaArtifactBundle::new(vec![JavaArtifactBundleEntry {
            logical_name: "fixture".into(),
            identity: identity.clone(),
            bytes,
        }])
        .unwrap();
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
