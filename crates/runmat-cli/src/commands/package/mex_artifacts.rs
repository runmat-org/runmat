use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Context, Result};
use runmat_execution_artifact::{LogicalObject, ObjectNamespace};
use runmat_mex::{
    MexArtifactBundle, MexArtifactBundleEntry, MexArtifactManifest,
    MEX_ARTIFACT_MANIFEST_MEDIA_TYPE, MEX_MODULE_MEDIA_TYPE,
};
use runmat_package::{ContentDigest, FrozenProject};
use runmat_types::InteropManifest;

pub(super) struct PreparedMexArtifacts {
    pub(super) interop: InteropManifest,
    pub(super) objects: Vec<LogicalObject>,
    pub(super) bundle: MexArtifactBundle,
}

pub(super) fn prepare(project: &FrozenProject) -> Result<PreparedMexArtifacts> {
    let mut names = BTreeMap::new();
    let mut identities = BTreeSet::new();
    let mut entries = Vec::with_capacity(project.mex_artifacts.len());
    let mut objects = Vec::with_capacity(project.mex_artifacts.len() * 2);
    let mut manifests = Vec::with_capacity(project.mex_artifacts.len());

    for declaration in &project.mex_artifacts {
        if let Some(owner) = names.insert(
            declaration.name.clone(),
            declaration.package_instance.clone(),
        ) {
            bail!(
                "MEX module `{}` is declared by package instances {} and {}",
                declaration.name,
                owner,
                declaration.package_instance
            );
        }
        let manifest_bytes = read_exact(
            &declaration.manifest_path,
            &declaration.manifest_digest,
            "manifest",
            &declaration.name,
        )?;
        let module_bytes = read_exact(
            &declaration.module_path,
            &declaration.module_digest,
            "module",
            &declaration.name,
        )?;
        let manifest =
            MexArtifactManifest::from_canonical_bytes(&manifest_bytes).with_context(|| {
                format!(
                    "MEX artifact `{}` has an invalid manifest",
                    declaration.name
                )
            })?;
        if manifest.module_name != declaration.name {
            bail!(
                "MEX artifact declaration `{}` names module `{}`",
                declaration.name,
                manifest.module_name
            );
        }
        manifest
            .validate_current_module(&module_bytes)
            .with_context(|| {
                format!(
                    "MEX artifact `{}` does not match this host",
                    declaration.name
                )
            })?;
        let identity = manifest.identity.to_string();
        if !identities.insert(identity.clone()) {
            bail!("MEX artifact identity `{identity}` is declared more than once");
        }

        manifests.push(manifest.interop_manifest());

        let logical_root = format!("mex/{}", logical_identity(&identity));
        let module_file = format!("{}.{}", declaration.name, manifest.target.suffix);
        objects.push(LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            format!("{logical_root}/{module_file}.runmat.json"),
            MEX_ARTIFACT_MANIFEST_MEDIA_TYPE,
            manifest_bytes.clone(),
        )?);
        objects.push(LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            format!("{logical_root}/{module_file}"),
            MEX_MODULE_MEDIA_TYPE,
            module_bytes.clone(),
        )?);
        entries.push(MexArtifactBundleEntry {
            manifest: manifest_bytes,
            module: module_bytes,
        });
    }

    let interop = InteropManifest::merge(manifests)
        .map_err(|error| anyhow::anyhow!("could not compose MEX requirements: {error}"))?;
    Ok(PreparedMexArtifacts {
        interop,
        objects,
        bundle: MexArtifactBundle::new(entries)?,
    })
}

fn read_exact(
    path: &std::path::Path,
    expected: &ContentDigest,
    kind: &str,
    name: &str,
) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read MEX artifact `{name}` {kind} `{}`", path.display()))?;
    let actual = ContentDigest::sha256(&bytes);
    if &actual != expected {
        bail!(
            "MEX artifact `{name}` {kind} changed after project resolution: expected {expected}, found {actual}"
        );
    }
    Ok(bytes)
}

fn logical_identity(identity: &str) -> String {
    identity
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_' {
                char::from(byte)
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::CapabilityRequirement;

    fn project() -> (tempfile::TempDir, FrozenProject) {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("src")).unwrap();
        std::fs::create_dir(temp.path().join("mex")).unwrap();
        std::fs::write(temp.path().join("src/main.m"), "x = packaged_fixture();\n").unwrap();
        let source = temp.path().join("mex/packaged_fixture.c");
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
        let artifact = runmat_mex::MexBuild::new(&source, temp.path().join("mex"))
            .compile()
            .unwrap();
        let manifest = artifact.manifest.strip_prefix(temp.path()).unwrap();
        let module = artifact.module.strip_prefix(temp.path()).unwrap();
        std::fs::write(
            temp.path().join("runmat.toml"),
            format!(
                "[package]\nname = \"mex-project\"\n[sources]\nroots = [\"src\"]\n[mex-artifacts.packaged_fixture]\nmanifest = \"{}\"\nmodule = \"{}\"\n",
                manifest.display(),
                module.display()
            ),
        )
        .unwrap();
        let project = runmat_package::build_frozen_project(
            &temp.path().join("runmat.toml"),
            BTreeSet::from([runmat_package::HostCapability::Mex]),
        )
        .unwrap();
        (temp, project)
    }

    #[test]
    fn exact_project_artifact_produces_adjacent_objects_and_aot_bundle() {
        let (_temp, project) = project();
        let prepared = prepare(&project).unwrap();
        assert_eq!(prepared.objects.len(), 2);
        assert_eq!(prepared.bundle.artifacts.len(), 1);
        let [adapter] = prepared.interop.adapters.as_slice() else {
            panic!("expected one MEX adapter requirement");
        };
        assert_eq!(adapter.adapter, runmat_mex::MEX_ADAPTER_ID);
        assert!(adapter
            .capabilities
            .0
            .contains(&CapabilityRequirement::NativeCode));
        assert!(!adapter
            .capabilities
            .0
            .contains(&CapabilityRequirement::Accelerator));
        let manifest_name = prepared
            .objects
            .iter()
            .find(|object| object.descriptor.media_type == MEX_ARTIFACT_MANIFEST_MEDIA_TYPE)
            .unwrap()
            .descriptor
            .logical_name
            .as_str();
        let module_name = prepared
            .objects
            .iter()
            .find(|object| object.descriptor.media_type == MEX_MODULE_MEDIA_TYPE)
            .unwrap()
            .descriptor
            .logical_name
            .as_str();
        assert_eq!(manifest_name, format!("{module_name}.runmat.json"));
    }

    #[test]
    fn artifact_changed_after_freeze_is_rejected() {
        let (_temp, project) = project();
        std::fs::write(&project.mex_artifacts[0].module_path, b"changed").unwrap();
        let error = prepare(&project).err().expect("changed artifact must fail");
        assert!(error
            .to_string()
            .contains("changed after project resolution"));
    }
}
