use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Context, Result};
use runmat_execution_artifact::{LogicalObject, ObjectNamespace};
use runmat_java::{
    JavaArtifactBundle, JavaArtifactBundleEntry, JavaArtifactIdentity, JAVA_ARCHIVE_MEDIA_TYPE,
};
use runmat_package::{ContentDigest, FrozenProject};
use runmat_types::{
    CapabilityRequirement, CapabilitySet, ForeignAdapterRequirement, InteropManifest,
    INTEROP_MANIFEST_SCHEMA_VERSION,
};

pub(super) struct PreparedJavaArtifacts {
    pub(super) interop: InteropManifest,
    pub(super) objects: Vec<LogicalObject>,
    pub(super) bundle: JavaArtifactBundle,
}

pub(super) fn prepare(project: &FrozenProject) -> Result<PreparedJavaArtifacts> {
    let mut declarations = BTreeMap::new();
    for declaration in &project.java_artifacts {
        if declarations
            .insert(declaration.name.clone(), declaration)
            .is_some()
        {
            bail!(
                "Java artifact name `{}` is declared by more than one package",
                declaration.name
            );
        }
    }
    let mut identities = BTreeSet::new();
    let mut entries = Vec::with_capacity(project.java_artifacts.len());
    let mut objects = Vec::with_capacity(project.java_artifacts.len());
    for (logical_name, declaration) in declarations {
        let bytes = std::fs::read(&declaration.path).with_context(|| {
            format!(
                "read Java artifact `{}` at `{}`",
                declaration.name,
                declaration.path.display()
            )
        })?;
        let digest = ContentDigest::sha256(&bytes);
        if digest != declaration.digest {
            bail!(
                "Java artifact `{}` changed after project resolution: expected {}, found {}",
                declaration.name,
                declaration.digest,
                digest
            );
        }
        let identity = JavaArtifactIdentity::for_bytes(&bytes);
        if !identities.insert(identity.clone()) {
            bail!("Java artifact identity `{identity}` is declared more than once");
        }
        entries.push(JavaArtifactBundleEntry {
            logical_name: logical_name.clone(),
            identity: identity.clone(),
            bytes: bytes.clone(),
        });
        objects.push(LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            format!("java/{logical_name}.jar"),
            JAVA_ARCHIVE_MEDIA_TYPE,
            bytes,
        )?);
    }
    let bundle = JavaArtifactBundle::new(entries)?;
    let artifact_identities = identities
        .into_iter()
        .map(|identity| identity.to_string())
        .collect::<Vec<_>>();
    let interop = if artifact_identities.is_empty() {
        InteropManifest::empty()
    } else {
        InteropManifest {
            schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: vec![ForeignAdapterRequirement {
                adapter: runmat_java::JAVA_ADAPTER_ID.into(),
                minimum_version: runmat_java::JAVA_ADAPTER_VERSION,
                capabilities: CapabilitySet(BTreeSet::from([
                    CapabilityRequirement::ForeignRuntime,
                ])),
                artifact_identities,
            }],
            adapter_contracts: Vec::new(),
        }
    };
    interop
        .validate()
        .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
    Ok(PreparedJavaArtifacts {
        interop,
        objects,
        bundle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project() -> (tempfile::TempDir, FrozenProject) {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("src")).unwrap();
        std::fs::create_dir(temp.path().join("lib")).unwrap();
        std::fs::write(
            temp.path().join("src/main.m"),
            "value = fixture.Value.read();\n",
        )
        .unwrap();
        std::fs::write(
            temp.path().join("lib/fixture.jar"),
            b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
        )
        .unwrap();
        std::fs::write(
            temp.path().join("runmat.toml"),
            "[package]\nname = \"java-project\"\n[sources]\nroots = [\"src\"]\n[java-artifacts.fixture]\npath = \"lib/fixture.jar\"\n",
        )
        .unwrap();
        let project = runmat_package::build_frozen_project(
            &temp.path().join("runmat.toml"),
            BTreeSet::from([runmat_package::HostCapability::Jvm]),
        )
        .unwrap();
        (temp, project)
    }

    #[test]
    fn exact_java_artifact_produces_manifest_object_and_aot_bundle() {
        let (_temp, project) = project();
        let prepared = prepare(&project).unwrap();
        assert_eq!(prepared.objects.len(), 1);
        assert_eq!(prepared.bundle.artifacts.len(), 1);
        let [adapter] = prepared.interop.adapters.as_slice() else {
            panic!("expected Java adapter requirement");
        };
        assert_eq!(adapter.adapter, runmat_java::JAVA_ADAPTER_ID);
        assert_eq!(adapter.artifact_identities.len(), 1);
        assert_eq!(
            prepared.objects[0].descriptor.media_type,
            JAVA_ARCHIVE_MEDIA_TYPE
        );
    }

    #[test]
    fn artifact_changed_after_freeze_is_rejected() {
        let (_temp, project) = project();
        std::fs::write(&project.java_artifacts[0].path, b"changed").unwrap();
        let error = prepare(&project).err().expect("changed artifact must fail");
        assert!(error
            .to_string()
            .contains("changed after project resolution"));
    }

    #[test]
    fn all_execution_modes_share_logical_artifact_order() {
        let (_temp, mut project) = project();
        project.java_artifacts[0].name = "zeta".into();
        let alpha_path = project.workspace_root.join("lib/alpha.jar");
        let alpha_bytes = b"PK\x05\x06\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x01\0a";
        std::fs::write(&alpha_path, alpha_bytes).unwrap();
        project
            .java_artifacts
            .push(runmat_package::FrozenJavaArtifact {
                package_instance: project.java_artifacts[0].package_instance.clone(),
                name: "alpha".into(),
                digest: ContentDigest::sha256(alpha_bytes),
                path: alpha_path,
            });

        let prepared = prepare(&project).unwrap();
        assert_eq!(
            prepared
                .bundle
                .artifacts
                .iter()
                .map(|artifact| artifact.logical_name.as_str())
                .collect::<Vec<_>>(),
            ["alpha", "zeta"]
        );
        assert_eq!(
            prepared
                .objects
                .iter()
                .map(|object| object.descriptor.logical_name.as_str())
                .collect::<Vec<_>>(),
            ["java/alpha.jar", "java/zeta.jar"]
        );
    }
}
