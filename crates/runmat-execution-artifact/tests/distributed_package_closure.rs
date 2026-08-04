use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{
    archive::{read_bundle, write_bundle, ArchiveLimits},
    encryption::{
        EncryptionContext, EncryptionPurpose, ExecutionEncryptionProvider,
        NativeExecutionEncryption,
    },
    ExecutableForm, ExecutionBundleBuilder, ProgramBuildRecipe,
};
use runmat_package::{
    build_resolved_graph, CanonicalPackageId, ContentDigest, FrozenProject, FrozenProjectHandoff,
    FrozenSourceDescriptor, GitCommitId, GitSourceId, NormalizedRelativePath, PackageInstanceId,
    PackageMount, PackageSourceCatalog, PackageVersion, PathSourceId, RegistryId, RegistryOrigin,
    RegistryReleaseId, RegistrySourceId, ResolvedDependencyInput, ResolvedGraphInput,
    ResolvedPackageInput, ServerProjectSourceId, SourceCatalog, SourceId, StableSourceId,
};
use serde::Serialize;

#[test]
fn every_source_kind_converges_into_one_portable_credential_free_closure() {
    let temp = tempfile::tempdir().unwrap();
    let source_inputs = source_inputs();
    let mut packages = BTreeMap::new();
    for (key, instance, _) in &source_inputs {
        packages.insert(
            key.clone(),
            ResolvedPackageInput {
                instance: instance.clone(),
                local_name: key.clone(),
                dependencies: if key == "root" {
                    ["git", "project", "registry"]
                        .into_iter()
                        .map(|target| ResolvedDependencyInput {
                            alias: target.parse().unwrap(),
                            target: target.to_string(),
                            group: runmat_package::DependencyGroup::Runtime,
                            optional: false,
                            target_predicate: None,
                        })
                        .collect()
                } else {
                    Vec::new()
                },
                required_capabilities: BTreeSet::new(),
                singleton: false,
            },
        );
    }
    let graph = build_resolved_graph(ResolvedGraphInput {
        root: "root".into(),
        packages,
        host_capabilities: BTreeSet::new(),
    })
    .unwrap();

    let mut catalogs = BTreeMap::new();
    let mut access_paths = BTreeMap::new();
    for (key, instance, source) in &source_inputs {
        let relative_path = NormalizedRelativePath::new(format!("src/{key}_answer.m")).unwrap();
        let bytes = format!("function y = {key}_answer(); y = 42; end\n").into_bytes();
        let path = temp.path().join(relative_path.as_str());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let stable = StableSourceId {
            package_instance: instance.identity_digest.clone(),
            relative_path,
            content_digest: ContentDigest::sha256(&bytes),
        };
        let logical_root = NormalizedRelativePath::new(format!(
            "packages/{}",
            instance.identity_digest.to_string().replace(':', "_")
        ))
        .unwrap();
        catalogs.insert(
            instance.identity_digest.clone(),
            PackageSourceCatalog {
                package_instance: instance.identity_digest.clone(),
                local_name: key.clone(),
                mount: PackageMount {
                    package_instance: instance.identity_digest.clone(),
                    source: source.clone(),
                    logical_root,
                },
                sources: vec![FrozenSourceDescriptor {
                    id: stable.clone(),
                    qualified_name: format!("{key}_answer"),
                    package_path: None,
                    class_name: None,
                    class_qualified_name: None,
                    is_private: false,
                }],
            },
        );
        access_paths.insert(stable, path);
    }
    let source_revision = source_revision(&graph.graph_digest, &catalogs);
    let project = FrozenProject {
        manifest_path: temp.path().join("runmat.toml"),
        workspace_root: temp.path().to_path_buf(),
        graph,
        sources: SourceCatalog {
            packages: catalogs,
            revision: source_revision,
        },
        access_paths,
    };
    let handoff = FrozenProjectHandoff::new(project.clone());
    handoff.validate().unwrap();
    let revision = revision(&project);
    let recipe = recipe(revision.clone());
    let bundle = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_materialized_program(
            recipe,
            ExecutableForm::InterpreterBytecodeV1,
            b"portable-bytecode".to_vec(),
        )
        .build()
        .unwrap();

    let kinds = bundle
        .manifest
        .project_handoff
        .project
        .graph
        .packages
        .values()
        .map(|package| match package.instance.source {
            SourceId::Path(_) => "path",
            SourceId::Git(_) => "git",
            SourceId::ServerProject(_) => "project",
            SourceId::Registry(_) => "registry",
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        kinds,
        BTreeSet::from(["git", "path", "project", "registry"])
    );

    let mut archive = Vec::new();
    write_bundle(&bundle, &mut archive, ArchiveLimits::default()).unwrap();
    let archive_text = String::from_utf8_lossy(&archive);
    assert!(!archive_text.contains(temp.path().to_string_lossy().as_ref()));
    assert!(!archive_text.contains("token"));
    assert!(!archive_text.contains("credential"));

    let encryption = NativeExecutionEncryption;
    let (recipient, private_key) = encryption
        .generate_recipient("credential-free-worker", 1, u64::MAX)
        .unwrap();
    let sealed = encryption
        .seal(
            &recipient,
            EncryptionContext {
                schema_version: 1,
                run_identity: "run-package-closure".into(),
                purpose: EncryptionPurpose::Bundle,
                object_digest: Digest::sha256(&archive),
                task_identity: None,
                attempt_identity: None,
                chunk_index: 0,
                total_length: archive.len() as u64,
                key_epoch: 1,
            },
            &archive,
        )
        .unwrap();
    let opened = encryption.open(&private_key, &sealed).unwrap();
    let decoded = read_bundle(opened.as_slice(), ArchiveLimits::default()).unwrap();
    assert_eq!(decoded.identity().unwrap(), bundle.identity().unwrap());
    assert_eq!(
        decoded.manifest.project_revision,
        bundle.manifest.project_revision
    );
    let worker_root = temp.path().join("worker-without-source-clients");
    let worker_handoff = decoded.project_handoff_at(&worker_root).unwrap();
    assert_eq!(worker_handoff.revision(), handoff.revision());
    assert!(worker_handoff
        .project
        .access_paths
        .values()
        .all(|path| path.starts_with(&worker_root)));
}

fn source_inputs() -> Vec<(String, PackageInstanceId, SourceId)> {
    let registry = RegistryId::default();
    let path_tree = ContentDigest::sha256("path-tree");
    let path = SourceId::Path(PathSourceId {
        workspace_path: ".".parse().unwrap(),
        manifest_digest: ContentDigest::sha256("path-manifest"),
        tree_digest: path_tree.clone(),
    });
    let git_tree = ContentDigest::sha256("git-tree");
    let git = SourceId::Git(
        GitSourceId::new(
            "https://example.com/acme/git-tools.git",
            GitCommitId::from_str("0123456789abcdef0123456789abcdef01234567").unwrap(),
            ".".parse().unwrap(),
            git_tree.clone(),
        )
        .unwrap(),
    );
    let project_tree = ContentDigest::sha256("project-tree");
    let project = SourceId::ServerProject(
        ServerProjectSourceId::new(
            "https://api.runmat.test",
            "proj_0123456789abcdef",
            "snap_0123456789abcdef",
            project_tree.clone(),
        )
        .unwrap(),
    );
    let registry_tree = ContentDigest::sha256("registry-tree");
    let registry_source = SourceId::Registry(RegistrySourceId {
        registry_origin: RegistryOrigin::new("https://packages.runmat.test").unwrap(),
        package: CanonicalPackageId::new(registry.clone(), "acme", "registry-tools").unwrap(),
        release: RegistryReleaseId::new("rel_0123456789abcdef0123456789abcdef").unwrap(),
        version: PackageVersion::from_str("1.2.3").unwrap(),
        release_digest: ContentDigest::sha256("registry-release"),
        artifact_digest: ContentDigest::sha256("registry-artifact"),
        tree_digest: registry_tree.clone(),
    });
    [
        ("root", "application", path, path_tree, Some("1.0.0")),
        ("git", "git-tools", git, git_tree, Some("2.0.0")),
        ("project", "project-tools", project, project_tree, None),
        (
            "registry",
            "registry-tools",
            registry_source,
            registry_tree,
            Some("1.2.3"),
        ),
    ]
    .into_iter()
    .map(|(key, name, source, tree, version)| {
        let package = CanonicalPackageId::new(registry.clone(), "fixture", name).unwrap();
        let instance = PackageInstanceId::new(
            package,
            source.clone(),
            version.map(|value| PackageVersion::from_str(value).unwrap()),
            tree,
        );
        (key.to_string(), instance, source)
    })
    .collect()
}

fn source_revision(
    graph_digest: &ContentDigest,
    packages: &BTreeMap<ContentDigest, PackageSourceCatalog>,
) -> ContentDigest {
    #[derive(Serialize)]
    struct Input<'a> {
        format: &'static str,
        graph_digest: &'a ContentDigest,
        packages: &'a BTreeMap<ContentDigest, PackageSourceCatalog>,
    }
    ContentDigest::sha256(
        serde_json::to_vec(&Input {
            format: "runmat-source-catalog-v1",
            graph_digest,
            packages,
        })
        .unwrap(),
    )
}

fn revision(project: &FrozenProject) -> ProgramRevision {
    ProgramRevision::new(
        Digest::from_bytes(*project.graph_digest().bytes()),
        Digest::from_bytes(*project.source_revision().bytes()),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

fn recipe(revision: ProgramRevision) -> ProgramBuildRecipe {
    ProgramBuildRecipe {
        schema_version: 1,
        program_revision: revision,
        entrypoint: "root_answer".into(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target_profile: "portable-package-closure-v1".into(),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    }
}
