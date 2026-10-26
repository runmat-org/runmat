//! Stage-source invalidation must track computation ownership rather than all
//! Rust files in the workspace. These fixtures execute the actual build helper.
#[path = "../build_support/meshing_fingerprint.rs"]
mod meshing_fingerprint;
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture {
    root: PathBuf,
    metadata: Value,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "runmat-meshing-source-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let names = [
            "runmat-meshing-execution",
            "runmat-geometry-io",
            "runmat-geometry-ops",
            "runmat-execution-runner",
            "runmat-filesystem",
            "runmat-cpu-execution",
            "runmat-execution-runner-native",
            "runmat-meshing-tetrahedron",
            "unrelated-physics",
            "runmat-cli",
            "shared-math",
        ];
        let packages = names.iter().map(|name| {
            let directory = root.join("crates").join(name);
            fs::create_dir_all(directory.join("src")).unwrap();
            fs::write(directory.join("Cargo.toml"), format!("[package]\nname = '{name}'\nversion = '0.0.0'\n")).unwrap();
            fs::write(directory.join("src/lib.rs"), "pub const VALUE: u8 = 1;\n").unwrap();
            let dependencies = if *name == "runmat-meshing-execution" {
                json!([{"path":root.join("crates/runmat-meshing-tetrahedron"),"kind":null}, {"path":root.join("crates/shared-math"),"kind":"build"}])
            } else { json!([]) };
            json!({"name":name,"manifest_path":directory.join("Cargo.toml"),"dependencies":dependencies})
        }).collect::<Vec<_>>();
        Self {
            root,
            metadata: json!({"packages":packages}),
        }
    }
    fn fingerprint(&self, volume: bool, context: &[u8]) -> String {
        meshing_fingerprint::fingerprint(&self.root, volume, context, false, Some(&self.metadata))
            .unwrap()
    }
    fn change(&self, name: &str) {
        fs::write(
            self.root.join("crates").join(name).join("src/lib.rs"),
            "pub const VALUE: u8 = 2;\n",
        )
        .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}
#[test]
fn unrelated_physics_preserves_both_stage_fingerprints() {
    let fixture = Fixture::new();
    let before = [
        fixture.fingerprint(false, b"configuration"),
        fixture.fingerprint(true, b"configuration"),
    ];
    fixture.change("unrelated-physics");
    assert_eq!(
        before,
        [
            fixture.fingerprint(false, b"configuration"),
            fixture.fingerprint(true, b"configuration")
        ]
    );
}
// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A tetrahedron-only source edit changes volume identity while preserving upstream
//   identity in an explicit dependency fixture. Current independent geometry/mesh admission remains
//   required on reuse; this does not qualify a fresh original CAD execution.
#[test]
fn tetrahedron_changes_invalidate_volume_and_preserve_upstream() {
    let fixture = Fixture::new();
    let upstream = fixture.fingerprint(false, b"configuration");
    let volume = fixture.fingerprint(true, b"configuration");
    fixture.change("runmat-meshing-tetrahedron");
    assert_eq!(upstream, fixture.fingerprint(false, b"configuration"));
    assert_ne!(volume, fixture.fingerprint(true, b"configuration"));
}
// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Actual build-helper fixtures bind both stage fingerprints to transitive build
//   dependencies, native adapters and configuration changes. Optional local dependencies remain
//   conservatively included; this is not exact active-feature reachability or an observed
//   production cache hit.
#[test]
fn transitive_math_adapter_and_configuration_changes_invalidate_consumers() {
    for name in [
        "shared-math",
        "runmat-geometry-io",
        "runmat-execution-runner-native",
    ] {
        let fixture = Fixture::new();
        let upstream = fixture.fingerprint(false, b"configuration");
        let volume = fixture.fingerprint(true, b"configuration");
        fixture.change(name);
        assert_ne!(upstream, fixture.fingerprint(false, b"configuration"));
        assert_ne!(volume, fixture.fingerprint(true, b"configuration"));
        assert_ne!(
            fixture.fingerprint(false, b"configuration"),
            fixture.fingerprint(false, b"new configuration")
        );
    }
}
#[test]
fn missing_metadata_disables_reuse() {
    let fixture = Fixture::new();
    assert_eq!(
        meshing_fingerprint::fingerprint(&fixture.root, true, b"configuration", false, None)
            .unwrap(),
        "unavailable"
    );
}
#[test]
fn current_workspace_has_a_complete_stage_source_closure() {
    let workspace = std::env::var_os("RUNMAT_MESHING_TEST_WORKSPACE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf()
        });
    let metadata = meshing_fingerprint::metadata(&workspace)
        .unwrap()
        .expect("complete workspace metadata");
    for volume in [false, true] {
        let fingerprint = meshing_fingerprint::fingerprint(
            &workspace,
            volume,
            b"test configuration",
            false,
            Some(&metadata),
        )
        .unwrap();
        assert_eq!(fingerprint.len(), 64);
    }
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A regular source module below src/target changes both stage identities. This checks
//   source visibility, not a production cache hit or numerical mesh accuracy.
#[test]
fn nested_source_target_directory_invalidates_both_stage_fingerprints() {
    let fixture = Fixture::new();
    let source = fixture
        .root
        .join("crates/shared-math/src/target/selection.rs");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    fs::write(&source, "pub const VALUE: u8 = 1;\n").unwrap();
    let before = [
        fixture.fingerprint(false, b"configuration"),
        fixture.fingerprint(true, b"configuration"),
    ];
    fs::write(&source, "pub const VALUE: u8 = 2;\n").unwrap();
    assert_ne!(before[0], fixture.fingerprint(false, b"configuration"));
    assert_ne!(before[1], fixture.fingerprint(true, b"configuration"));
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Package target/artifacts/.artifacts changes preserve stage identity while
//   src/artifacts remains source. The actual Cargo watch plan excludes the package root and watches
//   its source subtree; this does not measure cache throughput.
#[test]
fn package_output_changes_preserve_both_stage_fingerprints() {
    for output in ["target", "artifacts", ".artifacts"] {
        let fixture = Fixture::new();
        let source = fixture
            .root
            .join("crates/shared-math/src/artifacts/identity.rs");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, "pub const VALUE: u8 = 1;\n").unwrap();
        let inventory = meshing_fingerprint::source_inventory::collect(
            &fixture.root.join("crates/shared-math"),
            &std::collections::BTreeSet::new(),
        )
        .unwrap();
        assert!(!inventory
            .watches
            .contains(&fixture.root.join("crates/shared-math")));
        assert!(inventory
            .watches
            .contains(&fixture.root.join("crates/shared-math/src")));
        let before = [
            fixture.fingerprint(false, b"configuration"),
            fixture.fingerprint(true, b"configuration"),
        ];
        let generated = fixture
            .root
            .join("crates/shared-math")
            .join(output)
            .join("generated.rs");
        fs::create_dir_all(generated.parent().unwrap()).unwrap();
        fs::write(&generated, "pub const VALUE: u8 = 9;\n").unwrap();
        assert_eq!(
            before,
            [
                fixture.fingerprint(false, b"configuration"),
                fixture.fingerprint(true, b"configuration")
            ]
        );
        fs::write(&source, "pub const VALUE: u8 = 2;\n").unwrap();
        assert_ne!(before[0], fixture.fingerprint(false, b"configuration"));
        assert_ne!(before[1], fixture.fingerprint(true, b"configuration"));
    }
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An unsupported linked source entry produces unavailable identity and cannot authorize
//   cached work. This finite native fixture does not qualify arbitrary link graphs or a fresh
//   numerical solve.
#[cfg(unix)]
#[test]
fn linked_source_disables_reuse_instead_of_omitting_implementation() {
    let fixture = Fixture::new();
    let source = fixture.root.join("crates/shared-math/src/linked.rs");
    std::os::unix::fs::symlink("lib.rs", source).unwrap();
    assert_eq!(fixture.fingerprint(false, b"configuration"), "unavailable");
    assert_eq!(fixture.fingerprint(true, b"configuration"), "unavailable");
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A changed JSON input in an inventoried local package invalidates both stage
//   identities. Workspace-external generated inputs and exact active-feature reachability are not
//   established by this fixture.
#[test]
fn non_rust_package_inputs_invalidate_both_stage_fingerprints() {
    let fixture = Fixture::new();
    let source = fixture.root.join("crates/shared-math/src/tolerances.json");
    fs::write(&source, "{\"tolerance\":1}\n").unwrap();
    let before = [
        fixture.fingerprint(false, b"configuration"),
        fixture.fingerprint(true, b"configuration"),
    ];
    fs::write(&source, "{\"tolerance\":2}\n").unwrap();
    assert_ne!(before[0], fixture.fingerprint(false, b"configuration"));
    assert_ne!(before[1], fixture.fingerprint(true, b"configuration"));
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An unrelated registered Cargo package nested below a selected package is excluded by
//   ownership, while the selected package source remains influential. This is a
//   dependency-visibility fixture, not whole-study resume qualification.
#[test]
fn nested_unrelated_package_does_not_become_a_dependency_by_directory_layout() {
    let mut fixture = Fixture::new();
    let child = fixture.root.join("crates/shared-math/nested/unrelated");
    fs::create_dir_all(child.join("src")).unwrap();
    fs::write(
        child.join("Cargo.toml"),
        "[package]\nname='nested-unrelated'\nversion='0.0.0'\n",
    )
    .unwrap();
    fs::write(child.join("src/lib.rs"), "pub const VALUE: u8 = 1;\n").unwrap();
    fixture.metadata["packages"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "name":"nested-unrelated", "manifest_path":child.join("Cargo.toml"), "dependencies":[]
        }));
    let before = [
        fixture.fingerprint(false, b"configuration"),
        fixture.fingerprint(true, b"configuration"),
    ];
    fs::write(child.join("src/lib.rs"), "pub const VALUE: u8 = 2;\n").unwrap();
    assert_eq!(
        before,
        [
            fixture.fingerprint(false, b"configuration"),
            fixture.fingerprint(true, b"configuration")
        ]
    );
    fixture.change("shared-math");
    assert_ne!(before[0], fixture.fingerprint(false, b"configuration"));
    assert_ne!(before[1], fixture.fingerprint(true, b"configuration"));
}

#[test]
fn native_header_changes_invalidate_stages_while_cli_rendering_preserves_them() {
    let fixture = Fixture::new();
    let header = fixture.root.join("crates/shared-math/src/kernel.hxx");
    fs::write(&header, "original-header").unwrap();
    let before = [
        fixture.fingerprint(false, b"configuration"),
        fixture.fingerprint(true, b"configuration"),
    ];
    fs::write(&header, "changed-native-header").unwrap();
    assert_ne!(before[0], fixture.fingerprint(false, b"configuration"));
    assert_ne!(before[1], fixture.fingerprint(true, b"configuration"));
    fs::write(&header, "original-header").unwrap();
    fixture.change("runmat-cli");
    assert_eq!(
        before,
        [
            fixture.fingerprint(false, b"configuration"),
            fixture.fingerprint(true, b"configuration")
        ]
    );
}
