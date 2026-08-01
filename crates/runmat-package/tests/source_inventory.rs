use runmat_config::project::{ProjectSourceFile, ProjectSourceIndex};
use runmat_package::{ContentDigest, SourceInventory, SOURCE_INVENTORY_SCHEMA_VERSION};
use std::path::PathBuf;

#[test]
fn source_inventory_is_checkout_independent_and_strictly_sorted() {
    let index = ProjectSourceIndex {
        files: vec![
            source("src", "+stats/mean.m", "stats.mean"),
            source("src", "main.m", "main"),
        ],
        package_dirs: vec![PathBuf::from("src/+stats")],
        class_dirs: Vec::new(),
        private_dirs: Vec::new(),
    };
    let inventory =
        SourceInventory::from_project_index(ContentDigest::sha256("tree"), index).unwrap();
    assert_eq!(inventory.schema_version, SOURCE_INVENTORY_SCHEMA_VERSION);
    assert_eq!(inventory.entries[0].qualified_name, "stats.mean");
    assert!(serde_json::to_string(&inventory)
        .unwrap()
        .contains("\"relative_path\":\"+stats/mean.m\""));
    inventory.validate().unwrap();
}

fn source(root: &str, path: &str, name: &str) -> ProjectSourceFile {
    ProjectSourceFile {
        source_root: PathBuf::from(root),
        relative_path: PathBuf::from(path),
        qualified_name: name.to_string(),
        package_path: None,
        class_name: None,
        class_qualified_name: None,
        is_private: false,
    }
}
