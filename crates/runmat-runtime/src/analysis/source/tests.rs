use std::fs;

use runmat_analysis_language::{
    GeometryInspectionRequest, GeometryInspectionSource, LogicalSourcePath,
};
use tempfile::tempdir;

use super::{check_fea_path_in_root, FilesystemFeaHost};

#[test]
fn filesystem_host_checks_a_complete_constructed_study() {
    let temporary = tempdir().unwrap();
    let source = r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh volume on geometry.part {
    tetrahedra {
        order = linear
        target_size = 0.1 m
        maximum_deviation = 0.01 m
    }
}
model structure {
}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve {
        structural static {
        }
    }
    output displacement = analysis.solve.field.node.displacement
}
"#;
    let path = temporary.path().join("main.fea");
    fs::write(&path, source).unwrap();

    let program = check_fea_path_in_root(temporary.path(), &path).unwrap();

    assert!(program.is_valid(), "{:#?}", program.diagnostics);
    assert_eq!(program.geometries.len(), 1);
    assert_eq!(program.meshes.len(), 1);
    assert_eq!(program.models.len(), 1);
    assert_eq!(program.studies.len(), 1);
}

#[test]
fn filesystem_host_confines_imports_to_the_declared_root() {
    let temporary = tempdir().unwrap();
    let source_root = temporary.path().join("project");
    fs::create_dir(&source_root).unwrap();
    let path = source_root.join("main.fea");
    fs::write(
        &path,
        "import model escaped from \"../outside.fea\"\nmodel local {}\n",
    )
    .unwrap();

    let program = check_fea_path_in_root(&source_root, &path).unwrap();

    assert!(!program.is_valid());
    assert!(program
        .bound
        .graph
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "fea.module.invalid_path"));
}

#[test]
fn filesystem_host_retains_normalized_ap242_exchange_facts() {
    let temporary = tempdir().unwrap();
    let geometry_path = temporary.path().join("part.step");
    fs::write(
        &geometry_path,
        include_bytes!(
            "../../../../runmat-geometry/io/tests/fixtures/ap242_exchange_metadata.step"
        ),
    )
    .unwrap();
    let host = FilesystemFeaHost::new(temporary.path()).unwrap();
    let exchange = host
        .inspect(&GeometryInspectionRequest {
            referrer: LogicalSourcePath::new("project/main.fea").unwrap(),
            logical_path: LogicalSourcePath::new("project/part.step").unwrap(),
            expected_content_digest: None,
        })
        .unwrap()
        .exchange
        .unwrap();

    assert_eq!(exchange.parameters.len(), 1);
    assert_eq!(exchange.materials.len(), 1);
    assert_eq!(exchange.engineering_properties.len(), 1);
    assert_eq!(
        exchange.parameters[0].persistent_id.as_deref(),
        Some("parameter-wall-thickness")
    );
    assert_eq!(
        exchange.materials[0].persistent_id.as_deref(),
        Some("material-structural-steel")
    );
}
