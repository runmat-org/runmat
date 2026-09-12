use super::*;
use futures::executor::block_on;
use runmat_value::Value;

#[test]
fn geometry_inspect_builtin_returns_object_value() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("part.stl");
    std::fs::write(
            &path,
            "solid demo\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid demo\n",
        )
        .unwrap();

    let value = block_on(geometry_inspect_builtin(path.to_string_lossy().to_string()))
        .expect("inspect builtin should return an object");

    let Value::Object(result) = value else {
        panic!("expected object value");
    };
    assert!(result.class_name.is(GEOMETRY_INSPECT_RESULT_CLASS));
    assert!(result.properties.contains_key("format"));
    assert!(result.properties.contains_key("byte_count"));
}

#[test]
fn geometry_list_regions_builtin_returns_imported_regions() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("part.step");
    std::fs::write(
            &path,
            "ISO-10303-21;\nHEADER;\nFILE_NAME('Assembly_A');\nENDSEC;\nDATA;\n#10=PRODUCT('Bracket_A','',(#1));\nENDSEC;\nEND-ISO-10303-21;\n",
        )
        .unwrap();

    let asset = block_on(geometry_load_builtin(path.to_string_lossy().to_string()))
        .expect("geometry should load");
    let regions = block_on(geometry_list_regions_builtin(asset)).expect("regions should be listed");

    let Value::Struct(result) = regions else {
        panic!("expected struct value");
    };
    assert!(result.fields.contains_key("regions"));
}

#[test]
fn geometry_meshes_builtin_returns_patch_ready_surface_topology() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("part.stl");
    std::fs::write(
            &path,
            "solid demo\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid demo\n",
        )
        .unwrap();

    let asset = block_on(geometry_load_builtin(path.to_string_lossy().to_string()))
        .expect("geometry should load");
    let meshes = block_on(geometry_meshes_builtin(asset)).expect("meshes should project");

    let Value::Cell(cell) = meshes else {
        panic!("expected cell array of mesh structs");
    };
    assert_eq!(cell.data.len(), 1);
    let Value::Struct(mesh) = &cell.data[0] else {
        panic!("expected mesh struct");
    };
    let Some(Value::Tensor(vertices)) = mesh.fields.get("vertices") else {
        panic!("expected vertices tensor");
    };
    assert_eq!(vertices.shape, vec![3, 3]);
    let Some(Value::Tensor(faces)) = mesh.fields.get("faces") else {
        panic!("expected faces tensor");
    };
    assert_eq!(faces.shape, vec![1, 3]);
    assert_eq!(faces.materialize_f64(), vec![1.0, 2.0, 3.0]);
}
