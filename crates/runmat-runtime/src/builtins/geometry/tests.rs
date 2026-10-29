use super::*;
use futures::executor::block_on;
use runmat_value::Value;

#[path = "tests/path_admission.rs"]
mod path_admission;

const TRIANGLE_STL: &str = "solid demo\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid demo\n";

#[test]
fn geometry_load_returns_canonical_asset_without_native_cad() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("part.stl");
    std::fs::write(&path, TRIANGLE_STL).unwrap();
    let value = block_on(geometry_load_builtin(
        Value::String(path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    let Value::Object(asset) = value else {
        panic!("expected asset object")
    };
    assert!(asset.class_name().is(GEOMETRY_ASSET_CLASS));
    assert!(asset
        .properties()
        .contains_key(GEOMETRY_ASSET_JSON_PROPERTY));
    assert!(asset.properties().contains_key("surface_meshes"));
}

#[test]
fn geometry_file_read_error_retains_io_identifier() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("missing.stl");
    let error = block_on(geometry_load_builtin(
        Value::String(path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("missing file must fail");
    assert_eq!(error.identifier(), GEOMETRY_LOAD_ERROR_IO.identifier);
    assert!(error.source.is_some());
}

#[test]
fn geometry_output_count_is_rejected_before_file_read() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("missing.stl");
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(geometry_load_builtin(
        Value::String(path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("too many outputs must fail before I/O");
    assert_eq!(error.identifier(), GEOMETRY_LOAD_ERROR_OUTPUTS.identifier);
}

#[test]
fn geometry_inspect_output_count_is_rejected_before_file_read() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("missing.stl");
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(geometry_inspect_builtin(Value::String(
        path.to_string_lossy().to_string().into(),
    )))
    .expect_err("too many outputs must fail before I/O");
    assert_eq!(
        error.identifier(),
        GEOMETRY_INSPECT_ERROR_OUTPUTS.identifier
    );
}

#[test]
fn asset_queries_reject_nonasset_before_projection() {
    let regions = block_on(geometry_list_regions_builtin(Value::Num(3.0)))
        .expect_err("numeric receiver must fail");
    assert_eq!(
        regions.identifier(),
        GEOMETRY_INSPECT_ERROR_INTERNAL.identifier
    );
    let meshes =
        block_on(geometry_meshes_builtin(Value::Num(3.0))).expect_err("numeric receiver must fail");
    assert_eq!(
        meshes.identifier(),
        GEOMETRY_INSPECT_ERROR_INTERNAL.identifier
    );
}

#[test]
fn asset_method_registration_keeps_exact_public_spellings() {
    let methods = geometry_asset_methods();
    let regions = methods
        .get(&runmat_types::StaticMethodName::new("listRegions").owned())
        .expect("listRegions method");
    assert_eq!(regions.function_name, "geometry.listRegions");
    let meshes = methods
        .get(&runmat_types::StaticMethodName::new("meshes").owned())
        .expect("meshes method");
    assert_eq!(meshes.function_name, "geometry.meshes");
}

#[test]
fn geometry_inspect_builtin_returns_object_value() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("part.stl");
    std::fs::write(
            &path,
            "solid demo\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid demo\n",
        )
        .unwrap();

    let value = block_on(geometry_inspect_builtin(Value::String(
        path.to_string_lossy().to_string().into(),
    )))
    .expect("inspect builtin should return an object");

    let Value::Object(result) = value else {
        panic!("expected object value");
    };
    assert!(result.class_name().is(GEOMETRY_INSPECT_RESULT_CLASS));
    assert!(result.properties().contains_key("format"));
    assert!(result.properties().contains_key("byte_count"));
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

    let asset = block_on(geometry_load_builtin(
        Value::String(path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("geometry should load");
    let regions = block_on(geometry_list_regions_builtin(asset)).expect("regions should be listed");

    let Value::Struct(result) = regions else {
        panic!("expected struct value");
    };
    assert!(result.fields.contains_key("regions"));
}

#[test]
fn stl_region_projection_preserves_default_region_identity_and_count() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("part.stl");
    std::fs::write(&path, TRIANGLE_STL).unwrap();
    let asset = block_on(geometry_load_builtin(
        Value::String(path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("STL asset");
    let result = block_on(geometry_list_regions_builtin(asset)).expect("STL regions");
    let Value::Struct(outer) = result else {
        panic!("expected scalar result struct")
    };
    let Some(Value::Struct(region)) = outer.fields.get("regions") else {
        panic!("one imported region should be a scalar struct")
    };
    let region_id = region.fields.get("region_id").expect("region identity");
    assert_eq!(String::try_from(region_id).unwrap(), "region_default");
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

    let asset = block_on(geometry_load_builtin(
        Value::String(path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("geometry should load");
    let meshes = block_on(geometry_meshes_builtin(asset)).expect("meshes should project");

    let Value::Cell(cell) = meshes else {
        panic!("expected cell array of mesh structs");
    };
    assert_eq!(cell.data().len(), 1);
    let Value::Struct(mesh) = &cell.data()[0] else {
        panic!("expected mesh struct");
    };
    let Some(Value::Tensor(vertices)) = mesh.fields.get("vertices") else {
        panic!("expected vertices tensor");
    };
    assert_eq!(vertices.shape(), &[3, 3]);
    let Some(Value::Tensor(faces)) = mesh.fields.get("faces") else {
        panic!("expected faces tensor");
    };
    assert_eq!(faces.shape(), &[1, 3]);
    assert_eq!(faces.materialize_f64(), vec![1.0, 2.0, 3.0]);
}

fn load_with_options(path: &std::path::Path, options: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(geometry_load_builtin(
        Value::String(path.to_string_lossy().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(options),
    ))
}

#[test]
fn geometry_load_units_option_sets_the_asset_unit() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("part.stl");
    std::fs::write(&path, TRIANGLE_STL).unwrap();
    for (unit, expected) in [
        ("mm", runmat_geometry_core::LengthUnit::Millimeter),
        ("Inches", runmat_geometry_core::LengthUnit::Inch),
        ("metre", runmat_geometry_core::LengthUnit::Meter),
    ] {
        let value = load_with_options(
            &path,
            vec![Value::String("Units".into()), Value::String(unit.into())],
        )
        .unwrap_or_else(|error| panic!("{unit}: {error:?}"));
        let asset = geometry_asset_from_value_with_builtin(&value, GEOMETRY_LOAD_NAME).unwrap();
        assert_eq!(
            asset.coordinate_units.map(|units| units.unit()),
            Some(expected),
            "{unit}"
        );
    }
    let default = load_with_options(&path, Vec::new()).unwrap();
    let asset = geometry_asset_from_value_with_builtin(&default, GEOMETRY_LOAD_NAME).unwrap();
    // STL declares no length unit; without `Units` the asset carries no declared unit.
    assert_eq!(asset.coordinate_units.map(|units| units.unit()), None);
}

#[test]
fn geometry_load_rejects_invalid_options_before_reading_the_file() {
    let tmp = tempfile::TempDir::new().unwrap();
    // The file does not exist: option errors must win over I/O errors.
    let path = tmp.path().join("missing.stl");
    for options in [
        vec![Value::String("Units".into())],
        vec![Value::String("Scale".into()), Value::String("mm".into())],
        vec![Value::String("Units".into()), Value::String("ft".into())],
        vec![Value::String("Units".into()), Value::Num(1.0)],
        vec![Value::Num(1.0), Value::String("mm".into())],
        vec![
            Value::String("Units".into()),
            Value::String("mm".into()),
            Value::String("units".into()),
            Value::String("m".into()),
        ],
    ] {
        let error = load_with_options(&path, options).expect_err("invalid option");
        assert_eq!(
            error.identifier(),
            Some("RunMat:geometry:load:InvalidPath"),
            "{}",
            error.message()
        );
    }
}

#[cfg(feature = "plot-core")]
#[test]
fn geometry_mesh_method_reports_mesh_identifiers() {
    let error = block_on(geometry_mesh_method_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .expect_err("non-asset");
    assert_eq!(
        error.identifier(),
        Some("RunMat:geometry:mesh:InvalidInput")
    );
}
