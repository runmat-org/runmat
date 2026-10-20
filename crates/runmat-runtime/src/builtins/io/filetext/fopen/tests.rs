use super::*;
use crate::builtins::common::test_support;
use crate::builtins::io::filetext::registry;
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::system_time_now;
use runmat_value::{IntValue, IntegerStorage};
use std::io::Write;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::UNIX_EPOCH;

#[cfg(not(target_arch = "wasm32"))]
fn unique_path(prefix: &str) -> PathBuf {
    let now = system_time_now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards");
    let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
    std::env::temp_dir().join(filename)
}

#[cfg(target_arch = "wasm32")]
fn unique_path(prefix: &str) -> PathBuf {
    test_support::filesystem::unique_path(prefix)
}

fn run_evaluate(args: &[Value]) -> BuiltinResult<FopenEval> {
    futures::executor::block_on(evaluate(args))
}

fn registry_guard() -> registry::RegistryTestGuard {
    registry::test_guard()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = FOPEN_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"fid = fopen(filename)"));
    assert!(labels.contains(&"fid = fopen(filename, permission)"));
    assert!(labels.contains(&"fid = fopen(filename, permission, machinefmt)"));
    assert!(labels.contains(&"fid = fopen(filename, permission, machinefmt, encoding)"));
    assert!(labels.contains(&"[fid, msg] = fopen(filename, ...)"));
    assert!(labels.contains(&"[fid, msg, machinefmt, encoding] = fopen(filename, ...)"));
    assert!(labels.contains(&"filename = fopen(fid)"));
    assert!(labels.contains(&"[filename, permission, machinefmt, encoding] = fopen(fid)"));
    assert!(labels.contains(&"fids = fopen(\"all\")"));
    assert!(labels.contains(&"fids = fopen(\"all\", machinefmt)"));
    assert!(labels.contains(&"[fids, names, machinefmts, encodings] = fopen(\"all\")"));
}

#[test]
fn fopen_integer_capability_and_extension_policy_are_explicit() {
    assert_eq!(INTEGER_CAPABILITIES.len(), 1);
    assert_eq!(INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer = preflight_query_fid(&Value::Int(IntValue::I32(3))).unwrap_err();
    assert_eq!(
        integer.identifier(),
        Some("RunMat:compatibility:FopenIntegerIdExtension")
    );
    let legacy = run_evaluate(&[]).err().expect("legacy form is gated");
    assert_eq!(
        legacy.identifier(),
        Some("RunMat:compatibility:FopenLegacyAllExtension")
    );
}

#[test]
fn fopen_query_classifies_typed_integer_scalar_tensors_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor =
        Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("typed identifier");
    assert!(is_numeric_value(&Value::Tensor(tensor.clone())));
    assert_eq!(parse_fid(&Value::Tensor(tensor)).unwrap(), 3);
}

#[test]
fn invalid_open_controls_are_rejected_before_resident_materialization() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident fixture");
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let resident = Value::GpuTensor(allocation.handle.clone());

    let too_many = run_evaluate(&[
        Value::from("unused.bin"),
        Value::from("r"),
        Value::from("native"),
        Value::from("UTF-8"),
        resident.clone(),
    ])
    .err()
    .expect("arity must precede provider lookup");
    assert_eq!(too_many.identifier(), FOPEN_ERROR_INVALID_INPUT.identifier);

    let invalid_permission = run_evaluate(&[Value::from("unused.bin"), resident])
        .err()
        .expect("numeric permission must precede provider lookup");
    assert_eq!(
        invalid_permission.identifier(),
        FOPEN_ERROR_INVALID_PERMISSION.identifier
    );
    assert_eq!(allocation.probe.retirements(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_read_existing_file_returns_fid() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fopen_read");
    test_support::fs::write(&path, "hello world").unwrap();

    let args = vec![Value::from(path.to_string_lossy().to_string())];
    let eval = run_evaluate(&args).expect("fopen");
    let open = eval.as_open().expect("expected open result");
    assert!(open.fid >= 3.0);
    assert!(open.message.is_empty());
    assert_eq!(open.machinefmt, "native");
    assert_eq!(open.encoding, "UTF-8");

    let _ = registry::close(open.fid as i32);
    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_missing_file_returns_error() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fopen_missing");
    let args = vec![Value::from(path.to_string_lossy().to_string())];
    let eval = run_evaluate(&args).expect("fopen");
    let open = eval.as_open().expect("open output");
    assert_eq!(open.fid, -1.0);
    assert!(!open.message.is_empty());
    assert!(open.machinefmt.is_empty());
    assert!(open.encoding.is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_query_returns_metadata() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fopen_query");
    {
        let mut file = runmat_filesystem::File::create(&path).unwrap();
        writeln!(&mut file, "data").unwrap();
    }
    let args = vec![Value::from(path.to_string_lossy().to_string())];
    let eval = run_evaluate(&args).expect("fopen");
    let open = eval.as_open().expect("open result");
    let fid = open.fid;
    assert!(fid >= 3.0);

    let query_eval = run_evaluate(&[Value::from(fid)]).expect("fopen query");
    let query = query_eval.as_query().expect("query result");
    assert!(query.filename.contains("fopen_query"));
    assert_eq!(query.permission, "r");
    assert_eq!(query.machinefmt, "native");

    let _ = registry::close(fid as i32);
    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_all_lists_handles() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fopen_all");
    test_support::fs::write(&path, "abc").unwrap();
    let eval_open =
        run_evaluate(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
    let fid = eval_open.as_open().unwrap().fid;

    let list_eval = run_evaluate(&[Value::from("all")]).expect("fopen all");
    let list = list_eval.as_list().expect("list result");
    assert!(!list.handles.materialize_f64().is_empty());
    assert!(list
        .handles
        .materialize_f64()
        .iter()
        .any(|v| (*v - fid).abs() < f64::EPSILON));

    if let Value::Cell(names) = &list.names {
        assert_eq!(names.data().len(), list.handles.materialize_f64().len());
        assert_eq!(names.rows, list.handles.materialize_f64().len());
        assert_eq!(names.cols, 1);
    } else {
        panic!("expected cell array for names");
    }
    if let Value::Cell(machinefmts) = &list.machinefmts {
        assert_eq!(machinefmts.rows, list.handles.materialize_f64().len());
        assert_eq!(machinefmts.cols, 1);
    } else {
        panic!("expected cell array for machine formats");
    }
    if let Value::Cell(encodings) = &list.encodings {
        assert_eq!(encodings.rows, list.handles.materialize_f64().len());
        assert_eq!(encodings.cols, 1);
    } else {
        panic!("expected cell array for encodings");
    }

    let _ = registry::close(fid as i32);
    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_all_machinefmt_filters_entries() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let native_path = unique_path("fopen_native");
    let be_path = unique_path("fopen_ieee_be");
    test_support::fs::write(&native_path, "native").unwrap();
    test_support::fs::write(&be_path, "be").unwrap();

    let native_eval =
        run_evaluate(&[Value::from(native_path.to_string_lossy().to_string())]).expect("native");
    let native_fid = native_eval.as_open().unwrap().fid;

    let be_eval = run_evaluate(&[
        Value::from(be_path.to_string_lossy().to_string()),
        Value::from("r"),
        Value::from("ieee-be"),
    ])
    .expect("ieee-be");
    let be_fid = be_eval.as_open().unwrap().fid;

    let list_eval =
        run_evaluate(&[Value::from("all"), Value::from("ieee-be")]).expect("fopen all filter");
    let list = list_eval.as_list().expect("list result");
    assert_eq!(list.handles.materialize_f64().len(), 1);
    assert!((list.handles.materialize_f64()[0] - be_fid).abs() < f64::EPSILON);

    let _ = registry::close(native_fid as i32);
    let _ = registry::close(be_fid as i32);
    test_support::fs::remove_file(&native_path).unwrap();
    test_support::fs::remove_file(&be_path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_binary_default_encoding_binary() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fopen_binary");
    {
        let _ = runmat_filesystem::File::create(&path).unwrap();
    }
    let eval = run_evaluate(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("wb"),
    ])
    .expect("fopen");
    let open = eval.as_open().unwrap();
    assert_eq!(open.encoding, "binary");
    let _ = registry::close(open.fid as i32);
    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_encoding_argument_is_preserved() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fopen_encoding");
    let eval = run_evaluate(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w"),
        Value::from("ieee-be"),
        Value::from("latin1"),
    ])
    .expect("fopen");
    let open = eval.as_open().unwrap();
    assert_eq!(open.machinefmt, "ieee-be");
    assert_eq!(open.encoding, "latin1");
    let _ = registry::close(open.fid as i32);
    if path.exists() {
        test_support::fs::remove_file(&path).unwrap();
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_permission_canonicalizes_plus_binary_order() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fopen_perm_order");
    test_support::fs::write(&path, "seed").unwrap();
    let eval = run_evaluate(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("r+b"),
    ])
    .expect("fopen");
    let open = eval.as_open().unwrap();
    assert!(open.fid >= 3.0);
    assert_eq!(open.encoding, "binary");
    let query = run_evaluate(&[Value::Num(open.fid)]).expect("query");
    let info = query.as_query().unwrap();
    assert_eq!(info.permission, "rb+");
    let _ = registry::close(open.fid as i32);
    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_machinefmt_preserves_suffix() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fopen_machinefmt_suffix");
    let eval = run_evaluate(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w"),
        Value::from("ieee-be.l64"),
    ])
    .expect("fopen");
    let open = eval.as_open().unwrap();
    assert_eq!(open.machinefmt, "ieee-be.l64");
    let query = run_evaluate(&[Value::Num(open.fid)]).expect("query");
    let info = query.as_query().unwrap();
    assert_eq!(info.machinefmt, "ieee-be.l64");
    let _ = registry::close(open.fid as i32);
    if path.exists() {
        test_support::fs::remove_file(&path).unwrap();
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_machinefmt_pc_alias_maps_to_ieee_le() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fopen_machinefmt_pc");
    let eval = run_evaluate(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w"),
        Value::from("pc"),
    ])
    .expect("fopen");
    let open = eval.as_open().unwrap();
    assert_eq!(open.machinefmt, "ieee-le");
    let query = run_evaluate(&[Value::Num(open.fid)]).expect("query");
    let info = query.as_query().unwrap();
    assert_eq!(info.machinefmt, "ieee-le");
    let _ = registry::close(open.fid as i32);
    if path.exists() {
        test_support::fs::remove_file(&path).unwrap();
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_outputs_vector_padding() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fopen_outputs");
    test_support::fs::write(&path, "check").unwrap();
    let eval = run_evaluate(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
    let outputs = eval.outputs();
    assert_eq!(outputs.len(), 4);
    assert!(matches!(outputs[0], Value::Num(_)));
    assert!(matches!(outputs[1], Value::CharArray(_)));
    let _ = registry::close(eval.as_open().unwrap().fid as i32);
    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_invalid_fid_returns_empty() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let eval = run_evaluate(&[Value::from(9999.0)]).expect("fopen");
    let query = eval.as_query().expect("query result");
    assert!(query.filename.is_empty());
    assert!(query.permission.is_empty());
}
