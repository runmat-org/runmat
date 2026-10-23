use super::super::format::{MatArray, MatClass, MatData, MAT_HEADER_LEN, MI_COMPRESSED};
use super::*;
use runmat_builtins::LOAD_DESCRIPTOR;
use runmat_value::{CellArray, ComplexTensor, IntValue, NumericStorage, SparseTensor, Tensor};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn numeric_resident_text_control_rejects_without_download() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let owner = Box::leak(Box::new(super::super::tests::CountingProvider::new()));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let resident = owner.resident();
    let error = block_on(evaluate(&[Value::GpuTensor(resident)]))
        .expect_err("numeric resident handle cannot be a MAT filename");
    assert_eq!(error.identifier(), Some("RunMat:load:InvalidArgument"));
    assert_eq!(owner.downloads(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selected_outer_load_revoked_during_pending_open_never_reads_or_publishes() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    use super::super::tests::receipt::{GatedOpenFs, Witness};

    let directory = "/mat";
    let filename = "/mat/gated.mat";
    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "answer".to_string(),
        Value::Num(42.0),
    )]))
    .expect("valid MAT input");
    let (provider, release) = GatedOpenFs::new(directory, filename, &bytes);
    let provider = std::sync::Arc::new(provider);
    let _selected = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir(directory).expect("virtual cwd");
    let witness = Witness::new(provider.clone());
    let mut future = Box::pin(super::load_builtin(
        Some(witness.runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from(filename)]),
    ));
    assert!(matches!(
        super::super::tests::receipt::poll_once(future.as_mut()),
        std::task::Poll::Pending
    ));
    assert_eq!(witness.selections(), 1);
    assert_eq!(witness.probes(), 1);
    assert_eq!(provider.opens(), 1);
    assert_eq!(provider.reads(), 0);
    witness.revoke();
    release.send(()).expect("release pending open");
    let error = block_on(future).expect_err("revoked source cannot enter decoder");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
    assert_eq!(provider.reads(), 0);
    assert_eq!(witness.probes(), 0);
    assert!(witness.outcomes().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selected_outer_load_revoked_by_first_read_stops_decode_before_more_reads() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    use super::super::tests::receipt::{GatedOpenFs, Witness};

    let directory = "/mat";
    let filename = "/mat/read-revoked.mat";
    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "answer".to_string(),
        Value::Num(42.0),
    )]))
    .expect("valid MAT input");
    let (provider, release) = GatedOpenFs::new(directory, filename, &bytes);
    let provider = std::sync::Arc::new(provider);
    let _selected = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir(directory).expect("virtual cwd");
    let witness = Witness::new(provider.clone());
    provider.revoke_on_read(witness.revocation_flag());
    release
        .send(())
        .expect("open immediately for first-read revocation");
    let error = block_on(super::load_builtin(
        Some(witness.runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from(filename)]),
    ))
    .expect_err("first read revokes codec continuation");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
    assert_eq!(provider.opens(), 1);
    assert_eq!(provider.reads(), 1);
    assert_eq!(witness.selections(), 1);
    assert_eq!(witness.probes(), 0);
    assert!(witness.outcomes().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_invalid_option_rejects_before_resident_preparation() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let args = [Value::from("case.mat"), Value::from("-unsupported")];
    let error = preflight_known_options(crate::call::FunctionArgumentView::from_slice(&args))
        .expect_err("unsupported flag");
    assert_eq!(error.identifier(), Some("RunMat:load:InvalidOption"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn regexp_followed_by_another_option_has_no_pattern() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let args = [
        Value::from("case.mat"),
        Value::from("-regexp"),
        Value::from("-mat"),
    ];
    let error = preflight_known_options(crate::call::FunctionArgumentView::from_slice(&args))
        .expect_err("missing regexp pattern");
    assert_eq!(error.identifier(), Some("RunMat:load:InvalidOption"));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn nested_implicit_load_does_not_inherit_outer_zero_output_workspace_mode() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let folder = tempdir().expect("MAT fixture folder");
    let path = folder.path().join("nested-output.mat");
    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "x".into(),
        Value::Num(7.0),
    )]))
    .expect("encode fixture");
    std::fs::write(&path, bytes).expect("write fixture");
    let _presentation = crate::output_context::push_output_count(0).expect("output-count scope");
    let value = block_on(crate::dispatcher::call_builtin_async(
        "load",
        &[Value::String(path.to_string_lossy().into_owned().into())],
    ))
    .expect("implicit nested load returns a struct");
    let Value::Struct(result) = value.value() else {
        panic!("implicit nested load must not publish workspace bindings");
    };
    assert_eq!(result.fields.get("x"), Some(&Value::Num(7.0)));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn current_zero_output_load_publishes_workspace_despite_outer_captured_mode() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[]);
    let folder = tempdir().expect("MAT fixture folder");
    let path = folder.path().join("current-zero.mat");
    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "x".into(),
        Value::Num(9.0),
    )]))
    .expect("encode fixture");
    std::fs::write(&path, bytes).expect("write fixture");
    let _presentation = crate::output_context::push_output_count(1).expect("output-count scope");
    let result = block_on(crate::dispatcher::call_builtin_async_with_outputs(
        "load",
        &[Value::String(path.to_string_lossy().into_owned().into())],
        0,
    ))
    .expect("zero-output statement");
    assert_eq!(result, Value::output_list(Vec::new()));
    TEST_WORKSPACE.with(|slot| assert_eq!(slot.borrow().get("x"), Some(&Value::Num(9.0))));
}
use crate::builtins::common::test_support::filesystem::tempdir;
use crate::builtins::io::mat::encoding::encode_workspace_to_mat_bytes;
use crate::workspace::WorkspaceResolver;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use futures::executor::block_on;
use runmat_thread_local::runmat_thread_local;
use runmat_value::{IntegerComplexStorage, IntegerStorage, NumericDType, StringArray};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;

runmat_thread_local! {
    static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
}

fn ensure_test_resolver() {
    crate::workspace::register_workspace_resolver(WorkspaceResolver {
        lookup: |name| TEST_WORKSPACE.with(|slot| slot.borrow().get(name).cloned()),
        snapshot: || {
            let mut entries: Vec<(String, Value)> =
                TEST_WORKSPACE.with(|slot| slot.borrow().clone().into_iter().collect());
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries
        },
        globals: || Vec::new(),
        assign: Some(|name, value| {
            TEST_WORKSPACE.with(|slot| {
                slot.borrow_mut().insert(name.to_string(), value);
            });
            Ok(())
        }),
        clear: None,
        remove: None,
    });
}

fn set_workspace(entries: &[(&str, Value)]) {
    TEST_WORKSPACE.with(|slot| {
        let mut map = slot.borrow_mut();
        map.clear();
        for (name, value) in entries {
            map.insert((*name).to_string(), value.clone());
        }
    });
}

fn assert_error_contains<T>(result: crate::BuiltinResult<T>, snippet: &str) {
    match result {
        Err(err) => {
            assert!(
                err.message().contains(snippet),
                "expected error to contain '{snippet}', got '{}'",
                err.message()
            );
        }
        Ok(_) => panic!("expected error containing '{snippet}'"),
    }
}

fn wrap_payload_as_compressed(mat_bytes: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&mat_bytes[MAT_HEADER_LEN..]).unwrap();
    let compressed = encoder.finish().unwrap();

    let mut out = mat_bytes[..MAT_HEADER_LEN].to_vec();
    out.extend_from_slice(&MI_COMPRESSED.to_le_bytes());
    out.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    out.extend_from_slice(&compressed);
    out
}

fn load_entries_from_bytes(bytes: Vec<u8>) -> Vec<(String, Value)> {
    let mut cursor = Cursor::new(bytes);
    read_mat_reader(&mut cursor).expect("read MAT bytes")
}

#[test]
fn parsed_struct_fields_are_consumed_in_mat_field_major_order() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    fn scalar(value: f64) -> MatArray {
        MatArray {
            class: MatClass::Double,
            dims: vec![1, 1],
            data: MatData::Numeric {
                real: NumericStorage::F64(vec![value]),
                imag: None,
            },
        }
    }

    // This models the ordered matrix subelements from an independently
    // produced Level-5 structure: every element of `left`, then `right`.
    let fixture = MatArray {
        class: MatClass::Struct,
        dims: vec![2, 2],
        data: MatData::Struct {
            field_names: vec!["left".into(), "right".into()],
            field_values: vec![
                scalar(1.0),
                scalar(2.0),
                scalar(3.0),
                scalar(4.0),
                scalar(10.0),
                scalar(20.0),
                scalar(30.0),
                scalar(40.0),
            ],
        },
    };
    let Value::StructArray(array) = mat_array_to_value(fixture).unwrap() else {
        panic!("expected structure array")
    };
    assert_eq!(array.shape(), [2, 2]);
    assert_eq!(
        array.field_values("left").unwrap(),
        [
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ]
    );
    assert_eq!(
        array.field_values("right").unwrap(),
        [
            Value::Num(10.0),
            Value::Num(20.0),
            Value::Num(30.0),
            Value::Num(40.0),
        ]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_descriptor_signatures_cover_core_forms() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let labels: Vec<&str> = LOAD_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"S = load()"));
    assert!(labels.contains(&"S = load(filename)"));
    assert!(labels.contains(&"S = load(filename, varName1, varName2, ...)"));
    assert!(labels.contains(&"S = load(filename, '-regexp', pattern1, ...)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_roundtrip_numeric() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
    set_workspace(&[("A", Value::Tensor(tensor))]);

    let dir = tempdir().unwrap();
    let path = dir.path().join("numeric.mat");
    let save_arg = Value::from(path.to_string_lossy().to_string());
    block_on(crate::call_builtin_async(
        "save",
        std::slice::from_ref(&save_arg),
    ))
    .unwrap();

    let eval = block_on(evaluate(&[Value::from(path.to_string_lossy().to_string())]))
        .expect("load numeric");
    let struct_value = eval.first_output().unwrap();
    match struct_value.value() {
        Value::Struct(sv) => {
            assert!(sv.fields.contains_key("A"));
            match sv.fields.get("A").unwrap() {
                Value::Tensor(t) => {
                    assert_eq!(t.shape(), &[2, 2]);
                    assert_eq!(t.materialize_f64(), vec![1.0, 4.0, 2.0, 5.0]);
                }
                other => panic!("expected tensor, got {other:?}"),
            }
        }
        other => panic!("expected struct, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_selected_variables() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[("signal", Value::Num(42.0)), ("noise", Value::Num(5.0))]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("selection.mat");
    let save_arg = Value::from(path.to_string_lossy().to_string());
    block_on(crate::call_builtin_async(
        "save",
        std::slice::from_ref(&save_arg),
    ))
    .unwrap();

    let eval = block_on(evaluate(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("signal"),
    ]))
    .expect("load selection");
    let vars = eval.variables();
    assert_eq!(vars.len(), 1);
    assert_eq!(vars[0].0, "signal");
    assert!(matches!(vars[0].1, Value::Num(42.0)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_regex_selection() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[
        ("w1", Value::Num(1.0)),
        ("w2", Value::Num(2.0)),
        ("bias", Value::Num(3.0)),
    ]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("regex.mat");
    let save_arg = Value::from(path.to_string_lossy().to_string());
    block_on(crate::call_builtin_async(
        "save",
        std::slice::from_ref(&save_arg),
    ))
    .unwrap();

    let eval = block_on(evaluate(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("-regexp"),
        Value::from("^w\\d$"),
    ]))
    .expect("load regex");
    let mut names: Vec<_> = eval.variables().iter().map(|(n, _)| n.clone()).collect();
    names.sort();
    assert_eq!(names, vec!["w1".to_string(), "w2".to_string()]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_missing_variable_errors() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[("existing", Value::Num(7.0))]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("missing.mat");
    let save_arg = Value::from(path.to_string_lossy().to_string());
    block_on(crate::call_builtin_async(
        "save",
        std::slice::from_ref(&save_arg),
    ))
    .unwrap();

    assert_error_contains(
        block_on(evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("missing"),
        ])),
        "variable 'missing' was not found",
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_string_array_roundtrip() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    let strings = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
    set_workspace(&[("labels", Value::StringArray(strings))]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("strings.mat");
    let save_arg = Value::from(path.to_string_lossy().to_string());
    block_on(crate::call_builtin_async(
        "save",
        std::slice::from_ref(&save_arg),
    ))
    .unwrap();

    let eval = block_on(evaluate(&[Value::from(path.to_string_lossy().to_string())]))
        .expect("load strings");
    let struct_value = eval.first_output().unwrap();
    match struct_value.value() {
        Value::Struct(sv) => {
            let value = sv
                .fields
                .get("labels")
                .expect("labels field missing in struct");
            match value {
                Value::StringArray(sa) => {
                    assert_eq!(sa.shape, vec![1, 2]);
                    assert_eq!(sa.data(), vec![String::from("foo"), String::from("bar")]);
                }
                other => panic!("expected string array, got {other:?}"),
            }
        }
        other => panic!("expected struct, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mat_string_cell_roundtrip_decodes_utf16_surrogate_pairs() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let source =
        StringArray::new(vec!["A😀".into(), "plain".into()], vec![1, 2]).expect("two strings");
    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "labels".into(),
        Value::StringArray(source.clone()),
    )]))
    .expect("encode UTF-16 MAT cell");
    let entries = load_entries_from_bytes(bytes);
    let Value::StringArray(actual) = &entries[0].1 else {
        panic!("expected string array from char cells");
    };
    assert_eq!(actual.shape, source.shape);
    assert_eq!(actual.data(), source.data());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mat_unicode_char_row_roundtrip_restores_scalar_geometry() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let source = CharArray::new_row("A😀");
    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "text".into(),
        Value::CharArray(source.clone()),
    )]))
    .expect("encode Unicode character row");
    let entries = load_entries_from_bytes(bytes);
    assert_eq!(entries[0].1, Value::CharArray(source));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn external_utf16_char_matrix_preserves_rectangular_code_unit_geometry() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let value = mat_array_to_value(MatArray {
        class: MatClass::Char,
        dims: vec![2, 1],
        data: MatData::Char {
            data: vec![0xd83d, 0xde00],
        },
    })
    .expect("MAT UTF-16 units are the canonical character payload");
    let Value::CharArray(array) = value else {
        panic!("expected character array");
    };
    assert_eq!(array.shape(), &[2, 1]);
    assert_eq!(
        array
            .data()
            .iter()
            .map(|unit| unit.raw())
            .collect::<Vec<_>>(),
        vec![0xd83d, 0xde00]
    );
}

#[test]
fn load_save_nd_char_and_cell_roundtrip_preserves_shape_and_order() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    let chars = CharArray::from_unicode_scalars_with_shape("abcd".chars().collect(), vec![1, 2, 2])
        .expect("chars");
    let cells = CellArray::new_with_shape(vec![Value::Num(1.0), Value::Num(2.0)], vec![1, 1, 2])
        .expect("cells");
    set_workspace(&[
        ("chars", Value::CharArray(chars.clone())),
        ("cells", Value::Cell(cells.clone())),
    ]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("nd_char_cell.mat");
    let save_arg = Value::from(path.to_string_lossy().to_string());
    block_on(crate::call_builtin_async(
        "save",
        std::slice::from_ref(&save_arg),
    ))
    .expect("save nd values");

    let eval = block_on(evaluate(&[Value::from(path.to_string_lossy().to_string())]))
        .expect("load nd values");
    let output = eval.first_output().unwrap();
    let Value::Struct(output) = output.value() else {
        panic!("expected struct");
    };
    assert_eq!(output.fields.get("chars"), Some(&Value::CharArray(chars)));
    assert_eq!(output.fields.get("cells"), Some(&Value::Cell(cells)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_compressed_level5_payload() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let mat_bytes = block_on(encode_workspace_to_mat_bytes(&[
        ("A".to_string(), Value::Num(123.0)),
        ("B".to_string(), Value::from("text")),
    ]))
    .unwrap();
    let compressed = wrap_payload_as_compressed(&mat_bytes);

    let entries = load_entries_from_bytes(compressed);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].0, "A");
    assert!(matches!(entries[0].1, Value::Num(123.0)));
    assert_eq!(entries[1].0, "B");
    match &entries[1].1 {
        Value::CharArray(ca) => assert_eq!(
            runmat_value::decode_utf16_code_units(ca.data()).unwrap(),
            "text"
        ),
        other => panic!("expected char array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_preserves_supported_numeric_mat_classes() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let single = Tensor::new_with_dtype(vec![1.5, 2.5], vec![1, 2], NumericDType::F32)
        .expect("single tensor");
    let single_scalar = Tensor::from_f32(vec![0.1], vec![1, 1]).expect("single scalar");
    let complex_single = ComplexTensor::from_f32(vec![(0.1, -0.2), (3.5, 4.25)], vec![1, 2])
        .expect("complex single tensor");
    let uint16 = Tensor::new_with_dtype(vec![10.0, 20.0], vec![1, 2], NumericDType::U16)
        .expect("uint16 tensor");
    let bytes = block_on(encode_workspace_to_mat_bytes(&[
        ("s".to_string(), Value::Tensor(single)),
        ("ss".to_string(), Value::Tensor(single_scalar)),
        ("cs".to_string(), Value::ComplexTensor(complex_single)),
        ("u16".to_string(), Value::Tensor(uint16)),
        ("i".to_string(), Value::Int(IntValue::I16(-7))),
    ]))
    .unwrap();

    let entries = load_entries_from_bytes(bytes);
    let values: HashMap<_, _> = entries.into_iter().collect();
    match values.get("s").unwrap() {
        Value::Tensor(t) => {
            assert_eq!(t.numeric_dtype(), NumericDType::F32);
            assert_eq!(t.materialize_f64(), vec![1.5, 2.5]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    match values.get("u16").unwrap() {
        Value::Tensor(t) => {
            assert_eq!(t.numeric_dtype(), NumericDType::U16);
            assert_eq!(t.materialize_f64(), vec![10.0, 20.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    match values.get("ss").unwrap() {
        Value::Tensor(t) => {
            assert_eq!(t.numeric_dtype(), NumericDType::F32);
            assert_eq!(
                t.clone().into_numeric_storage(),
                Ok(NumericStorage::F32(vec![0.1_f32]))
            );
        }
        other => panic!("expected single scalar tensor, got {other:?}"),
    }
    match values.get("cs").unwrap() {
        Value::ComplexTensor(t) => {
            assert_eq!(t.numeric_dtype(), NumericDType::F32);
            assert_eq!(
                t.as_f32_slice().map(|values| values
                    .iter()
                    .copied()
                    .map(<(f32, f32)>::from)
                    .collect::<Vec<_>>()),
                Some(vec![(0.1_f32, -0.2_f32), (3.5_f32, 4.25_f32)])
            );
        }
        other => panic!("expected complex single tensor, got {other:?}"),
    }
    assert_eq!(values.get("i"), Some(&Value::Int(IntValue::I16(-7))));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_save_roundtrip_preserves_every_integer_tensor_class_exactly() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let cases = vec![
        ("i8", IntegerStorage::I8(vec![i8::MIN, i8::MAX])),
        ("u8", IntegerStorage::U8(vec![0, u8::MAX])),
        ("i16", IntegerStorage::I16(vec![i16::MIN, i16::MAX])),
        ("u16", IntegerStorage::U16(vec![0, u16::MAX])),
        ("i32", IntegerStorage::I32(vec![i32::MIN, i32::MAX])),
        ("u32", IntegerStorage::U32(vec![0, u32::MAX])),
        ("i64", IntegerStorage::I64(vec![i64::MIN, i64::MAX])),
        ("u64", IntegerStorage::U64(vec![1_u64 << 63, u64::MAX])),
    ];
    let mut entries = Vec::new();
    for (name, storage) in &cases {
        let tensor = Tensor::new_integer(storage.clone(), vec![2, 1]).expect("integer tensor");
        entries.push(((*name).to_string(), Value::Tensor(tensor)));
    }
    let empty = Tensor::new_integer(IntegerStorage::U64(Vec::new()), vec![0, 2])
        .expect("empty integer tensor");
    entries.push(("empty_u64".to_string(), Value::Tensor(empty)));
    entries.push((
        "scalar_u64".to_string(),
        Value::Int(IntValue::U64(u64::MAX)),
    ));

    let bytes = block_on(encode_workspace_to_mat_bytes(&entries)).expect("encode MAT bytes");
    let values: HashMap<_, _> = load_entries_from_bytes(bytes).into_iter().collect();

    for (name, storage) in cases {
        match values.get(name).expect("loaded integer tensor") {
            Value::Tensor(tensor) => assert_eq!(tensor.integer_storage(), Some(&storage)),
            other => panic!("expected tensor for {name}, got {other:?}"),
        }
    }
    match values.get("empty_u64").expect("loaded empty tensor") {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[0, 2]);
            assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::U64(Vec::new()))
            );
        }
        other => panic!("expected empty tensor, got {other:?}"),
    }
    assert_eq!(
        values.get("scalar_u64"),
        Some(&Value::Int(IntValue::U64(u64::MAX)))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_save_roundtrip_preserves_every_complex_integer_tensor_class_exactly() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let cases = vec![
        (
            "i8",
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I8(vec![1, -2]),
        ),
        (
            "u8",
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U8(vec![3, 4]),
        ),
        (
            "i16",
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I16(vec![5, -6]),
        ),
        (
            "u16",
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U16(vec![7, 8]),
        ),
        (
            "i32",
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I32(vec![9, -10]),
        ),
        (
            "u32",
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U32(vec![11, 12]),
        ),
        (
            "i64",
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::I64(vec![13, i64::MIN]),
        ),
        (
            "u64",
            IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 14]),
        ),
    ];
    let mut entries = Vec::new();
    for (name, real, imag) in &cases {
        let storage = IntegerComplexStorage::new(real.clone(), imag.clone()).expect("components");
        let tensor = ComplexTensor::new_integer(storage, vec![2, 1]).expect("typed complex");
        entries.push(((*name).to_string(), Value::ComplexTensor(tensor)));
    }

    let bytes = block_on(encode_workspace_to_mat_bytes(&entries)).expect("encode MAT bytes");
    let values: HashMap<_, _> = load_entries_from_bytes(bytes).into_iter().collect();

    for (name, real, imag) in cases {
        match values.get(name).expect("loaded typed complex tensor") {
            Value::ComplexTensor(tensor) => assert_eq!(
                tensor.integer_storage().cloned(),
                Some(IntegerComplexStorage::new(real, imag).expect("components"))
            ),
            other => panic!("expected typed complex tensor for {name}, got {other:?}"),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_save_roundtrip_preserves_nested_integer_tensors() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let unsigned =
        Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
            .expect("unsigned tensor");
    let signed = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![2, 1])
        .expect("signed tensor");
    let cell = crate::make_cell(
        vec![
            Value::Tensor(unsigned.clone()),
            Value::Tensor(signed.clone()),
        ],
        1,
        2,
    )
    .expect("cell");
    let mut nested = runmat_value::UnboundedStructValueBuilder::new();
    nested.insert("unsigned", Value::Tensor(unsigned));
    nested.insert("values", cell);

    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "nested".to_string(),
        Value::Struct(nested.finish()),
    )]))
    .expect("encode MAT bytes");
    let values: HashMap<_, _> = load_entries_from_bytes(bytes).into_iter().collect();

    let Value::Struct(nested) = values.get("nested").expect("nested struct") else {
        panic!("expected struct");
    };
    match nested.fields.get("unsigned") {
        Some(Value::Tensor(tensor)) => assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]))
        ),
        other => panic!("expected unsigned tensor, got {other:?}"),
    }
    match nested.fields.get("values") {
        Some(Value::Cell(cell)) => {
            match &cell.data()[0] {
                Value::Tensor(tensor) => assert_eq!(
                    tensor.integer_storage(),
                    Some(&IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]))
                ),
                other => panic!("expected unsigned cell tensor, got {other:?}"),
            }
            match &cell.data()[1] {
                Value::Tensor(tensor) => assert_eq!(
                    tensor.integer_storage(),
                    Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))
                ),
                other => panic!("expected signed cell tensor, got {other:?}"),
            }
        }
        other => panic!("expected cell, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_save_real_sparse_roundtrip() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let sparse = SparseTensor::new(3, 3, vec![0, 1, 1, 3], vec![1, 0, 2], vec![4.0, 5.0, 6.0])
        .expect("sparse");
    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "S".to_string(),
        Value::SparseTensor(sparse.clone()),
    )]))
    .unwrap();

    let entries = load_entries_from_bytes(bytes);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, "S");
    assert_eq!(entries[0].1, Value::SparseTensor(sparse));
}

#[test]
fn load_save_native_single_sparse_roundtrip_preserves_dtype_and_values() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let sparse =
        SparseTensor::new_f32(3, 3, vec![0, 1, 1, 3], vec![1, 0, 2], vec![4.25, 5.5, 6.75])
            .expect("single sparse");
    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "single_sparse".to_string(),
        Value::SparseTensor(sparse.clone()),
    )]))
    .expect("save single sparse");
    let entries = load_entries_from_bytes(bytes);
    assert_eq!(
        entries,
        vec![("single_sparse".to_string(), Value::SparseTensor(sparse))]
    );
}

#[test]
fn load_save_logical_sparse_roundtrip_preserves_pattern_and_class() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let sparse = SparseTensor::new_logical(3, 3, vec![0, 1, 1, 3], vec![1, 0, 2]).expect("logical");
    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "logical_sparse".to_string(),
        Value::SparseTensor(sparse.clone()),
    )]))
    .expect("save logical sparse");
    let entries = load_entries_from_bytes(bytes);
    assert_eq!(
        entries,
        vec![("logical_sparse".to_string(), Value::SparseTensor(sparse))]
    );
}

#[test]
fn registered_load_gates_restored_sparse_integer_by_compatibility_mode() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let sparse = SparseTensor::new_integer(
        1,
        1,
        vec![0, 1],
        vec![0],
        IntegerStorage::U64(vec![u64::MAX]),
    )
    .expect("sparse integer");
    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "S".to_string(),
        Value::SparseTensor(sparse),
    )]))
    .expect("encode sparse integer");
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sparse_integer.mat");
    crate::builtins::common::test_support::fs::write(&path, bytes).expect("write MAT file");
    let args = [Value::String(path.to_string_lossy().into_owned().into())];

    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = crate::dispatcher::call_builtin("load", &args)
            .expect_err("MATLAB mode must reject restored sparse integer");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:SparseIntegerExtension")
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let restored = crate::dispatcher::call_builtin("load", &args).expect("RunMat mode load");
        assert!(matches!(
            restored.value(),
            Value::Struct(ref fields)
                if matches!(
                    fields.fields.get("S"),
                    Some(Value::SparseTensor(sparse))
                        if sparse.integer_storage()
                            == Some(&IntegerStorage::U64(vec![u64::MAX]))
                )
        ));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_save_sparse_integer_roundtrips_preserve_every_dtype_and_exact_values() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let cases = [
        (
            "i8",
            SparseTensor::new_integer(
                2,
                1,
                vec![0, 2],
                vec![0, 1],
                IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            )
            .expect("int8 sparse"),
        ),
        (
            "u8",
            SparseTensor::new_integer(
                2,
                1,
                vec![0, 2],
                vec![0, 1],
                IntegerStorage::U8(vec![0, u8::MAX]),
            )
            .expect("uint8 sparse"),
        ),
        (
            "i16",
            SparseTensor::new_integer(
                2,
                1,
                vec![0, 2],
                vec![0, 1],
                IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            )
            .expect("int16 sparse"),
        ),
        (
            "u16",
            SparseTensor::new_integer(
                2,
                1,
                vec![0, 2],
                vec![0, 1],
                IntegerStorage::U16(vec![0, u16::MAX]),
            )
            .expect("uint16 sparse"),
        ),
        (
            "i32",
            SparseTensor::new_integer(
                2,
                1,
                vec![0, 2],
                vec![0, 1],
                IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            )
            .expect("int32 sparse"),
        ),
        (
            "u32",
            SparseTensor::new_integer(
                2,
                1,
                vec![0, 2],
                vec![0, 1],
                IntegerStorage::U32(vec![0, u32::MAX]),
            )
            .expect("uint32 sparse"),
        ),
        (
            "i64",
            SparseTensor::new_integer(
                4,
                3,
                vec![0, 1, 1, 3],
                vec![0, 1, 3],
                IntegerStorage::I64(vec![i64::MIN + 1, i64::MAX, -(1_i64 << 54)]),
            )
            .expect("int64 sparse"),
        ),
        (
            "u64",
            SparseTensor::new_integer(
                2,
                4,
                vec![0, 0, 1, 2, 2],
                vec![1, 0],
                IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            )
            .expect("uint64 sparse"),
        ),
    ];

    let workspace: Vec<_> = cases
        .iter()
        .map(|(name, sparse)| (name.to_string(), Value::SparseTensor(sparse.clone())))
        .collect();
    let bytes = block_on(encode_workspace_to_mat_bytes(&workspace)).expect("encode MAT bytes");
    let entries: HashMap<_, _> = load_entries_from_bytes(bytes).into_iter().collect();

    for (name, expected) in cases {
        let Value::SparseTensor(actual) = entries.get(name).expect("loaded sparse") else {
            panic!("expected sparse matrix");
        };
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.col_ptrs, expected.col_ptrs);
        assert_eq!(actual.row_indices, expected.row_indices);
        assert_eq!(actual.integer_storage(), expected.integer_storage());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_save_empty_sparse_integer_roundtrip_preserves_dtype_and_shape() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let sparse = SparseTensor::new_integer(
        0,
        5,
        vec![0, 0, 0, 0, 0, 0],
        Vec::new(),
        IntegerStorage::U64(Vec::new()),
    )
    .expect("empty sparse");
    let bytes = block_on(encode_workspace_to_mat_bytes(&[(
        "empty".to_string(),
        Value::SparseTensor(sparse),
    )]))
    .expect("encode MAT bytes");

    let entries = load_entries_from_bytes(bytes);
    let Value::SparseTensor(actual) = &entries[0].1 else {
        panic!("expected sparse matrix");
    };
    assert_eq!(actual.shape(), vec![0, 5]);
    assert_eq!(actual.col_ptrs, vec![0, 0, 0, 0, 0, 0]);
    assert_eq!(
        actual.integer_storage(),
        Some(&IntegerStorage::U64(Vec::new()))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_option_before_filename() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[("alpha", Value::Num(1.0)), ("beta", Value::Num(2.0))]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("option_first.mat");
    let save_arg = Value::from(path.to_string_lossy().to_string());
    block_on(crate::call_builtin_async(
        "save",
        std::slice::from_ref(&save_arg),
    ))
    .unwrap();

    let eval = block_on(evaluate(&[
        Value::from("-mat"),
        Value::from(path.to_string_lossy().to_string()),
        Value::from("beta"),
    ]))
    .expect("load with option first");
    let vars = eval.variables();
    assert_eq!(vars.len(), 1);
    assert_eq!(vars[0].0, "beta");
    assert!(matches!(vars[0].1, Value::Num(2.0)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_char_array_names_trimmed() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[("short", Value::Num(5.0)), ("longer", Value::Num(9.0))]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("char_names.mat");
    let save_arg = Value::from(path.to_string_lossy().to_string());
    block_on(crate::call_builtin_async(
        "save",
        std::slice::from_ref(&save_arg),
    ))
    .unwrap();

    let cols = 6;
    let mut data = Vec::new();
    for name in ["short", "longer"] {
        let mut chars: Vec<char> = name.chars().collect();
        while chars.len() < cols {
            chars.push(' ');
        }
        data.extend(chars);
    }
    let name_array = CharArray::from_unicode_scalars(data, 2, cols).unwrap();

    let eval = block_on(evaluate(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::CharArray(name_array),
    ]))
    .expect("load with char array names");
    let vars = eval.variables();
    assert_eq!(vars.len(), 2);
    assert_eq!(vars[0].0, "short");
    assert!(matches!(vars[0].1, Value::Num(5.0)));
    assert_eq!(vars[1].0, "longer");
    assert!(matches!(vars[1].1, Value::Num(9.0)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn load_duplicate_names_last_wins() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[("dup", Value::Num(11.0))]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("duplicates.mat");
    let save_arg = Value::from(path.to_string_lossy().to_string());
    block_on(crate::call_builtin_async(
        "save",
        std::slice::from_ref(&save_arg),
    ))
    .unwrap();

    let eval = block_on(evaluate(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("dup"),
        Value::from("dup"),
    ]))
    .expect("load with duplicate names");
    let vars = eval.variables();
    assert_eq!(vars.len(), 1);
    assert_eq!(vars[0].0, "dup");
    assert!(matches!(vars[0].1, Value::Num(11.0)));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn load_wgpu_tensor_roundtrip() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };

    use runmat_accelerate_api::HostTensorView;

    let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: tensor.shape(),
    };
    let handle = provider.upload(&view).expect("upload tensor");
    set_workspace(&[("gpu_var", Value::GpuTensor(handle))]);

    let dir = tempdir().unwrap();
    let path = dir.path().join("wgpu_load.mat");
    let save_args = vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::from("gpu_var"),
    ];
    block_on(crate::call_builtin_async("save", &save_args)).unwrap();

    let eval = block_on(evaluate(&[Value::from(path.to_string_lossy().to_string())]))
        .expect("load wgpu file");
    let struct_value = eval.first_output().unwrap();
    match struct_value.value() {
        Value::Struct(sv) => match sv.fields.get("gpu_var") {
            Some(Value::Tensor(t)) => {
                assert_eq!(t.shape(), &[2, 2]);
                assert_eq!(t.materialize_f64(), tensor.materialize_f64());
            }
            other => panic!("expected tensor, got {other:?}"),
        },
        other => panic!("expected struct, got {other:?}"),
    }
}
