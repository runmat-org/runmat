//! MATLAB-compatible `whos` builtin for RunMat.

use runmat_macros::runtime_builtin;
use runmat_value::{StructArray, Value};
use std::collections::HashSet;

#[cfg(test)]
use crate::builtins::introspection::workspace_metadata::value_memory_bytes;
use crate::builtins::introspection::workspace_metadata::WhosRecord;
use crate::builtins::introspection::workspace_query::{
    parse_request, WorkspaceQueryOwner, WorkspaceQuerySource,
};
use crate::builtins::io::mat::load::read_mat_file_for_builtin;
#[cfg(test)]
use crate::BuiltinResult;
use crate::{build_runtime_error, RuntimeError};

#[runtime_builtin(
    name = "whos",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::whos"
)]
async fn whos_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    let request = parse_request(args.view(), WorkspaceQueryOwner::Whos)?;

    let file_entries = match &request.source {
        WorkspaceQuerySource::File(path) => Some(read_mat_file_for_builtin(path, "whos").await?),
        WorkspaceQuerySource::Workspace => None,
    };
    let workspace_entries = if file_entries.is_none() {
        crate::workspace::try_snapshot()
            .map_err(whos_flow)?
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let entries = file_entries
        .as_ref()
        .map(|entries| entries.value().0.as_slice())
        .unwrap_or(&workspace_entries);

    let global_names: HashSet<String> = if matches!(request.source, WorkspaceQuerySource::Workspace)
    {
        crate::workspace::try_global_names()
            .map_err(whos_flow)?
            .into_iter()
            .collect()
    } else {
        HashSet::new()
    };

    let mut records = Vec::new();
    for (name, value) in entries {
        if !request.matches(name) {
            continue;
        }
        let is_global = global_names.contains(name);
        if request.only_global && !is_global {
            continue;
        }
        let record = WhosRecord::from_value(name.clone(), value, is_global).await?;
        records.push(record);
    }

    records.sort_by(|a, b| a.name().cmp(b.name()));

    let mut values = Vec::with_capacity(records.len());
    for record in records {
        let Value::Struct(value) = record.into_value()? else {
            unreachable!("whos records are scalar structures")
        };
        values.push(value);
    }
    let rows = values.len();
    StructArray::normalize(
        vec![
            "name".into(),
            "size".into(),
            "bytes".into(),
            "class".into(),
            "global".into(),
            "sparse".into(),
            "complex".into(),
            "nesting".into(),
            "persistent".into(),
        ],
        values,
        vec![rows, 1],
    )
    .map_err(|error| build_runtime_error(error).with_builtin("whos").build())
}

fn whos_flow(mut err: RuntimeError) -> RuntimeError {
    err.context = err.context.with_builtin("whos");
    err
}
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::call_builtin_async;
    use futures::executor::block_on;
    use runmat_thread_local::runmat_thread_local;
    use runmat_value::{
        CellArray, CharArray, ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericDType,
        StructValue as TestStruct, Tensor,
    };
    use std::cell::RefCell;
    use std::collections::{HashMap, HashSet};
    use tempfile::tempdir;

    fn whos_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::whos_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    runmat_thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
        static TEST_GLOBALS: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
    }

    pub(crate) fn ensure_test_resolver() {
        crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
            lookup: |name| TEST_WORKSPACE.with(|slot| slot.borrow().get(name).cloned()),
            snapshot: || {
                let mut entries: Vec<(String, Value)> =
                    TEST_WORKSPACE.with(|slot| slot.borrow().clone().into_iter().collect());
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                entries
            },
            globals: || TEST_GLOBALS.with(|slot| slot.borrow().iter().cloned().collect()),
            assign: None,
            clear: None,
            remove: None,
        });
    }

    pub(crate) fn set_workspace(entries: &[(&str, Value)], globals: &[&str]) {
        TEST_WORKSPACE.with(|slot| {
            let mut map = slot.borrow_mut();
            map.clear();
            for (name, value) in entries {
                map.insert((*name).to_string(), value.clone());
            }
        });
        TEST_GLOBALS.with(|slot| {
            let mut set = slot.borrow_mut();
            set.clear();
            for name in globals {
                set.insert((*name).to_string());
            }
        });
    }

    fn workspace_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::workspace::test_guard()
    }

    fn structs_from_value(value: Value) -> Vec<TestStruct> {
        match value {
            Value::StructArray(array) => array.into_elements(),
            Value::Struct(st) => vec![st],
            other => panic!("expected struct array, got {other:?}"),
        }
    }

    fn field_string(struct_value: &TestStruct, field: &str) -> Option<String> {
        struct_value
            .fields
            .get(field)
            .and_then(|value| match value {
                Value::String(runmat_value::StringElement::Text(s)) => Some(s.to_string()),
                Value::CharArray(ca) if ca.rows == 1 => {
                    ca.row_string().map(|text| text.trim().to_string())
                }
                _ => None,
            })
    }

    fn field_bool(struct_value: &TestStruct, field: &str) -> Option<bool> {
        struct_value
            .fields
            .get(field)
            .and_then(|value| match value {
                Value::Bool(b) => Some(*b),
                _ => None,
            })
    }

    fn field_bytes(struct_value: &TestStruct) -> Option<f64> {
        struct_value
            .fields
            .get("bytes")
            .and_then(|value| match value {
                Value::Num(n) => Some(*n),
                _ => None,
            })
    }

    fn field_size(struct_value: &TestStruct) -> Option<Vec<f64>> {
        struct_value
            .fields
            .get("size")
            .and_then(|value| match value {
                Value::Tensor(t) => Some(t.materialize_f64().clone()),
                _ => None,
            })
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    pub(crate) fn char_array_from_rows(rows: &[&str]) -> CharArray {
        let cols = rows
            .iter()
            .map(|text| text.encode_utf16().count())
            .max()
            .unwrap_or(0);
        let mut data = Vec::with_capacity(rows.len() * cols.max(1));
        for row in rows {
            let mut chars: Vec<char> = row.chars().collect();
            chars.extend(std::iter::repeat_n(' ', cols - row.encode_utf16().count()));
            if cols == 0 {
                continue;
            }
            data.extend(chars);
        }
        if cols == 0 {
            CharArray::from_code_units(Vec::new(), rows.len(), 0).expect("char array")
        } else {
            CharArray::from_unicode_scalars(data, rows.len(), cols).expect("char array")
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_lists_workspace_variables() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        set_workspace(
            &[("a", Value::Num(42.0)), ("b", Value::Tensor(tensor))],
            &[],
        );

        let value = whos_builtin(Vec::new()).expect("whos");
        let entries = structs_from_value(value);
        assert_eq!(entries.len(), 2);

        let first = entries
            .iter()
            .find(|st| field_string(st, "name").as_deref() == Some("a"))
            .expect("entry 'a'");
        assert_eq!(field_size(first).unwrap(), vec![1.0, 1.0]);
        assert_eq!(field_bytes(first).unwrap(), 8.0);
        assert_eq!(field_bool(first, "global"), Some(false));

        let second = entries
            .iter()
            .find(|st| field_string(st, "name").as_deref() == Some("b"))
            .expect("entry 'b'");
        assert_eq!(field_size(second).unwrap(), vec![2.0, 3.0]);
        assert_eq!(field_bytes(second).unwrap(), 48.0);
        assert_eq!(field_bool(second, "complex"), Some(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_filters_with_wildcard() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(
            &[("alpha", Value::Num(1.0)), ("beta", Value::Num(2.0))],
            &[],
        );

        let value = whos_builtin(vec![Value::from("a*")]).expect("whos");
        let entries = structs_from_value(value);
        assert_eq!(entries.len(), 1);
        assert_eq!(field_string(&entries[0], "name").unwrap(), "alpha");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_filters_with_regex() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(
            &[
                ("foo", Value::Num(1.0)),
                ("bar", Value::Num(2.0)),
                ("baz", Value::Num(3.0)),
            ],
            &[],
        );

        let value = whos_builtin(vec![Value::from("-regexp"), Value::from("^ba")]).expect("whos");
        let entries = structs_from_value(value);
        let names: Vec<String> = entries
            .iter()
            .map(|st| field_string(st, "name").unwrap())
            .collect();
        assert_eq!(names, vec!["bar", "baz"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_filters_global_only() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(
            &[("shared", Value::Num(1.0)), ("local", Value::Num(2.0))],
            &["shared"],
        );

        let value = whos_builtin(vec![Value::from("global")]).expect("whos");
        let entries = structs_from_value(value);
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(field_string(entry, "name").unwrap(), "shared");
        assert_eq!(field_bool(entry, "global"), Some(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_accepts_char_array_arguments() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("gamma", Value::Num(3.0)),
                ("theta", Value::Num(4.0)),
            ],
            &[],
        );

        let arg = Value::CharArray(char_array_from_rows(&["alpha", "gamma"]));
        let value = whos_builtin(vec![arg]).expect("whos");
        let entries = structs_from_value(value);
        let names: Vec<String> = entries
            .iter()
            .map(|st| field_string(st, "name").unwrap())
            .collect();
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_accepts_cell_array_arguments() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("gamma", Value::Num(3.0)),
                ("omega", Value::Num(4.0)),
            ],
            &[],
        );

        let cell = CellArray::new(vec![Value::from("gamma"), Value::from("alpha")], 2, 1).unwrap();
        let value = whos_builtin(vec![Value::Cell(cell)]).expect("whos");
        let entries = structs_from_value(value);
        let names: Vec<String> = entries
            .iter()
            .map(|st| field_string(st, "name").unwrap())
            .collect();
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_rejects_numeric_selection() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[], &[]);
        let err = whos_builtin(vec![Value::Num(7.0)]).expect_err("whos should error");
        let message = error_message(err);
        assert!(
            message.contains("whos: selections must"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn whos_rejects_resident_numeric_selector_with_owning_error() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[], &[]);
        test_support::with_test_provider(|_| {
            let resident = block_on(crate::call_builtin_async("gpuArray", &[Value::Num(7.0)]))
                .expect("resident scalar");
            let error =
                whos_builtin(vec![resident.value().clone()]).expect_err("selector is not text");
            assert_eq!(error.context.builtin.as_deref(), Some("whos"));
            assert!(error.message().contains("whos: selections must"));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_rejects_unknown_option() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[], &[]);
        let err = whos_builtin(vec![Value::from("-bogus")]).expect_err("whos should error");
        let message = error_message(err);
        assert!(
            message.contains("unsupported option"),
            "unexpected error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_requires_filename_for_file_option() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[], &[]);
        let err = whos_builtin(vec![Value::from("-file")]).expect_err("whos should error");
        let message = error_message(err);
        assert!(
            message.contains("'-file' requires a filename"),
            "error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_requires_pattern_for_regexp() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[], &[]);
        let err = whos_builtin(vec![Value::from("-regexp")]).expect_err("whos should error");
        let message = error_message(err);
        assert!(
            message.contains("'-regexp' requires at least one pattern"),
            "error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_rejects_invalid_regex() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[], &[]);
        let err = whos_builtin(vec![Value::from("-regexp"), Value::from("[")])
            .expect_err("whos should error");
        let message = error_message(err);
        assert!(
            message.contains("invalid regular expression"),
            "error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_file_option_reads_mat_file() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("beta", Value::Tensor(tensor.clone())),
            ],
            &[],
        );

        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("snapshot.mat");
        let path_str = file_path.to_string_lossy().to_string();
        block_on(call_builtin_async(
            "save",
            &[
                Value::from(path_str.clone()),
                Value::from("alpha"),
                Value::from("beta"),
            ],
        ))
        .expect("save");

        set_workspace(&[], &[]);
        let value = whos_builtin(vec![Value::from("-file"), Value::from(path_str)]).expect("whos");
        let entries = structs_from_value(value);
        let names: Vec<String> = entries
            .iter()
            .map(|st| field_string(st, "name").unwrap())
            .collect();
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_reports_gpu_bytes() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..16).map(|v| v as f64).collect(), vec![4, 4]).unwrap();
            let gpu_value = block_on(crate::call_builtin_async(
                "gpuArray",
                &[Value::Tensor(tensor.clone())],
            ))
            .expect("gpuArray");
            set_workspace(&[("G", gpu_value.value().clone())], &[]);

            let value = whos_builtin(vec![Value::from("G")]).expect("whos");
            let entries = structs_from_value(value);
            assert_eq!(entries.len(), 1);
            let bytes = field_bytes(&entries[0]).unwrap();
            let elem = match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F32 => 4.0,
                runmat_accelerate_api::ProviderPrecision::F64 => 8.0,
            };
            assert_eq!(bytes, elem * tensor.materialize_f64().len() as f64);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_reports_gpu_logical_bytes() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        test_support::with_test_provider(|provider| {
            let data = [0, 1, 0, 1];
            let handle = runmat_accelerate_api::upload_logical_protecting(
                provider,
                &runmat_accelerate_api::HostLogicalTensorView {
                    data: &data,
                    shape: &[4, 1],
                },
                &[],
            )
            .expect("logical producer upload");
            set_workspace(&[("logical_gpu", Value::GpuTensor(handle))], &[]);

            let value = whos_builtin(vec![Value::from("logical_gpu")]).expect("whos");
            let entries = structs_from_value(value);
            assert_eq!(entries.len(), 1);
            let bytes = field_bytes(&entries[0]).unwrap();
            assert_eq!(bytes, data.len() as f64);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn whos_reports_gpu_bytes_with_wgpu_provider() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let _provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("wgpu provider");
        let tensor = Tensor::new(vec![0.0; 12], vec![3, 4]).unwrap();
        let gpu_value = block_on(crate::call_builtin_async(
            "gpuArray",
            &[Value::Tensor(tensor.clone())],
        ))
        .expect("gpuArray");
        let Value::GpuTensor(handle) = gpu_value.value() else {
            panic!("gpuArray must retain a resident handle")
        };
        assert_eq!(handle.shape(), tensor.shape());
        assert_eq!(
            handle.physical().element_type(),
            runmat_accelerate_api::NumericElementType::F64
        );
        set_workspace(&[("WG", gpu_value.value().clone())], &[]);

        let value = whos_builtin(vec![Value::from("WG")]).expect("whos");
        let entries = structs_from_value(value);
        assert_eq!(entries.len(), 1);
        let bytes = field_bytes(&entries[0]).unwrap();
        let expected = (std::mem::size_of::<f64>() * tensor.materialize_f64().len()) as f64;
        assert!((bytes - expected).abs() < 1e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_reports_matlab_payload_width_for_typed_numeric_tensors() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let u8_tensor =
            Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::U8).unwrap();
        let u16_tensor =
            Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::U16)
                .unwrap();
        let i64_tensor = Tensor::new_integer(
            IntegerStorage::I64(vec![i64::MIN, -1, 0, i64::MAX]),
            vec![2, 2],
        )
        .unwrap();
        let u64_tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![0, 1, 9_007_199_254_740_993, u64::MAX]),
            vec![2, 2],
        )
        .unwrap();
        set_workspace(
            &[
                ("u8mat", Value::Tensor(u8_tensor)),
                ("u16mat", Value::Tensor(u16_tensor)),
                ("i64mat", Value::Tensor(i64_tensor)),
                ("u64mat", Value::Tensor(u64_tensor)),
            ],
            &[],
        );

        let value = whos_builtin(Vec::new()).expect("whos");
        let entries = structs_from_value(value);

        let u8_entry = entries
            .iter()
            .find(|st| field_string(st, "name").as_deref() == Some("u8mat"))
            .expect("u8mat");
        assert_eq!(field_bytes(u8_entry).unwrap(), 4.0);

        let u16_entry = entries
            .iter()
            .find(|st| field_string(st, "name").as_deref() == Some("u16mat"))
            .expect("u16mat");
        assert_eq!(field_bytes(u16_entry).unwrap(), 8.0);

        let i64_entry = entries
            .iter()
            .find(|st| field_string(st, "name").as_deref() == Some("i64mat"))
            .expect("i64mat");
        assert_eq!(field_bytes(i64_entry).unwrap(), 32.0);

        let u64_entry = entries
            .iter()
            .find(|st| field_string(st, "name").as_deref() == Some("u64mat"))
            .expect("u64mat");
        assert_eq!(field_bytes(u64_entry).unwrap(), 32.0);
    }

    #[test]
    fn whos_memory_bytes_use_native_width_for_all_integer_classes() {
        let storages = [
            IntegerStorage::I8(vec![1, 2]),
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I64(vec![1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ];
        let expected = [2, 4, 8, 16, 2, 4, 8, 16];

        for (storage, expected) in storages.into_iter().zip(expected) {
            let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
            assert_eq!(
                value_memory_bytes(&Value::Tensor(tensor), &mut HashSet::new()).expect("bytes"),
                expected
            );
        }
    }

    #[test]
    fn whos_memory_bytes_use_native_single_width() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("single tensor");
        assert_eq!(
            value_memory_bytes(&Value::Tensor(tensor), &mut HashSet::new()).expect("bytes"),
            12
        );
    }

    #[test]
    fn whos_memory_bytes_use_native_width_for_typed_sparse_integer_values() {
        let sparse = runmat_value::SparseTensor::new_integer(
            4,
            2,
            vec![0, 1, 2],
            vec![0, 3],
            IntegerStorage::U16(vec![1, u16::MAX]),
        )
        .expect("uint16 sparse");
        let expected = 2 * std::mem::size_of::<u16>()
            + 2 * std::mem::size_of::<usize>()
            + 3 * std::mem::size_of::<usize>();

        assert_eq!(
            value_memory_bytes(&Value::SparseTensor(sparse), &mut HashSet::new()).expect("bytes"),
            expected
        );
    }

    #[test]
    fn whos_memory_bytes_preserve_typed_complex_integer_component_width() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U16(vec![3, 4]),
        )
        .expect("typed complex storage");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).expect("typed complex tensor");

        assert_eq!(
            value_memory_bytes(&Value::ComplexTensor(tensor), &mut HashSet::new()).expect("bytes"),
            8
        );
    }

    #[test]
    fn resident_byte_estimates_follow_native_class_not_ambient_precision() {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
        };
        test_support::with_f32_test_provider(|_| {
            let owner = TestAllocationOwner::new();
            for element in [
                NumericElementType::F64,
                NumericElementType::F32,
                NumericElementType::I8,
                NumericElementType::I16,
                NumericElementType::I32,
                NumericElementType::I64,
                NumericElementType::U8,
                NumericElementType::U16,
                NumericElementType::U32,
                NumericElementType::U64,
            ] {
                for (storage, lanes) in [
                    (GpuTensorStorage::Real, 1),
                    (GpuTensorStorage::ComplexInterleaved, 2),
                ] {
                    let allocation = owner.allocate_zeroed(vec![2, 3], element, storage).unwrap();
                    let bytes = value_memory_bytes(
                        &Value::GpuTensor(allocation.handle),
                        &mut HashSet::new(),
                    )
                    .unwrap();
                    assert_eq!(bytes, 6 * element.element_size() * lanes);
                    assert_eq!(allocation.probe.retirements(), 1);
                }
            }
        });
    }
}
