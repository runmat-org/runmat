use super::*;
use runmat_value::IntValue;
use runmat_value::{ComplexTensor, IntegerComplexStorage, StructValue};

#[path = "tests/handle_receipt.rs"]
mod handle_receipt;
#[path = "tests/slice_cooperation.rs"]
mod slice_cooperation;
#[path = "tests/publication.rs"]
mod publication;

fn tensor_values(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

#[test]
fn typed_data_slice_indices_do_not_saturate_through_i64() {
    let extent = usize::MAX;
    let result = parse_dim_range(&Value::Int(IntValue::U64(u64::MAX)), extent);
    match usize::try_from(u64::MAX)
        .ok()
        .and_then(|value| value.checked_sub(1))
    {
        Some(index) if index < extent => {
            let range = result.expect("index");
            assert_eq!(range.start, index);
            assert_eq!(range.end, index + 1);
        }
        _ => assert!(result.is_err()),
    }
    assert!(parse_dim_range(&Value::Int(IntValue::I64(-1)), extent).is_err());
}

#[test]
fn data_shape_and_slice_parsers_read_typed_integer_storage_exactly() {
    let cases = [
        runmat_value::IntegerStorage::I8(vec![2, 3]),
        runmat_value::IntegerStorage::I16(vec![2, 3]),
        runmat_value::IntegerStorage::I32(vec![2, 3]),
        runmat_value::IntegerStorage::I64(vec![2, 3]),
        runmat_value::IntegerStorage::U8(vec![2, 3]),
        runmat_value::IntegerStorage::U16(vec![2, 3]),
        runmat_value::IntegerStorage::U32(vec![2, 3]),
        runmat_value::IntegerStorage::U64(vec![2, 3]),
    ];
    for storage in cases {
        let shape = Tensor::new_integer(storage.clone(), vec![1, 2]).expect("shape");
        assert_eq!(
            parse_shape_from_value(&Value::Tensor(shape)).expect("shape"),
            vec![2, 3]
        );

        let range = Tensor::new_integer(storage, vec![1, 2]).expect("range");
        let parsed = parse_dim_range(&Value::Tensor(range), 5).expect("range");
        assert_eq!(parsed.start, 1);
        assert_eq!(parsed.end, 3);
    }

    let negative = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![-1]), vec![1, 1])
        .expect("shape");
    assert!(parse_shape_from_value(&Value::Tensor(negative)).is_err());

    let invalid = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![u64::MAX, u64::MAX]),
        vec![1, 2],
    )
    .expect("range");
    assert!(parse_dim_range(&Value::Tensor(invalid), 5).is_err());
}

#[test]
fn data_shape_and_slice_floating_parsers_reject_fractional_and_unrepresentable_bounds() {
    assert!(parse_shape_from_value(&Value::Num(1.5)).is_err());
    let boundary_shape = parse_shape_from_value(&Value::Num(usize::MAX as f64));
    if usize::BITS == 64 {
        assert!(boundary_shape.is_err());
    } else {
        assert_eq!(boundary_shape.unwrap(), vec![usize::MAX]);
    }
    assert!(parse_shape_from_value(&Value::Num((usize::MAX as f64) + 1.0)).is_err());

    let fractional_shape = Tensor::new(vec![2.0, 3.25], vec![1, 2]).expect("shape");
    assert!(parse_shape_from_value(&Value::Tensor(fractional_shape)).is_err());
    let single_shape = Tensor::from_f32(vec![2.0, 3.0], vec![1, 2]).expect("single shape");
    assert_eq!(
        parse_shape_from_value(&Value::Tensor(single_shape)).expect("single shape"),
        vec![2, 3]
    );
    let fractional_single_shape =
        Tensor::from_f32(vec![2.0, 3.25], vec![1, 2]).expect("fractional single shape");
    assert!(parse_shape_from_value(&Value::Tensor(fractional_single_shape)).is_err());

    assert!(parse_dim_range(&Value::Num(1.5), 5).is_err());
    let boundary_range = parse_dim_range(&Value::Num(usize::MAX as f64), usize::MAX);
    if usize::BITS == 64 {
        assert!(boundary_range.is_err());
    } else {
        let parsed = boundary_range.expect("32-bit boundary");
        assert_eq!(parsed.start, usize::MAX - 1);
        assert_eq!(parsed.end, usize::MAX);
    }

    let fractional_range = Tensor::new(vec![1.0, 3.5], vec![1, 2]).expect("range");
    assert!(parse_dim_range(&Value::Tensor(fractional_range), 5).is_err());
    let single_range = Tensor::from_f32(vec![2.0, 3.0], vec![1, 2]).expect("single range");
    let parsed = parse_dim_range(&Value::Tensor(single_range), 5).expect("single range");
    assert_eq!((parsed.start, parsed.end), (1, 3));

    let too_large_range = Tensor::new(vec![1.0, usize::MAX as f64], vec![1, 2]).expect("range");
    let tensor_boundary = parse_dim_range(&Value::Tensor(too_large_range), usize::MAX);
    if usize::BITS == 64 {
        assert!(tensor_boundary.is_err());
    } else {
        let parsed = tensor_boundary.expect("32-bit tensor boundary");
        assert_eq!(parsed.start, 0);
        assert_eq!(parsed.end, usize::MAX);
    }
}
use crate::dispatcher::call_builtin;
use async_trait::async_trait;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{post, put};
use axum::{Json, Router};
use runmat_filesystem::data_contract::{
    DataChunkUploadRequest, DataChunkUploadTarget, DataManifestDescriptor, DataManifestRequest,
};
use runmat_filesystem::{
    DirEntry, FileHandle, FsMetadata, FsProvider, NativeFsProvider, OpenFlags,
};
use runmat_value::{CellArray, IntegerStorage};
use serde::Deserialize;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::runtime::Runtime;
use tokio::sync::oneshot;

fn serial_test_guard() -> MutexGuard<'static, ()> {
    runmat_filesystem::provider_override_lock()
}

fn native_provider_guard() -> runmat_filesystem::ProviderGuard {
    runmat_filesystem::replace_provider(Arc::new(NativeFsProvider))
}

#[test]
fn attribute_json_preserves_native_integer_text() {
    for value in [
        runmat_value::IntValue::I64(i64::MIN),
        runmat_value::IntValue::U64(u64::MAX),
    ] {
        let json = value_to_json(&Value::Int(value.clone())).unwrap();
        assert_eq!(json.to_string(), value.decimal_string());
    }
}

#[test]
fn attribute_json_roundtrips_exact_scalars_and_rejects_untyped_aggregates() {
    for value in [IntValue::I64(i64::MIN), IntValue::U64(u64::MAX)] {
        let json = value_to_json(&Value::Int(value.clone())).unwrap();
        assert_eq!(json_to_value(&json), Value::Int(value));
    }
    for element in [
        "<missing>".into(),
        "<MISSING>".into(),
        "".into(),
        runmat_value::StringElement::Missing,
    ] {
        let original = Value::String(element.clone());
        let json = value_to_json(&original).unwrap();
        assert_eq!(json_to_value(&json), original);
        let scalar_array =
            runmat_value::StringArray::from_elements_unbounded(vec![element], vec![1, 1]).unwrap();
        assert_eq!(
            value_to_json(&Value::StringArray(scalar_array)).unwrap(),
            json
        );
    }
    for number in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(value_to_json(&Value::Num(number)).is_err());
    }
    assert_eq!(
        value_to_json(&Value::Num(1.5)).unwrap(),
        serde_json::json!(1.5)
    );
    let row = runmat_value::CharArray::new_row("<missing>");
    assert_eq!(
        value_to_json(&Value::CharArray(row)).unwrap(),
        serde_json::json!("<missing>")
    );
    let matrix = runmat_value::CharArray::from_unicode_scalars(vec!['a', 'b'], 2, 1).unwrap();
    assert!(value_to_json(&Value::CharArray(matrix)).is_err());
    let pages = runmat_value::CharArray::from_code_units_with_shape(
        vec![
            runmat_value::Utf16CodeUnit::from_raw(97),
            runmat_value::Utf16CodeUnit::from_raw(98),
        ],
        vec![1, 1, 2],
    )
    .unwrap();
    assert!(value_to_json(&Value::CharArray(pages)).is_err());
    let strings = runmat_value::StringArray::from_elements_unbounded(
        vec!["literal".into(), runmat_value::StringElement::Missing],
        vec![1, 2],
    )
    .unwrap();
    assert!(value_to_json(&Value::StringArray(strings)).is_err());
}

#[test]
fn io_data_descriptors_cover_constructor_and_transaction_surface() {
    let data_labels: Vec<&str> = DATA_CREATE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(data_labels.contains(&"ds = data.create(path, schema)"));

    let read_labels: Vec<&str> = runmat_builtins::DATAARRAY_READ_CATALOG_ENTRY
        .descriptor
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(read_labels.contains(&"X = DataArray.read(arr, sliceSpec)"));

    let write_labels: Vec<&str> = runmat_builtins::DATAARRAY_WRITE_CATALOG_ENTRY
        .descriptor
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(write_labels.contains(&"tf = DataArray.write(arr, values)"));
    assert!(write_labels.contains(&"tf = DataArray.write(arr, sliceSpec, values)"));

    let tx_labels: Vec<&str> = runmat_builtins::DATATX_COMMIT_CATALOG_ENTRY
        .descriptor
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(tx_labels.contains(&"tf = DataTransaction.commit(tx, options)"));

    assert_eq!(DATA_CREATE_INTEGER_CAPABILITIES.len(), 1);
    let audited = [
        &runmat_builtins::DATAARRAY_READ_CATALOG_ENTRY,
        &runmat_builtins::DATAARRAY_WRITE_CATALOG_ENTRY,
        &runmat_builtins::DATAARRAY_RESIZE_CATALOG_ENTRY,
        &runmat_builtins::DATAARRAY_FILL_CATALOG_ENTRY,
        &runmat_builtins::DATASET_ATTRS_CATALOG_ENTRY,
        &runmat_builtins::DATASET_GET_ATTR_CATALOG_ENTRY,
        &runmat_builtins::DATASET_SET_ATTR_CATALOG_ENTRY,
        &runmat_builtins::DATASET_SET_ATTRS_CATALOG_ENTRY,
        &runmat_builtins::DATATX_WRITE_CATALOG_ENTRY,
        &runmat_builtins::DATATX_SET_ATTR_CATALOG_ENTRY,
        &runmat_builtins::DATATX_SET_ATTRS_CATALOG_ENTRY,
        &runmat_builtins::DATATX_RESIZE_CATALOG_ENTRY,
        &runmat_builtins::DATATX_FILL_CATALOG_ENTRY,
        &runmat_builtins::DATATX_CREATE_ARRAY_CATALOG_ENTRY,
    ];
    assert_eq!(
        runmat_builtins::DATASET_GET_ATTR_CATALOG_ENTRY
            .integer_capabilities
            .len(),
        2
    );
    for entry in audited {
        assert!(
            !entry.integer_capabilities.is_empty(),
            "{}",
            entry.identity.name
        );
        for capability in entry.integer_capabilities {
            assert!(capability
                .inputs
                .iter()
                .all(|input| { input.classes == runmat_builtins::ALL_INTEGER_CLASSES }));
        }
    }
}

#[test]
fn data_create_uses_exact_all_class_schema_controls_and_integer_payloads() {
    let _serial = serial_test_guard();
    let _provider = native_provider_guard();
    let cases = vec![
        (
            "int8",
            IntegerStorage::I8(vec![2, 3]),
            IntegerStorage::I8(vec![1, 3]),
            IntegerStorage::I8(vec![0; 6]),
        ),
        (
            "int16",
            IntegerStorage::I16(vec![2, 3]),
            IntegerStorage::I16(vec![1, 3]),
            IntegerStorage::I16(vec![0; 6]),
        ),
        (
            "int32",
            IntegerStorage::I32(vec![2, 3]),
            IntegerStorage::I32(vec![1, 3]),
            IntegerStorage::I32(vec![0; 6]),
        ),
        (
            "int64",
            IntegerStorage::I64(vec![2, 3]),
            IntegerStorage::I64(vec![1, 3]),
            IntegerStorage::I64(vec![0; 6]),
        ),
        (
            "uint8",
            IntegerStorage::U8(vec![2, 3]),
            IntegerStorage::U8(vec![1, 3]),
            IntegerStorage::U8(vec![0; 6]),
        ),
        (
            "uint16",
            IntegerStorage::U16(vec![2, 3]),
            IntegerStorage::U16(vec![1, 3]),
            IntegerStorage::U16(vec![0; 6]),
        ),
        (
            "uint32",
            IntegerStorage::U32(vec![2, 3]),
            IntegerStorage::U32(vec![1, 3]),
            IntegerStorage::U32(vec![0; 6]),
        ),
        (
            "uint64",
            IntegerStorage::U64(vec![2, 3]),
            IntegerStorage::U64(vec![1, 3]),
            IntegerStorage::U64(vec![0; 6]),
        ),
    ];

    for (dtype, shape, chunk, expected) in cases {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir
            .path()
            .join(format!("schema-{dtype}.data"))
            .to_string_lossy()
            .to_string();
        let mut meta = runmat_value::UnboundedStructValueBuilder::new();
        meta.fields
            .insert("dtype".to_string(), Value::String(dtype.to_string().into()));
        meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new_integer(shape, vec![1, 2]).expect("shape")),
        );
        meta.fields.insert(
            "chunk".to_string(),
            Value::Tensor(Tensor::new_integer(chunk, vec![1, 2]).expect("chunk")),
        );
        let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
        arrays
            .fields
            .insert("samples".to_string(), Value::Struct(meta.finish()));
        let mut schema = runmat_value::UnboundedStructValueBuilder::new();
        schema
            .fields
            .insert("arrays".to_string(), Value::Struct(arrays.finish()));

        let dataset = call_builtin(
            "data.create",
            &[Value::String(path.into()), Value::Struct(schema.finish())],
        )
        .unwrap_or_else(|error| panic!("{dtype}: create failed: {error}"));
        let array = call_builtin(
            "Dataset.array",
            &[
                dataset.value().clone(),
                Value::String("samples".to_string().into()),
            ],
        )
        .expect("array handle");
        let read = call_builtin("DataArray.read", std::slice::from_ref(array.value()))
            .expect("read zeros");
        let Value::Tensor(values) = read.value() else {
            panic!("{dtype}: expected tensor");
        };
        assert_eq!(values.shape(), &[2, 3], "{dtype}");
        assert_eq!(values.integer_storage(), Some(&expected), "{dtype}");
    }
}

#[test]
fn data_namespace_lifecycle_rejects_integer_arguments_and_unsupported_extras() {
    let integer_values = [
        IntValue::I8(1),
        IntValue::I16(1),
        IntValue::I32(1),
        IntValue::I64(1),
        IntValue::U8(1),
        IntValue::U16(1),
        IntValue::U32(1),
        IntValue::U64(1),
    ];
    for integer in integer_values {
        let value = Value::Int(integer);
        for (name, args) in [
            (
                "data.create",
                vec![value.clone(), Value::Struct(StructValue::new())],
            ),
            ("data.open", vec![value.clone()]),
            ("data.exists", vec![value.clone()]),
            ("data.delete", vec![value.clone()]),
            ("data.inspect", vec![value.clone()]),
            ("data.list", vec![value.clone()]),
            (
                "data.copy",
                vec![value.clone(), Value::String("to.data".to_string().into())],
            ),
            (
                "data.copy",
                vec![Value::String("from.data".to_string().into()), value.clone()],
            ),
            (
                "data.move",
                vec![value.clone(), Value::String("to.data".to_string().into())],
            ),
            (
                "data.move",
                vec![Value::String("from.data".to_string().into()), value.clone()],
            ),
            (
                "data.import",
                vec![
                    value.clone(),
                    Value::String("data".to_string().into()),
                    Value::String("source.data".to_string().into()),
                ],
            ),
            (
                "data.import",
                vec![
                    Value::String("path.data".to_string().into()),
                    value.clone(),
                    Value::String("source.data".to_string().into()),
                ],
            ),
            (
                "data.import",
                vec![
                    Value::String("path.data".to_string().into()),
                    Value::String("data".to_string().into()),
                    value.clone(),
                ],
            ),
            (
                "data.export",
                vec![
                    value.clone(),
                    Value::String("data".to_string().into()),
                    Value::String("target.data".to_string().into()),
                ],
            ),
            (
                "data.export",
                vec![
                    Value::String("path.data".to_string().into()),
                    value.clone(),
                    Value::String("target.data".to_string().into()),
                ],
            ),
            (
                "data.export",
                vec![
                    Value::String("path.data".to_string().into()),
                    Value::String("data".to_string().into()),
                    value.clone(),
                ],
            ),
            ("commit", vec![value]),
        ] {
            let error = call_builtin(name, &args).expect_err("expected integer rejection");
            assert!(!error.message().is_empty(), "{name}");
        }
    }

    let error = call_builtin(
        "data.exists",
        &[
            Value::String("missing.data".to_string().into()),
            Value::Int(IntValue::U8(1)),
        ],
    )
    .expect_err("fixed documented arity must reject extras");
    assert!(!error.message().is_empty());

    let error = call_builtin(
        "DataTransaction.commit",
        &[Value::Int(IntValue::U8(1)), Value::Int(IntValue::U8(1))],
    )
    .expect_err("commit options must be a struct");
    assert!(error.message().contains("options must be a scalar struct"));

    let error = call_builtin(
        "DataTransaction.commit",
        &[
            Value::Int(IntValue::U8(1)),
            Value::Struct(StructValue::new()),
            Value::Struct(StructValue::new()),
        ],
    )
    .expect_err("commit must reject more than one options struct");
    assert!(error.message().contains("at most one options struct"));
}

#[test]
fn dataset_nonattribute_methods_reject_integer_receivers() {
    for (name, extra) in [
        (
            "Dataset.array",
            Some(Value::String("samples".to_string().into())),
        ),
        ("Dataset.arrays", None),
        ("Dataset.begin", None),
        (
            "Dataset.has_array",
            Some(Value::String("samples".to_string().into())),
        ),
        ("Dataset.id", None),
        ("Dataset.path", None),
        ("Dataset.refresh", None),
        (
            "Dataset.snapshot",
            Some(Value::String("checkpoint".to_string().into())),
        ),
        ("Dataset.version", None),
    ] {
        let mut args = vec![Value::Int(IntValue::U64(u64::MAX))];
        args.extend(extra);
        let error = call_builtin(name, &args).expect_err("method requires a Dataset receiver");
        assert!(
            error.message().contains("expected object"),
            "{name}: {error}"
        );
    }
}

#[test]
fn dataset_attributes_keep_exact_integer_values_and_documented_class_boundaries() {
    let _serial = serial_test_guard();
    let _provider = native_provider_guard();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir
        .path()
        .join("integer-attrs.data")
        .to_string_lossy()
        .to_string();
    let mut array_meta = runmat_value::UnboundedStructValueBuilder::new();
    array_meta
        .fields
        .insert("dtype".to_string(), Value::String("f64".to_string().into()));
    array_meta.fields.insert(
        "shape".to_string(),
        Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).expect("shape")),
    );
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays
        .fields
        .insert("base".to_string(), Value::Struct(array_meta.finish()));
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".to_string(), Value::Struct(arrays.finish()));
    let ds = call_builtin(
        "data.create",
        &[Value::String(path.into()), Value::Struct(schema.finish())],
    )
    .expect("create dataset");
    let cases = [
        ("int8", IntValue::I8(i8::MIN), IntValue::I64(i8::MIN as i64)),
        (
            "int16",
            IntValue::I16(i16::MIN),
            IntValue::I64(i16::MIN as i64),
        ),
        (
            "int32",
            IntValue::I32(i32::MIN),
            IntValue::I64(i32::MIN as i64),
        ),
        ("int64", IntValue::I64(i64::MIN), IntValue::I64(i64::MIN)),
        (
            "uint8",
            IntValue::U8(u8::MAX),
            IntValue::I64(u8::MAX as i64),
        ),
        (
            "uint16",
            IntValue::U16(u16::MAX),
            IntValue::I64(u16::MAX as i64),
        ),
        (
            "uint32",
            IntValue::U32(u32::MAX),
            IntValue::I64(u32::MAX as i64),
        ),
        ("uint64", IntValue::U64(u64::MAX), IntValue::U64(u64::MAX)),
    ];

    let mut bulk = runmat_value::UnboundedStructValueBuilder::new();
    for (name, source, _) in &cases {
        call_builtin(
            "Dataset.set_attr",
            &[
                ds.value().clone(),
                Value::String(format!("direct_{name}").into()),
                Value::Int(source.clone()),
            ],
        )
        .expect("set integer attribute");
        bulk.fields
            .insert(format!("bulk_{name}"), Value::Int(source.clone()));
    }
    call_builtin(
        "Dataset.set_attrs",
        &[ds.value().clone(), Value::Struct(bulk.finish())],
    )
    .expect("set integer attributes");

    let attributes =
        call_builtin("Dataset.attrs", std::slice::from_ref(ds.value())).expect("read attributes");
    let Value::Struct(attrs) = attributes.value() else {
        panic!("expected attribute struct");
    };
    for (name, source, canonical) in cases {
        let expected = Value::Int(canonical);
        assert_eq!(
            call_builtin(
                "Dataset.get_attr",
                &[
                    ds.value().clone(),
                    Value::String(format!("direct_{name}").into())
                ],
            )
            .expect("get integer attribute"),
            expected,
            "{name} direct"
        );
        assert_eq!(
            attrs.fields.get(&format!("bulk_{name}")),
            Some(&expected),
            "{name} bulk"
        );
        assert_eq!(
            call_builtin(
                "Dataset.get_attr",
                &[
                    ds.value().clone(),
                    Value::String(format!("missing_{name}").into()),
                    Value::Int(source.clone()),
                ],
            )
            .expect("get integer default"),
            Value::Int(source),
            "{name} default"
        );
    }
}

#[test]
fn data_transaction_lifecycle_methods_reject_integer_receivers() {
    for (name, extra) in [
        ("DataTransaction.abort", None),
        ("DataTransaction.commit", None),
        (
            "DataTransaction.delete_array",
            Some(Value::String("samples".to_string().into())),
        ),
        ("DataTransaction.id", None),
        ("DataTransaction.status", None),
    ] {
        let mut args = vec![Value::Int(IntValue::U64(u64::MAX))];
        args.extend(extra);
        let error = call_builtin(name, &args)
            .expect_err("lifecycle method requires a DataTransaction receiver");
        assert!(
            error.message().contains("expected object"),
            "{name}: {error}"
        );
    }
}

#[test]
fn data_array_metadata_accessors_reject_integer_receivers() {
    for name in [
        "DataArray.chunk_shape",
        "DataArray.codec",
        "DataArray.dtype",
        "DataArray.name",
        "DataArray.rank",
        "DataArray.shape",
    ] {
        let error = call_builtin(name, &[Value::Int(IntValue::U64(u64::MAX))])
            .expect_err("metadata accessor requires a DataArray receiver");
        assert!(
            error.message().contains("expected object"),
            "{name}: {error}"
        );
    }
}

#[derive(Default)]
struct CountingDataUploadProvider {
    inner: NativeFsProvider,
    uploaded_keys: Arc<Mutex<Vec<String>>>,
    default_current_dir: Option<PathBuf>,
    swap_on_write: Mutex<Option<Arc<dyn FsProvider>>>,
    suspend_after_swap: bool,
    suspensions: Arc<Mutex<usize>>,
    reject_manifest_writes: std::sync::atomic::AtomicBool,
}

struct HttpDataUploadProvider {
    inner: NativeFsProvider,
    base_url: String,
    client: reqwest::blocking::Client,
}

impl HttpDataUploadProvider {
    fn new(base_url: String) -> Self {
        Self {
            inner: NativeFsProvider,
            base_url,
            client: reqwest::blocking::Client::new(),
        }
    }
}

#[async_trait(?Send)]
impl FsProvider for HttpDataUploadProvider {
    fn open(&self, path: &Path, flags: &OpenFlags) -> std::io::Result<Box<dyn FileHandle>> {
        self.inner.open(path, flags)
    }

    async fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        self.inner.read(path).await
    }

    async fn write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
        self.inner.write(path, data).await
    }

    async fn remove_file(&self, path: &Path) -> std::io::Result<()> {
        self.inner.remove_file(path).await
    }

    async fn metadata(&self, path: &Path) -> std::io::Result<FsMetadata> {
        self.inner.metadata(path).await
    }

    async fn symlink_metadata(&self, path: &Path) -> std::io::Result<FsMetadata> {
        self.inner.symlink_metadata(path).await
    }

    async fn read_dir(&self, path: &Path) -> std::io::Result<Vec<DirEntry>> {
        self.inner.read_dir(path).await
    }

    async fn canonicalize(&self, path: &Path) -> std::io::Result<std::path::PathBuf> {
        self.inner.canonicalize(path).await
    }

    async fn create_dir(&self, path: &Path) -> std::io::Result<()> {
        self.inner.create_dir(path).await
    }

    async fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
        self.inner.create_dir_all(path).await
    }

    async fn remove_dir(&self, path: &Path) -> std::io::Result<()> {
        self.inner.remove_dir(path).await
    }

    async fn remove_dir_all(&self, path: &Path) -> std::io::Result<()> {
        self.inner.remove_dir_all(path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        self.inner.rename(from, to).await
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> std::io::Result<()> {
        self.inner.set_readonly(path, readonly).await
    }

    async fn data_manifest_descriptor(
        &self,
        request: &DataManifestRequest,
    ) -> std::io::Result<DataManifestDescriptor> {
        self.inner.data_manifest_descriptor(request).await
    }

    async fn data_chunk_upload_targets(
        &self,
        request: &DataChunkUploadRequest,
    ) -> std::io::Result<Vec<DataChunkUploadTarget>> {
        #[derive(Deserialize)]
        struct UploadTargetsResponse {
            targets: Vec<DataChunkUploadTarget>,
        }
        let url = format!("{}/data/chunks/upload-targets", self.base_url);
        let response = self
            .client
            .post(url)
            .json(request)
            .send()
            .map_err(|err| std::io::Error::other(err.to_string()))?;
        if !response.status().is_success() {
            return Err(std::io::Error::other(format!(
                "upload targets request failed: {}",
                response.status()
            )));
        }
        let parsed: UploadTargetsResponse = response
            .json()
            .map_err(|err| std::io::Error::other(err.to_string()))?;
        Ok(parsed.targets)
    }

    async fn data_upload_chunk(
        &self,
        target: &DataChunkUploadTarget,
        data: &[u8],
    ) -> std::io::Result<()> {
        let upload_url = if let Some(key) = target.upload_url.strip_prefix("upload://") {
            format!("{}/upload?key={}", self.base_url, key)
        } else {
            target.upload_url.clone()
        };
        let method = reqwest::Method::from_bytes(target.method.as_bytes())
            .map_err(|err| std::io::Error::other(err.to_string()))?;
        let mut request = self.client.request(method, &upload_url);
        for (k, v) in &target.headers {
            request = request.header(k, v);
        }
        let response = request
            .body(data.to_vec())
            .send()
            .map_err(|err| std::io::Error::other(err.to_string()))?;
        if !response.status().is_success() {
            return Err(std::io::Error::other(format!(
                "chunk upload failed: {}",
                response.status()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
struct UploadHarness {
    uploads: Arc<Mutex<Vec<String>>>,
}

#[derive(Deserialize)]
struct UploadChunkQuery {
    key: String,
}

async fn upload_targets_handler(
    Json(req): Json<DataChunkUploadRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let targets = req
        .chunks
        .iter()
        .map(|chunk| {
            serde_json::json!({
                "key": chunk.key,
                "method": "PUT",
                "upload_url": format!("upload://{}", chunk.key),
                "headers": {
                    "x-runmat-hash": chunk.hash,
                }
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(serde_json::json!({ "targets": targets })))
}

async fn upload_handler(
    State(harness): State<UploadHarness>,
    Query(query): Query<UploadChunkQuery>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<(), StatusCode> {
    if body.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    if headers.get("x-runmat-hash").is_none() {
        return Err(StatusCode::BAD_REQUEST);
    }
    let mut guard = harness.uploads.lock().expect("uploads lock poisoned");
    guard.push(query.key);
    Ok(())
}

fn spawn_upload_server() -> (
    String,
    Arc<Mutex<Vec<String>>>,
    Runtime,
    oneshot::Sender<()>,
) {
    let harness = UploadHarness::default();
    let uploads = Arc::clone(&harness.uploads);
    let runtime = Runtime::new().expect("tokio runtime");
    let (addr, shutdown_tx) = runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind upload server");
        let addr = listener.local_addr().expect("local addr");
        let app = Router::new()
            .route("/data/chunks/upload-targets", post(upload_targets_handler))
            .route("/upload", put(upload_handler))
            .with_state(harness);
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let server = axum::serve(listener, app).with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        });
        tokio::spawn(async move {
            let _ = server.await;
        });
        (addr, shutdown_tx)
    });
    (format!("http://{}", addr), uploads, runtime, shutdown_tx)
}

impl CountingDataUploadProvider {
    fn uploaded_keys(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.uploaded_keys)
    }
}

#[async_trait(?Send)]
impl FsProvider for CountingDataUploadProvider {
    fn current_dir_override(&self) -> Option<PathBuf> {
        self.default_current_dir.clone()
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> std::io::Result<Box<dyn FileHandle>> {
        self.inner.open(path, flags)
    }

    async fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        self.inner.read(path).await
    }

    async fn write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
        let replacement = self.swap_on_write.lock().expect("swap lock").take();
        if path.file_name().is_some_and(|name| name == "manifest.json")
            && self
                .reject_manifest_writes
                .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "manifest head changed",
            ));
        }
        if let Some(replacement) = replacement {
            runmat_filesystem::set_provider(replacement);
            runmat_filesystem::set_current_dir_async("/replacement")
                .await
                .expect("replacement fixture working directory");
            assert_eq!(
                runmat_filesystem::current_dir().unwrap(),
                Path::new("/replacement")
            );
            if self.suspend_after_swap {
                let mut yielded = false;
                futures::future::poll_fn(|cx| {
                    if yielded {
                        std::task::Poll::Ready(())
                    } else {
                        yielded = true;
                        *self.suspensions.lock().expect("suspension count") += 1;
                        cx.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                })
                .await;
            }
        }
        self.inner.write(path, data).await
    }

    async fn remove_file(&self, path: &Path) -> std::io::Result<()> {
        self.inner.remove_file(path).await
    }

    async fn metadata(&self, path: &Path) -> std::io::Result<FsMetadata> {
        self.inner.metadata(path).await
    }

    async fn symlink_metadata(&self, path: &Path) -> std::io::Result<FsMetadata> {
        self.inner.symlink_metadata(path).await
    }

    async fn read_dir(&self, path: &Path) -> std::io::Result<Vec<DirEntry>> {
        self.inner.read_dir(path).await
    }

    async fn canonicalize(&self, path: &Path) -> std::io::Result<std::path::PathBuf> {
        self.inner.canonicalize(path).await
    }

    async fn create_dir(&self, path: &Path) -> std::io::Result<()> {
        self.inner.create_dir(path).await
    }

    async fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
        self.inner.create_dir_all(path).await
    }

    async fn remove_dir(&self, path: &Path) -> std::io::Result<()> {
        self.inner.remove_dir(path).await
    }

    async fn remove_dir_all(&self, path: &Path) -> std::io::Result<()> {
        self.inner.remove_dir_all(path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        self.inner.rename(from, to).await
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> std::io::Result<()> {
        self.inner.set_readonly(path, readonly).await
    }

    async fn data_manifest_descriptor(
        &self,
        request: &DataManifestRequest,
    ) -> std::io::Result<DataManifestDescriptor> {
        self.inner.data_manifest_descriptor(request).await
    }

    async fn data_chunk_upload_targets(
        &self,
        request: &DataChunkUploadRequest,
    ) -> std::io::Result<Vec<DataChunkUploadTarget>> {
        Ok(request
            .chunks
            .iter()
            .map(|chunk| DataChunkUploadTarget {
                key: chunk.key.clone(),
                method: "PUT".to_string(),
                upload_url: format!("count://{}", chunk.object_id),
                headers: std::collections::HashMap::new(),
            })
            .collect())
    }

    async fn data_upload_chunk(
        &self,
        target: &DataChunkUploadTarget,
        _data: &[u8],
    ) -> std::io::Result<()> {
        let mut guard = match self.uploaded_keys.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.push(target.key.clone());
        Ok(())
    }
}

#[test]
fn data_create_retains_selected_provider_and_relative_cwd_after_first_write() {
    let _serial = serial_test_guard();
    let dir = tempfile::tempdir().expect("selected data directory");
    let selected = Arc::new(CountingDataUploadProvider {
        default_current_dir: Some(dir.path().to_path_buf()),
        suspend_after_swap: true,
        ..Default::default()
    });
    let suspensions = selected.suspensions.clone();
    let replacement = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/replacement",
    ));
    *selected.swap_on_write.lock().expect("swap lock") =
        Some(replacement.clone() as Arc<dyn FsProvider>);
    let _guard = runmat_filesystem::replace_provider(selected);
    runmat_filesystem::set_current_dir(dir.path()).expect("selected fixture directory");

    let mut array = runmat_value::UnboundedStructValueBuilder::new();
    array
        .fields
        .insert("dtype".into(), Value::String("uint64".into()));
    array.fields.insert(
        "shape".into(),
        Value::Tensor(
            Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![1, 1]), vec![1, 2])
                .expect("shape"),
        ),
    );
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays
        .fields
        .insert("sample".into(), Value::Struct(array.finish()));
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".into(), Value::Struct(arrays.finish()));

    let result = call_builtin(
        "data.create",
        &[
            Value::String("pinned.data".into()),
            Value::Struct(schema.finish()),
        ],
    )
    .expect("create finishes on original authority after ambient swap");
    assert!(matches!(result.value(), Value::Object(_)));
    assert_eq!(
        call_builtin("Dataset.path", std::slice::from_ref(result.value()))
            .expect("sealed public path"),
        Value::String("pinned.data".into())
    );
    for method in ["Dataset.id", "Dataset.version"] {
        let identity = call_builtin(method, std::slice::from_ref(result.value()))
            .expect("sealed Dataset identity accessor");
        let Value::String(runmat_value::StringElement::Text(value)) = identity.value() else {
            panic!("{method} must return text");
        };
        assert!(!value.is_empty());
    }
    assert_eq!(
        call_builtin(
            "Dataset.has_array",
            &[result.value().clone(), Value::String("sample".into())]
        )
        .expect("pinned membership"),
        Value::Bool(true)
    );
    assert_eq!(
        call_builtin(
            "Dataset.has_array",
            &[result.value().clone(), Value::String("other".into())]
        )
        .expect("pinned non-membership"),
        Value::Bool(false)
    );
    let arrays = call_builtin("Dataset.arrays", std::slice::from_ref(result.value()))
        .expect("follow-on Dataset read retains selected provider");
    let Value::Cell(arrays) = arrays.value() else {
        panic!("Dataset.arrays must return a cell");
    };
    assert_eq!(arrays.data(), vec![Value::String("sample".into())]);
    let array = call_builtin(
        "Dataset.array",
        &[result.value().clone(), Value::String("sample".into())],
    )
    .expect("child array inherits selected provider");
    let dtype = call_builtin("DataArray.dtype", std::slice::from_ref(array.value()))
        .expect("array metadata retains selected provider");
    assert_eq!(dtype, Value::String("uint64".into()));
    assert_eq!(
        call_builtin("DataArray.name", std::slice::from_ref(array.value()))
            .expect("sealed array name"),
        Value::String("sample".into())
    );
    assert_eq!(
        call_builtin("DataArray.codec", std::slice::from_ref(array.value()))
            .expect("pinned codec metadata"),
        Value::String("zstd".into())
    );
    assert_eq!(
        call_builtin("DataArray.rank", std::slice::from_ref(array.value()))
            .expect("pinned rank metadata"),
        Value::Num(2.0)
    );
    for method in ["DataArray.shape", "DataArray.chunk_shape"] {
        let shape_result = call_builtin(method, std::slice::from_ref(array.value()))
            .expect("pinned shape metadata");
        let Value::Tensor(shape) = shape_result.value() else {
            panic!("{method} must return a tensor");
        };
        assert_eq!(shape.materialize_f64(), vec![1.0, 1.0]);
    }
    let values = call_builtin("DataArray.read", std::slice::from_ref(array.value()))
        .expect("array payload retains selected provider");
    let Value::Tensor(values) = values.value() else {
        panic!("DataArray.read must return a tensor");
    };
    assert_eq!(
        values.integer_storage(),
        Some(&runmat_value::IntegerStorage::U64(vec![0]))
    );
    call_builtin(
        "DataArray.write",
        &[
            array.value().clone(),
            Value::Tensor(
                Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![42]), vec![1, 1])
                    .expect("replacement value"),
            ),
        ],
    )
    .expect("array write retains selected provider");
    let written_result = call_builtin("DataArray.read", std::slice::from_ref(array.value()))
        .expect("read after pinned write");
    let Value::Tensor(written) = written_result.value() else {
        panic!("DataArray.read must return a tensor");
    };
    assert_eq!(
        written.integer_storage(),
        Some(&runmat_value::IntegerStorage::U64(vec![42]))
    );
    call_builtin(
        "DataArray.fill",
        &[array.value().clone(), Value::Int(IntValue::U64(9))],
    )
    .expect("fill stays on original provider");
    let filled_result = call_builtin("DataArray.read", std::slice::from_ref(array.value()))
        .expect("read after pinned fill");
    let Value::Tensor(filled) = filled_result.value() else {
        panic!("DataArray.read must return a tensor");
    };
    assert_eq!(
        filled.integer_storage(),
        Some(&runmat_value::IntegerStorage::U64(vec![9]))
    );
    call_builtin(
        "DataArray.resize",
        &[
            array.value().clone(),
            Value::Tensor(
                Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![1, 2]), vec![1, 2])
                    .expect("resize shape"),
            ),
        ],
    )
    .expect("resize stays on original provider");
    let resized_result = call_builtin("DataArray.read", std::slice::from_ref(array.value()))
        .expect("read after pinned resize");
    let Value::Tensor(resized) = resized_result.value() else {
        panic!("DataArray.read must return a tensor");
    };
    assert_eq!(resized.shape(), &[1, 2]);
    assert_eq!(
        resized.integer_storage(),
        Some(&runmat_value::IntegerStorage::U64(vec![0, 0]))
    );
    let tx = call_builtin("Dataset.begin", std::slice::from_ref(result.value()))
        .expect("transaction inherits selected provider");
    call_builtin(
        "DataTransaction.set_attr",
        &[
            tx.value().clone(),
            Value::String("owner".into()),
            Value::String("selected".into()),
        ],
    )
    .expect("enqueue transaction attribute");
    call_builtin("DataTransaction.commit", std::slice::from_ref(tx.value()))
        .expect("commit retains selected provider");
    assert_eq!(
        call_builtin(
            "Dataset.get_attr",
            &[result.value().clone(), Value::String("owner".into())],
        )
        .expect("read committed attribute through selected provider"),
        Value::String("selected".into())
    );
    let attributes = call_builtin("Dataset.attrs", std::slice::from_ref(result.value()))
        .expect("attributes stay on original provider");
    let Value::Struct(attrs) = attributes.value() else {
        panic!("Dataset.attrs must return a struct");
    };
    assert_eq!(
        attrs.fields.get("owner"),
        Some(&Value::String("selected".into()))
    );
    let refreshed = call_builtin("Dataset.refresh", std::slice::from_ref(result.value()))
        .expect("refresh stays on original provider");
    assert_ne!(
        call_builtin("Dataset.version", std::slice::from_ref(result.value())).unwrap(),
        call_builtin("Dataset.version", std::slice::from_ref(refreshed.value())).unwrap(),
        "old handle retains its captured version"
    );
    let snapshot = call_builtin(
        "Dataset.snapshot",
        &[result.value().clone(), Value::String("owner-check".into())],
    )
    .expect("snapshot stays on original provider");
    let Value::String(runmat_value::StringElement::Text(snapshot)) = snapshot.value() else {
        panic!("Dataset.snapshot must return a path");
    };
    assert!(std::path::Path::new(snapshot.as_str()).is_file());
    assert_eq!(*suspensions.lock().expect("suspension count"), 1);
    let root = dir.path().join("pinned.data");
    assert!(root.join("manifest.json").is_file());
    let manifest: crate::data::DataManifest =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let meta = &manifest.arrays["sample"];
    assert!(root.join(&meta.data_path).is_file());
    let index: DataChunkIndex = serde_json::from_slice(
        &std::fs::read(root.join(meta.chunk_index_path.as_ref().unwrap())).unwrap(),
    )
    .unwrap();
    assert!(index
        .chunks
        .iter()
        .all(|chunk| root.join(&chunk.data_path).is_file()));
    assert!(
        futures::executor::block_on(replacement.read(std::path::Path::new(
            "/replacement/pinned.data/manifest.json"
        )))
        .is_err()
    );
}

#[test]
fn create_open_write_read_dataset() {
    let _serial = serial_test_guard();
    let _provider_guard = native_provider_guard();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("sample.data").to_string_lossy().to_string();

    let mut array_meta = runmat_value::UnboundedStructValueBuilder::new();
    array_meta
        .fields
        .insert("dtype".to_string(), Value::String("f64".to_string().into()));
    array_meta.fields.insert(
        "shape".to_string(),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("shape tensor")),
    );
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays.fields.insert(
        "temperature".to_string(),
        Value::Struct(array_meta.finish()),
    );
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".to_string(), Value::Struct(arrays.finish()));

    let ds = call_builtin(
        "data.create",
        &[
            Value::String(path.clone().into()),
            Value::Struct(schema.finish()),
        ],
    )
    .expect("create dataset");

    let arr = call_builtin(
        "Dataset.array",
        &[
            ds.value().clone(),
            Value::String("temperature".to_string().into()),
        ],
    )
    .expect("dataset array");
    let write_tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("write tensor");
    call_builtin(
        "DataArray.write",
        &[arr.value().clone(), Value::Tensor(write_tensor)],
    )
    .expect("write array");

    let read_back =
        call_builtin("DataArray.read", std::slice::from_ref(arr.value())).expect("read array");
    let Value::Tensor(t) = read_back.value() else {
        panic!("expected tensor");
    };
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(tensor_values(t), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn write_and_read_slice_payload() {
    let _serial = serial_test_guard();
    let _provider_guard = native_provider_guard();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("slice.data").to_string_lossy().to_string();

    let mut array_meta = runmat_value::UnboundedStructValueBuilder::new();
    array_meta
        .fields
        .insert("dtype".to_string(), Value::String("f64".to_string().into()));
    array_meta.fields.insert(
        "shape".to_string(),
        Value::Tensor(Tensor::new(vec![3.0, 3.0], vec![1, 2]).expect("shape tensor")),
    );
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays.fields.insert(
        "temperature".to_string(),
        Value::Struct(array_meta.finish()),
    );
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".to_string(), Value::Struct(arrays.finish()));

    let ds = call_builtin(
        "data.create",
        &[
            Value::String(path.clone().into()),
            Value::Struct(schema.finish()),
        ],
    )
    .expect("create dataset");
    let arr = call_builtin(
        "Dataset.array",
        &[
            ds.value().clone(),
            Value::String("temperature".to_string().into()),
        ],
    )
    .expect("dataset array");

    let slice = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("range")),
                Value::String(":".to_string().into()),
            ],
            1,
            2,
        )
        .expect("slice cell"),
    );
    let rhs = Value::Tensor(
        Tensor::new(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0], vec![2, 3]).expect("rhs"),
    );
    call_builtin(
        "DataArray.write",
        &[arr.value().clone(), slice.clone(), rhs],
    )
    .expect("slice write");

    let read_back =
        call_builtin("DataArray.read", &[arr.value().clone(), slice]).expect("slice read");
    let Value::Tensor(t) = read_back.value() else {
        panic!("expected tensor");
    };
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(tensor_values(t), vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
}

#[test]
fn slice_write_updates_only_touched_chunks() {
    let _serial = serial_test_guard();
    let _provider_guard = native_provider_guard();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir
        .path()
        .join("chunked.data")
        .to_string_lossy()
        .to_string();

    let mut array_meta = runmat_value::UnboundedStructValueBuilder::new();
    array_meta
        .fields
        .insert("dtype".to_string(), Value::String("f64".to_string().into()));
    array_meta.fields.insert(
        "shape".to_string(),
        Value::Tensor(Tensor::new(vec![4.0, 4.0], vec![1, 2]).expect("shape tensor")),
    );
    array_meta.fields.insert(
        "chunk".to_string(),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("chunk tensor")),
    );
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays.fields.insert(
        "temperature".to_string(),
        Value::Struct(array_meta.finish()),
    );
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".to_string(), Value::Struct(arrays.finish()));

    let ds = call_builtin(
        "data.create",
        &[
            Value::String(path.clone().into()),
            Value::Struct(schema.finish()),
        ],
    )
    .expect("create dataset");
    let arr = call_builtin(
        "Dataset.array",
        &[
            ds.value().clone(),
            Value::String("temperature".to_string().into()),
        ],
    )
    .expect("dataset array");

    let full = Value::Tensor(
        Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4]).expect("full tensor"),
    );
    call_builtin("DataArray.write", &[arr.value().clone(), full]).expect("initial write");

    let root = std::path::PathBuf::from(&path);
    let read_index = || {
        let manifest =
            futures::executor::block_on(crate::data::read_manifest_async(&root)).unwrap();
        let path = root.join(
            manifest.arrays["temperature"]
                .chunk_index_path
                .as_ref()
                .unwrap(),
        );
        serde_json::from_slice::<DataChunkIndex>(
            &futures::executor::block_on(runmat_filesystem::read_async(path)).unwrap(),
        )
        .unwrap()
    };
    let before_index = read_index();
    let untouched_path = root.join(
        &before_index
            .chunks
            .iter()
            .find(|c| c.key == "1.1")
            .unwrap()
            .data_path,
    );
    let touched_path = root.join(
        &before_index
            .chunks
            .iter()
            .find(|c| c.key == "0.0")
            .unwrap()
            .data_path,
    );
    let untouched_before =
        futures::executor::block_on(runmat_filesystem::read_async(&untouched_path))
            .expect("read untouched before");
    let touched_before = futures::executor::block_on(runmat_filesystem::read_async(&touched_path))
        .expect("read touched before");

    let slice = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("range")),
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("range")),
            ],
            1,
            2,
        )
        .expect("slice cell"),
    );
    let rhs = Value::Tensor(Tensor::new(vec![99.0, 98.0, 97.0, 96.0], vec![2, 2]).expect("rhs"));
    call_builtin("DataArray.write", &[arr.value().clone(), slice, rhs]).expect("slice write");

    let untouched_after =
        futures::executor::block_on(runmat_filesystem::read_async(&untouched_path))
            .expect("read untouched after");
    let after_index = read_index();
    let next_untouched = root.join(
        &after_index
            .chunks
            .iter()
            .find(|c| c.key == "1.1")
            .unwrap()
            .data_path,
    );
    let next_touched = root.join(
        &after_index
            .chunks
            .iter()
            .find(|c| c.key == "0.0")
            .unwrap()
            .data_path,
    );
    assert_eq!(next_untouched, untouched_path);
    assert_ne!(next_touched, touched_path);
    assert_eq!(
        touched_before,
        futures::executor::block_on(runmat_filesystem::read_async(&touched_path)).unwrap()
    );
    let touched_after = futures::executor::block_on(runmat_filesystem::read_async(&next_touched))
        .expect("read touched after");
    assert_eq!(untouched_before, untouched_after);
    assert_ne!(touched_before, touched_after);
}

#[test]
fn slice_write_uploads_only_touched_chunk_targets() {
    let _serial = serial_test_guard();
    let provider = Arc::new(CountingDataUploadProvider::default());
    let uploaded = provider.uploaded_keys();
    let _guard = runmat_filesystem::replace_provider(provider);

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir
        .path()
        .join("remote-chunked.data")
        .to_string_lossy()
        .to_string();

    let mut array_meta = runmat_value::UnboundedStructValueBuilder::new();
    array_meta
        .fields
        .insert("dtype".to_string(), Value::String("f64".to_string().into()));
    array_meta.fields.insert(
        "shape".to_string(),
        Value::Tensor(Tensor::new(vec![4.0, 4.0], vec![1, 2]).expect("shape tensor")),
    );
    array_meta.fields.insert(
        "chunk".to_string(),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("chunk tensor")),
    );
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays.fields.insert(
        "temperature".to_string(),
        Value::Struct(array_meta.finish()),
    );
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".to_string(), Value::Struct(arrays.finish()));

    let ds = call_builtin(
        "data.create",
        &[
            Value::String(path.clone().into()),
            Value::Struct(schema.finish()),
        ],
    )
    .expect("create dataset");
    let arr = call_builtin(
        "Dataset.array",
        &[
            ds.value().clone(),
            Value::String("temperature".to_string().into()),
        ],
    )
    .expect("dataset array");

    call_builtin(
        "DataArray.write",
        &[
            arr.value().clone(),
            Value::Tensor(
                Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4]).expect("full tensor"),
            ),
        ],
    )
    .expect("initial write");

    let manifest =
        futures::executor::block_on(crate::data::read_manifest_async(&dataset_root(&path)))
            .expect("manifest after initial write");
    let meta = manifest
        .arrays
        .get("temperature")
        .expect("temperature meta");
    let chunk_index_path =
        dataset_root(&path).join(meta.chunk_index_path.clone().expect("chunk index path"));
    assert!(
        futures::executor::block_on(runmat_filesystem::metadata_async(&chunk_index_path)).is_ok()
    );

    {
        let mut keys = uploaded.lock().expect("uploaded keys lock");
        keys.clear();
    }

    let slice = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("range")),
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("range")),
            ],
            1,
            2,
        )
        .expect("slice cell"),
    );
    let rhs = Value::Tensor(Tensor::new(vec![9.0, 8.0, 7.0, 6.0], vec![2, 2]).expect("rhs"));
    call_builtin("DataArray.write", &[arr.value().clone(), slice, rhs]).expect("slice write");

    let keys = uploaded.lock().expect("uploaded keys lock");
    assert_eq!(keys.as_slice(), ["0.0".to_string()].as_slice());
}

#[test]
fn slice_write_uploads_expected_cross_boundary_chunk_targets() {
    let _serial = serial_test_guard();
    let provider = Arc::new(CountingDataUploadProvider::default());
    let uploaded = provider.uploaded_keys();
    let _guard = runmat_filesystem::replace_provider(provider);

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir
        .path()
        .join("remote-chunked-boundary.data")
        .to_string_lossy()
        .to_string();

    let mut array_meta = runmat_value::UnboundedStructValueBuilder::new();
    array_meta
        .fields
        .insert("dtype".to_string(), Value::String("f64".to_string().into()));
    array_meta.fields.insert(
        "shape".to_string(),
        Value::Tensor(Tensor::new(vec![4.0, 4.0], vec![1, 2]).expect("shape tensor")),
    );
    array_meta.fields.insert(
        "chunk".to_string(),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("chunk tensor")),
    );
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays.fields.insert(
        "temperature".to_string(),
        Value::Struct(array_meta.finish()),
    );
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".to_string(), Value::Struct(arrays.finish()));

    let ds = call_builtin(
        "data.create",
        &[
            Value::String(path.clone().into()),
            Value::Struct(schema.finish()),
        ],
    )
    .expect("create dataset");
    let arr = call_builtin(
        "Dataset.array",
        &[
            ds.value().clone(),
            Value::String("temperature".to_string().into()),
        ],
    )
    .expect("dataset array");

    call_builtin(
        "DataArray.write",
        &[
            arr.value().clone(),
            Value::Tensor(
                Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4]).expect("full tensor"),
            ),
        ],
    )
    .expect("initial write");
    {
        let mut keys = uploaded.lock().expect("uploaded keys lock");
        keys.clear();
    }

    let slice = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("range")),
                Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("range")),
            ],
            1,
            2,
        )
        .expect("slice cell"),
    );
    let rhs = Value::Tensor(Tensor::new(vec![19.0, 18.0, 17.0, 16.0], vec![2, 2]).expect("rhs"));
    call_builtin("DataArray.write", &[arr.value().clone(), slice, rhs]).expect("slice write");

    let mut keys = uploaded.lock().expect("uploaded keys lock").clone();
    keys.sort();
    keys.dedup();
    assert_eq!(
        keys.as_slice(),
        [
            "0.0".to_string(),
            "0.1".to_string(),
            "1.0".to_string(),
            "1.1".to_string(),
        ]
        .as_slice()
    );
}

#[test]
fn slice_write_hits_http_server_data_endpoints_with_expected_keys() {
    let _serial = serial_test_guard();
    let (base_url, uploads, runtime, shutdown_tx) = spawn_upload_server();
    let provider = Arc::new(HttpDataUploadProvider::new(base_url));
    let _guard = runmat_filesystem::replace_provider(provider);

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir
        .path()
        .join("http-endpoint.data")
        .to_string_lossy()
        .to_string();

    let mut array_meta = runmat_value::UnboundedStructValueBuilder::new();
    array_meta
        .fields
        .insert("dtype".to_string(), Value::String("f64".to_string().into()));
    array_meta.fields.insert(
        "shape".to_string(),
        Value::Tensor(Tensor::new(vec![4.0, 4.0], vec![1, 2]).expect("shape tensor")),
    );
    array_meta.fields.insert(
        "chunk".to_string(),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("chunk tensor")),
    );
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays.fields.insert(
        "temperature".to_string(),
        Value::Struct(array_meta.finish()),
    );
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".to_string(), Value::Struct(arrays.finish()));

    let ds = call_builtin(
        "data.create",
        &[
            Value::String(path.clone().into()),
            Value::Struct(schema.finish()),
        ],
    )
    .expect("create dataset");
    let arr = call_builtin(
        "Dataset.array",
        &[
            ds.value().clone(),
            Value::String("temperature".to_string().into()),
        ],
    )
    .expect("dataset array");

    call_builtin(
        "DataArray.write",
        &[
            arr.value().clone(),
            Value::Tensor(
                Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4]).expect("full tensor"),
            ),
        ],
    )
    .expect("initial write");

    {
        let mut keys = uploads.lock().expect("uploads lock");
        keys.clear();
    }

    let slice = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("range")),
                Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("range")),
            ],
            1,
            2,
        )
        .expect("slice cell"),
    );
    let rhs = Value::Tensor(Tensor::new(vec![19.0, 18.0, 17.0, 16.0], vec![2, 2]).expect("rhs"));
    call_builtin("DataArray.write", &[arr.value().clone(), slice, rhs]).expect("slice write");

    let mut keys = uploads.lock().expect("uploads lock").clone();
    keys.sort();
    keys.dedup();
    assert_eq!(
        keys.as_slice(),
        [
            "0.0".to_string(),
            "0.1".to_string(),
            "1.0".to_string(),
            "1.1".to_string(),
        ]
        .as_slice()
    );

    let _ = shutdown_tx.send(());
    drop(runtime);
}

#[test]
fn tx_create_resize_fill_and_delete_array() {
    let _serial = serial_test_guard();
    let _provider = native_provider_guard();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tx-ops.data").to_string_lossy().to_string();

    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    let mut array_meta = runmat_value::UnboundedStructValueBuilder::new();
    array_meta
        .fields
        .insert("dtype".to_string(), Value::String("f64".to_string().into()));
    array_meta.fields.insert(
        "shape".to_string(),
        Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).expect("shape tensor")),
    );
    arrays
        .fields
        .insert("base".to_string(), Value::Struct(array_meta.finish()));
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".to_string(), Value::Struct(arrays.finish()));

    let ds = call_builtin(
        "data.create",
        &[
            Value::String(path.clone().into()),
            Value::Struct(schema.finish()),
        ],
    )
    .expect("create dataset");

    let tx = call_builtin("Dataset.begin", std::slice::from_ref(ds.value())).expect("begin tx");
    let mut new_meta = runmat_value::UnboundedStructValueBuilder::new();
    new_meta
        .fields
        .insert("dtype".to_string(), Value::String("f64".to_string().into()));
    new_meta.fields.insert(
        "shape".to_string(),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("shape tensor")),
    );
    call_builtin(
        "DataTransaction.create_array",
        &[
            tx.value().clone(),
            Value::String("new_array".to_string().into()),
            Value::Struct(new_meta.finish()),
        ],
    )
    .expect("create array in tx");
    call_builtin(
        "DataTransaction.resize",
        &[
            tx.value().clone(),
            Value::String("new_array".to_string().into()),
            Value::Tensor(Tensor::new(vec![3.0, 1.0], vec![1, 2]).expect("shape tensor")),
        ],
    )
    .expect("resize array in tx");
    call_builtin(
        "DataTransaction.fill",
        &[
            tx.value().clone(),
            Value::String("new_array".to_string().into()),
            Value::Num(7.0),
        ],
    )
    .expect("fill array in tx");
    call_builtin(
        "DataTransaction.delete_array",
        &[tx.value().clone(), Value::String("base".to_string().into())],
    )
    .expect("delete array in tx");
    call_builtin(
        "commit",
        &[tx.value().clone(), Value::Struct(StructValue::new())],
    )
    .expect("commit tx through alias options form");

    let ds = call_builtin("data.open", &[Value::String(path.into())]).expect("open dataset");
    let has_base = call_builtin(
        "Dataset.has_array",
        &[ds.value().clone(), Value::String("base".to_string().into())],
    )
    .expect("has base");
    assert_eq!(has_base, Value::Bool(false));
    let arr = call_builtin(
        "Dataset.array",
        &[
            ds.value().clone(),
            Value::String("new_array".to_string().into()),
        ],
    )
    .expect("new array");
    let read_back =
        call_builtin("DataArray.read", std::slice::from_ref(arr.value())).expect("read array");
    let Value::Tensor(t) = read_back.value() else {
        panic!("expected tensor");
    };
    assert_eq!(t.shape(), &[3, 1]);
    assert_eq!(tensor_values(t), vec![7.0, 7.0, 7.0]);
}

#[test]
fn data_transactions_preserve_every_integer_class_and_typed_controls() {
    let _serial = serial_test_guard();
    let _provider = native_provider_guard();
    let cases = vec![
        (
            "int8",
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I8(vec![2, 1]),
            IntegerStorage::I8(vec![1, 1]),
            IntegerStorage::I8(vec![1, 2]),
            IntValue::I8(1),
            IntValue::I8(i8::MIN),
        ),
        (
            "int16",
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I16(vec![2, 1]),
            IntegerStorage::I16(vec![1, 1]),
            IntegerStorage::I16(vec![1, 2]),
            IntValue::I16(1),
            IntValue::I16(i16::MIN),
        ),
        (
            "int32",
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I32(vec![2, 1]),
            IntegerStorage::I32(vec![1, 1]),
            IntegerStorage::I32(vec![1, 2]),
            IntValue::I32(1),
            IntValue::I32(i32::MIN),
        ),
        (
            "int64",
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::I64(vec![2, 1]),
            IntegerStorage::I64(vec![1, 1]),
            IntegerStorage::I64(vec![1, 2]),
            IntValue::I64(1),
            IntValue::I64(i64::MIN),
        ),
        (
            "uint8",
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U8(vec![2, 1]),
            IntegerStorage::U8(vec![1, 1]),
            IntegerStorage::U8(vec![1, 2]),
            IntValue::U8(1),
            IntValue::U8(u8::MAX),
        ),
        (
            "uint16",
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U16(vec![2, 1]),
            IntegerStorage::U16(vec![1, 1]),
            IntegerStorage::U16(vec![1, 2]),
            IntValue::U16(1),
            IntValue::U16(u16::MAX),
        ),
        (
            "uint32",
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U32(vec![2, 1]),
            IntegerStorage::U32(vec![1, 1]),
            IntegerStorage::U32(vec![1, 2]),
            IntValue::U32(1),
            IntValue::U32(u32::MAX),
        ),
        (
            "uint64",
            IntegerStorage::U64(vec![0, u64::MAX]),
            IntegerStorage::U64(vec![2, 1]),
            IntegerStorage::U64(vec![1, 1]),
            IntegerStorage::U64(vec![1, 2]),
            IntValue::U64(1),
            IntValue::U64(u64::MAX),
        ),
    ];

    for (dtype, values, shape, chunk, range, one, attribute) in cases {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir
            .path()
            .join(format!("tx-{dtype}.data"))
            .to_string_lossy()
            .to_string();
        let mut base_meta = runmat_value::UnboundedStructValueBuilder::new();
        base_meta
            .fields
            .insert("dtype".to_string(), Value::String("f64".to_string().into()));
        base_meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).expect("base shape")),
        );
        let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
        arrays
            .fields
            .insert("base".to_string(), Value::Struct(base_meta.finish()));
        let mut schema = runmat_value::UnboundedStructValueBuilder::new();
        schema
            .fields
            .insert("arrays".to_string(), Value::Struct(arrays.finish()));
        let ds = call_builtin(
            "data.create",
            &[Value::String(path.into()), Value::Struct(schema.finish())],
        )
        .expect("create dataset");
        let tx = call_builtin("Dataset.begin", std::slice::from_ref(ds.value()))
            .expect("begin transaction");

        let mut meta = runmat_value::UnboundedStructValueBuilder::new();
        meta.fields
            .insert("dtype".to_string(), Value::String(dtype.to_string().into()));
        meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new_integer(shape.clone(), vec![1, 2]).expect("shape")),
        );
        meta.fields.insert(
            "chunk".to_string(),
            Value::Tensor(Tensor::new_integer(chunk, vec![1, 2]).expect("chunk")),
        );
        call_builtin(
            "DataTransaction.create_array",
            &[
                tx.value().clone(),
                Value::String("typed".to_string().into()),
                Value::Struct(meta.finish()),
            ],
        )
        .expect("queue create");
        call_builtin(
            "DataTransaction.resize",
            &[
                tx.value().clone(),
                Value::String("typed".to_string().into()),
                Value::Tensor(Tensor::new_integer(shape, vec![1, 2]).expect("resize shape")),
            ],
        )
        .expect("queue resize");
        call_builtin(
            "DataTransaction.fill",
            &[
                tx.value().clone(),
                Value::String("typed".to_string().into()),
                Value::Int(attribute.clone()),
            ],
        )
        .expect("queue fill");
        let slice = Value::Cell(
            CellArray::new(
                vec![
                    Value::Tensor(Tensor::new_integer(range, vec![1, 2]).expect("row range")),
                    Value::Int(one),
                ],
                1,
                2,
            )
            .expect("slice"),
        );
        call_builtin(
            "DataTransaction.write",
            &[
                tx.value().clone(),
                Value::String("typed".to_string().into()),
                slice,
                Value::Tensor(Tensor::new_integer(values.clone(), vec![2, 1]).expect("values")),
            ],
        )
        .expect("queue write");
        call_builtin(
            "DataTransaction.set_attr",
            &[
                tx.value().clone(),
                Value::String(format!("{dtype}_one").into()),
                Value::Int(attribute.clone()),
            ],
        )
        .expect("queue attribute");
        let mut attrs = runmat_value::UnboundedStructValueBuilder::new();
        attrs
            .fields
            .insert(format!("{dtype}_many"), Value::Int(attribute));
        call_builtin(
            "DataTransaction.set_attrs",
            &[tx.value().clone(), Value::Struct(attrs.finish())],
        )
        .expect("queue attributes");
        call_builtin("DataTransaction.commit", std::slice::from_ref(tx.value()))
            .expect("commit transaction");

        let arr = call_builtin(
            "Dataset.array",
            &[
                ds.value().clone(),
                Value::String("typed".to_string().into()),
            ],
        )
        .expect("open typed array");
        let read = call_builtin("DataArray.read", std::slice::from_ref(arr.value()))
            .expect("read typed array");
        let Value::Tensor(read_back) = read.value() else {
            panic!("expected tensor");
        };
        assert_eq!(read_back.integer_storage(), Some(&values), "{dtype}");
    }
}

#[test]
fn data_arrays_preserve_every_integer_class_through_chunked_write_and_read() {
    let _serial = serial_test_guard();
    let _provider = native_provider_guard();
    let cases = vec![
        (
            "int8",
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntValue::I8(i8::MIN),
            IntegerStorage::I8(vec![i8::MIN; 2]),
        ),
        (
            "int16",
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntValue::I16(i16::MIN),
            IntegerStorage::I16(vec![i16::MIN; 2]),
        ),
        (
            "int32",
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntValue::I32(i32::MIN),
            IntegerStorage::I32(vec![i32::MIN; 2]),
        ),
        (
            "int64",
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntValue::I64(i64::MIN),
            IntegerStorage::I64(vec![i64::MIN; 2]),
        ),
        (
            "uint8",
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntValue::U8(u8::MAX),
            IntegerStorage::U8(vec![u8::MAX; 2]),
        ),
        (
            "uint16",
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntValue::U16(u16::MAX),
            IntegerStorage::U16(vec![u16::MAX; 2]),
        ),
        (
            "uint32",
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntValue::U32(u32::MAX),
            IntegerStorage::U32(vec![u32::MAX; 2]),
        ),
        (
            "uint64",
            IntegerStorage::U64(vec![0, u64::MAX]),
            IntValue::U64(u64::MAX),
            IntegerStorage::U64(vec![u64::MAX; 2]),
        ),
    ];

    for (dtype, storage, fill, filled_storage) in cases {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir
            .path()
            .join(format!("{dtype}.data"))
            .to_string_lossy()
            .to_string();
        let mut array_meta = runmat_value::UnboundedStructValueBuilder::new();
        array_meta
            .fields
            .insert("dtype".to_string(), Value::String(dtype.to_string().into()));
        array_meta.fields.insert(
            "shape".to_string(),
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).expect("shape")),
        );
        array_meta.fields.insert(
            "chunk".to_string(),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).expect("chunk")),
        );
        let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
        arrays
            .fields
            .insert("samples".to_string(), Value::Struct(array_meta.finish()));
        let mut schema = runmat_value::UnboundedStructValueBuilder::new();
        schema
            .fields
            .insert("arrays".to_string(), Value::Struct(arrays.finish()));

        let ds = call_builtin(
            "data.create",
            &[Value::String(path.into()), Value::Struct(schema.finish())],
        )
        .expect("create dataset");
        let arr = call_builtin(
            "Dataset.array",
            &[
                ds.value().clone(),
                Value::String("samples".to_string().into()),
            ],
        )
        .expect("array");
        let input = Tensor::new_integer(storage.clone(), vec![2, 1]).expect("integer tensor");
        call_builtin(
            "DataArray.write",
            &[arr.value().clone(), Value::Tensor(input)],
        )
        .expect("write integer array");

        let read = call_builtin("DataArray.read", std::slice::from_ref(arr.value())).expect("read");
        let Value::Tensor(read_back) = read.value() else {
            panic!("expected tensor");
        };
        assert_eq!(read_back.integer_storage(), Some(&storage), "{dtype}");

        call_builtin("DataArray.fill", &[arr.value().clone(), Value::Int(fill)])
            .expect("fill integer array");
        let filled = call_builtin("DataArray.read", std::slice::from_ref(arr.value()))
            .expect("read filled array");
        let Value::Tensor(read_back) = filled.value() else {
            panic!("expected tensor");
        };
        assert_eq!(
            read_back.integer_storage(),
            Some(&filled_storage),
            "{dtype} fill"
        );
    }
}

#[test]
fn uint64_data_array_slice_fill_and_transaction_paths_remain_exact() {
    let _serial = serial_test_guard();
    let _provider = native_provider_guard();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("uint64.data").to_string_lossy().to_string();
    let mut array_meta = runmat_value::UnboundedStructValueBuilder::new();
    array_meta.fields.insert(
        "dtype".to_string(),
        Value::String("uint64".to_string().into()),
    );
    array_meta.fields.insert(
        "shape".to_string(),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("shape")),
    );
    array_meta.fields.insert(
        "chunk".to_string(),
        Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).expect("chunk")),
    );
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays
        .fields
        .insert("samples".to_string(), Value::Struct(array_meta.finish()));
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".to_string(), Value::Struct(arrays.finish()));
    let ds = call_builtin(
        "data.create",
        &[Value::String(path.into()), Value::Struct(schema.finish())],
    )
    .expect("create dataset");
    let arr = call_builtin(
        "Dataset.array",
        &[
            ds.value().clone(),
            Value::String("samples".to_string().into()),
        ],
    )
    .expect("array");

    call_builtin(
        "DataArray.fill",
        &[
            arr.value().clone(),
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        ],
    )
    .expect("fill");
    let slice = Value::Cell(
        CellArray::new(
            vec![
                Value::Int(runmat_value::IntValue::I32(1)),
                Value::String(":".to_string().into()),
            ],
            1,
            2,
        )
        .expect("slice"),
    );
    let replacement = Tensor::new_integer(
        IntegerStorage::U64(vec![1_u64 << 63, u64::MAX - 1]),
        vec![1, 2],
    )
    .expect("replacement");
    call_builtin(
        "DataArray.write",
        &[arr.value().clone(), slice, Value::Tensor(replacement)],
    )
    .expect("slice write");

    let tx =
        call_builtin("Dataset.begin", std::slice::from_ref(ds.value())).expect("begin transaction");
    call_builtin(
        "DataTransaction.fill",
        &[
            tx.value().clone(),
            Value::String("samples".to_string().into()),
            Value::Int(runmat_value::IntValue::U64(1_u64 << 63)),
        ],
    )
    .expect("queue transaction fill");
    call_builtin("DataTransaction.commit", std::slice::from_ref(tx.value()))
        .expect("commit transaction");

    let read = call_builtin("DataArray.read", std::slice::from_ref(arr.value())).expect("read");
    let Value::Tensor(read_back) = read.value() else {
        panic!("expected tensor");
    };
    assert_eq!(
        read_back.integer_storage(),
        Some(&IntegerStorage::U64(vec![1_u64 << 63; 4]))
    );
}

#[test]
fn paired_complex_uint64_data_array_chunk_and_slice_paths_remain_exact() {
    let _serial = serial_test_guard();
    let _provider = native_provider_guard();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir
        .path()
        .join("complex-uint64.data")
        .to_string_lossy()
        .to_string();
    let mut array_meta = runmat_value::UnboundedStructValueBuilder::new();
    array_meta.fields.insert(
        "dtype".to_string(),
        Value::String("uint64".to_string().into()),
    );
    array_meta.fields.insert(
        "shape".to_string(),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).expect("shape")),
    );
    array_meta.fields.insert(
        "chunk".to_string(),
        Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).expect("chunk")),
    );
    let mut arrays = runmat_value::UnboundedStructValueBuilder::new();
    arrays
        .fields
        .insert("samples".to_string(), Value::Struct(array_meta.finish()));
    let mut schema = runmat_value::UnboundedStructValueBuilder::new();
    schema
        .fields
        .insert("arrays".to_string(), Value::Struct(arrays.finish()));
    let ds = call_builtin(
        "data.create",
        &[Value::String(path.into()), Value::Struct(schema.finish())],
    )
    .expect("create dataset");
    let arr = call_builtin(
        "Dataset.array",
        &[
            ds.value().clone(),
            Value::String("samples".to_string().into()),
        ],
    )
    .expect("array");
    let wide = (1_u64 << 53) + 1;
    let initial_storage = IntegerComplexStorage::new(
        IntegerStorage::U64(vec![wide, u64::MAX, 3, 4]),
        IntegerStorage::U64(vec![u64::MAX, wide, 5, 6]),
    )
    .expect("initial paired storage");
    let initial = ComplexTensor::new_integer(initial_storage.clone(), vec![2, 2])
        .expect("initial paired tensor");
    call_builtin(
        "DataArray.write",
        &[arr.value().clone(), Value::ComplexTensor(initial)],
    )
    .expect("write paired array");

    let read = call_builtin("DataArray.read", std::slice::from_ref(arr.value()))
        .expect("read paired array");
    let Value::ComplexTensor(read_back) = read.value() else {
        panic!("expected paired tensor");
    };
    assert_eq!(read_back.integer_storage(), Some(&initial_storage));

    let slice = Value::Cell(
        CellArray::new(
            vec![
                Value::Int(IntValue::I32(1)),
                Value::String(":".to_string().into()),
            ],
            1,
            2,
        )
        .expect("slice"),
    );
    let replacement_storage = IntegerComplexStorage::new(
        IntegerStorage::U64(vec![1_u64 << 63, u64::MAX - 1]),
        IntegerStorage::U64(vec![u64::MAX - 2, 1_u64 << 63]),
    )
    .expect("replacement storage");
    let replacement =
        ComplexTensor::new_integer(replacement_storage, vec![1, 2]).expect("replacement tensor");
    call_builtin(
        "DataArray.write",
        &[
            arr.value().clone(),
            slice,
            Value::ComplexTensor(replacement),
        ],
    )
    .expect("write paired slice");

    let read = call_builtin("DataArray.read", std::slice::from_ref(arr.value()))
        .expect("read paired slice result");
    let Value::ComplexTensor(read_back) = read.value() else {
        panic!("expected paired tensor");
    };
    let storage = read_back.integer_storage().expect("paired storage");
    assert_eq!(
        storage.real(),
        &IntegerStorage::U64(vec![1_u64 << 63, u64::MAX, u64::MAX - 1, 4])
    );
    assert_eq!(
        storage.imag(),
        &IntegerStorage::U64(vec![u64::MAX - 2, wide, 1_u64 << 63, 6])
    );
}
