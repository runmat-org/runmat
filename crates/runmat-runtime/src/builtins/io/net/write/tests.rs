use super::*;
use crate::builtins::io::net::test_support::{CountingDownloadProvider, TcpTestHost};
use crate::context::RuntimeTcpByteOrder;
use runmat_builtins::WRITE_DESCRIPTOR;
use runmat_value::{CharArray, IntValue, IntegerStorage, ObjectInstance, SparseTensor, Tensor};
use std::io::{self, Read};
use std::net::TcpListener;
use std::thread;

fn assert_error_identifier(err: RuntimeError, expected: &str) {
    assert_eq!(err.identifier(), Some(expected));
}

fn prepare_payload(
    data: &Value,
    datatype: DataType,
    order: ByteOrder,
) -> BuiltinResult<codec::Payload> {
    let mut work = super::super::transport::NetCooperation::new(None, BUILTIN_NAME);
    futures::executor::block_on(super::prepare_payload(data, datatype, order, &mut work))
}

fn prepare_default_payload(data: &Value, order: ByteOrder) -> BuiltinResult<codec::Payload> {
    let mut work = super::super::transport::NetCooperation::new(None, BUILTIN_NAME);
    futures::executor::block_on(super::prepare_default_payload(data, order, &mut work))
}

#[test]
fn terminal_socket_errors_retain_original_causes_without_replay() {
    use std::error::Error;
    for (kind, descriptor, io_kind) in [
        (
            TcpTransportErrorKind::Timeout,
            &WRITE_ERROR_TIMEOUT,
            io::ErrorKind::TimedOut,
        ),
        (
            TcpTransportErrorKind::Closed,
            &WRITE_ERROR_CONNECTION_CLOSED,
            io::ErrorKind::BrokenPipe,
        ),
    ] {
        let error = crate::tcp::TcpTransportError::with_source(
            kind,
            "TCP write failed",
            io::Error::new(io_kind, "wire failure"),
        );
        let failure = super::super::error::with_source(
            BUILTIN_NAME,
            descriptor,
            format!("write: {error}"),
            error,
        );
        assert_eq!(failure.identifier(), descriptor.identifier);
        assert!(failure.message().contains(descriptor.message));
        assert_eq!(failure.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert_eq!(
            Error::source(&failure)
                .and_then(|source| source.downcast_ref::<crate::tcp::TcpTransportError>())
                .and_then(Error::source)
                .and_then(|source| source.downcast_ref::<io::Error>())
                .map(io::Error::kind),
            Some(io_kind),
        );
    }
}

#[test]
fn detached_client_object_is_rejected() {
    let host = TcpTestHost::new();
    let client = Value::Object(ObjectInstance::new(runmat_builtins::TCPCLIENT_CLASS));
    let err = run_write(&host, client, Value::Num(1.0), Vec::new()).unwrap_err();
    assert_error_identifier(err, WRITE_ERROR_INVALID_CLIENT.identifier.unwrap());
}

fn run_write(
    host: &TcpTestHost,
    client: Value,
    data: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    futures::executor::block_on(write_builtin(
        Some(host.runtime.clone()),
        client,
        data,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_invalid_datatype_and_arity_reject_before_payload_download() {
    let host = TcpTestHost::new();
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let owner = Box::leak(Box::new(CountingDownloadProvider::new()));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let payload = Value::GpuTensor(owner.resident());
    let invalid_datatype = run_write(
        &host,
        Value::Num(0.0),
        payload.clone(),
        vec![Value::from("uint128")],
    )
    .expect_err("known invalid datatype rejects before resident payload download");
    assert_eq!(
        invalid_datatype.identifier(),
        WRITE_ERROR_INVALID_DATATYPE.identifier
    );
    assert_eq!(owner.downloads(), 0);
    let overarity = run_write(
        &host,
        Value::Num(0.0),
        payload,
        vec![Value::from("uint8"), Value::from("double")],
    )
    .expect_err("extra controls reject before resident payload download");
    assert_eq!(overarity.identifier(), WRITE_ERROR_INVALID_INPUT.identifier);
    assert_eq!(owner.downloads(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn write_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = WRITE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"count = write(client, data)"));
    assert!(labels.contains(&"count = write(client, data, datatype)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn write_default_preserves_double_datatype() {
    let host = TcpTestHost::new();
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut received = Vec::new();
        stream.read_to_end(&mut received).unwrap_or_default();
        received
    });

    let client = host.connect(port, 1.0, RuntimeTcpByteOrder::Little);
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
    let result =
        run_write(&host, client.clone(), Value::Tensor(tensor), Vec::new()).expect("write");
    match result {
        Value::Num(count) => assert_eq!(count, 4.0),
        other => panic!("expected numeric result, got {other:?}"),
    }
    host.close(&client);
    let received = handle.join().expect("join");
    let expected = [1.0_f64, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f64::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(received, expected);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn write_double_big_endian_encodes_correctly() {
    let host = TcpTestHost::new();
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut buf = [0u8; 24];
        stream.read_exact(&mut buf).expect("read");
        buf
    });

    let client = host.connect(port, 1.0, RuntimeTcpByteOrder::Big);
    let tensor = Tensor::new(vec![1.5, 2.5, 3.5], vec![1, 3]).unwrap();
    let result = run_write(
        &host,
        client.clone(),
        Value::Tensor(tensor),
        vec![Value::from("double")],
    )
    .expect("write");
    match result {
        Value::Num(count) => assert_eq!(count, 3.0),
        other => panic!("expected numeric count, got {other:?}"),
    }
    host.close(&client);

    let received = handle.join().expect("join");
    let mut expected = Vec::new();
    extend_f64(&mut expected, 1.5, ByteOrder::Big);
    extend_f64(&mut expected, 2.5, ByteOrder::Big);
    extend_f64(&mut expected, 3.5, ByteOrder::Big);
    assert_eq!(received.to_vec(), expected);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn write_uint64_payload_reads_typed_integer_storage_exactly() {
    let wide = (1_u64 << 53) + 1;
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2])
        .expect("typed integer tensor");

    let payload =
        prepare_payload(&Value::Tensor(tensor), DataType::UInt64, ByteOrder::Little).unwrap();
    let mut expected = Vec::new();
    extend_u64(&mut expected, wide, ByteOrder::Little);
    extend_u64(&mut expected, u64::MAX, ByteOrder::Little);
    assert_eq!(payload.elements, 2);
    assert_eq!(payload.bytes, expected);
}

#[test]
fn write_default_payload_preserves_uint64_datatype_exactly() {
    let wide = (1_u64 << 53) + 1;
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2])
        .expect("typed integer tensor");

    let payload = prepare_default_payload(&Value::Tensor(tensor), ByteOrder::Big).unwrap();
    let expected = [wide, u64::MAX]
        .into_iter()
        .flat_map(u64::to_be_bytes)
        .collect::<Vec<_>>();
    assert_eq!(payload.elements, 2);
    assert_eq!(payload.bytes, expected);
}

#[test]
fn write_extensions_are_gated_before_client_access() {
    let host = TcpTestHost::new();
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let data = Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
    let error = run_write(&host, Value::Num(0.0), data, Vec::new())
        .expect_err("strict mode rejects nonvector data before client access");
    assert_eq!(
        error.identifier(),
        WRITE_NONVECTOR_EXTENSION.error_identifier
    );

    for (shape, values) in [(Vec::new(), vec![1.0]), (vec![2], vec![1.0, 2.0])] {
        let data = Value::Tensor(Tensor::new(values, shape).expect("low-rank payload"));
        let error = run_write(&host, Value::Num(0.0), data, Vec::new())
            .expect_err("vector payload reaches client admission");
        assert_eq!(error.identifier(), WRITE_ERROR_INVALID_CLIENT.identifier);
    }

    let sparse =
        SparseTensor::new_integer(2, 1, vec![0, 1], vec![0], IntegerStorage::U8(vec![1])).unwrap();
    let error = run_write(
        &host,
        Value::Num(0.0),
        Value::SparseTensor(sparse),
        Vec::new(),
    )
    .expect_err("strict mode rejects sparse data before client access");
    assert_eq!(error.identifier(), WRITE_SPARSE_EXTENSION.error_identifier);

    let error = run_write(&host, Value::Num(0.0), Value::Bool(true), Vec::new())
        .expect_err("strict mode rejects logical data before client access");
    assert_eq!(error.identifier(), WRITE_LOGICAL_EXTENSION.error_identifier);

    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let original = allocation.handle.wire_descriptor();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let handle = allocation
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("explicit immutable view");
    let error = run_write(&host, Value::Num(0.0), Value::GpuTensor(handle), Vec::new())
        .expect_err("strict mode rejects explicit GPU data before provider access");
    assert_eq!(
        error.identifier(),
        WRITE_EXPLICIT_GPU_EXTENSION.error_identifier
    );

    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn write_sparse_integer_payload_uses_exact_storage() {
    let wide = (1_u64 << 53) + 1;
    let sparse = SparseTensor::new_integer(
        2,
        2,
        vec![0, 1, 2],
        vec![1, 0],
        IntegerStorage::U64(vec![wide, u64::MAX]),
    )
    .expect("typed sparse tensor");

    let payload = prepare_payload(
        &Value::SparseTensor(sparse),
        DataType::UInt64,
        ByteOrder::Little,
    )
    .expect("sparse uint64 payload");
    let mut expected = Vec::new();
    extend_u64(&mut expected, 0, ByteOrder::Little);
    extend_u64(&mut expected, wide, ByteOrder::Little);
    extend_u64(&mut expected, u64::MAX, ByteOrder::Little);
    extend_u64(&mut expected, 0, ByteOrder::Little);
    assert_eq!(payload.elements, 4);
    assert_eq!(payload.bytes, expected);

    let signed = SparseTensor::new_integer(
        2,
        1,
        vec![0, 2],
        vec![0, 1],
        IntegerStorage::I16(vec![-7, 42]),
    )
    .expect("signed sparse tensor");
    let payload = prepare_payload(
        &Value::SparseTensor(signed),
        DataType::Int16,
        ByteOrder::Little,
    )
    .expect("sparse int16 payload");
    assert_eq!(
        payload.bytes,
        [-7_i16, 42]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn write_integer_payload_narrowing_uses_exact_integer_bounds() {
    let value = Value::Int(IntValue::U64((1_u64 << 53) + 1));
    let payload = prepare_payload(&value, DataType::UInt32, ByteOrder::Big).unwrap();
    assert_eq!(payload.bytes, u32::MAX.to_be_bytes());

    let value = Value::Int(IntValue::I64(i64::MIN + 1));
    let payload = prepare_payload(&value, DataType::Int16, ByteOrder::Little).unwrap();
    assert_eq!(payload.bytes, i16::MIN.to_le_bytes());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn write_char_payload_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::U16(vec![82, 77]), vec![1, 2]).expect("chars");

    let payload = prepare_payload(&Value::Tensor(tensor), DataType::Char, ByteOrder::Little)
        .expect("char payload");
    assert_eq!(payload.elements, 2);
    assert_eq!(payload.bytes, b"RM");
}

#[test]
fn borrowed_codec_preserves_column_major_pages_and_text_classes() {
    let tensor = Tensor::new_integer(
        IntegerStorage::I32(vec![-1, 2, i32::MIN, i32::MAX]),
        vec![1, 2, 2],
    )
    .expect("paged exact integer tensor");
    let payload = prepare_default_payload(&Value::Tensor(tensor), ByteOrder::Big)
        .expect("borrowed paged payload");
    assert_eq!(payload.elements, 4);
    assert_eq!(
        payload.bytes,
        [-1_i32, 2, i32::MIN, i32::MAX]
            .into_iter()
            .flat_map(i32::to_be_bytes)
            .collect::<Vec<_>>()
    );

    let chars = CharArray::from_unicode_scalars_with_shape(vec!['A', 'é', '𐀀'], vec![1, 1, 4])
        .expect("paged character input");
    assert_eq!(chars.shape(), &[1, 1, 4]);
    let payload = prepare_payload(
        &Value::CharArray(chars.clone()),
        DataType::Char,
        ByteOrder::Little,
    )
    .expect("byte-oriented char payload");
    assert_eq!(payload.elements, 4);
    assert_eq!(payload.bytes, [b'A', 0xe9, 0, 0]);
    let payload = prepare_payload(
        &Value::CharArray(chars),
        DataType::String,
        ByteOrder::Little,
    )
    .expect("UTF-8 string payload");
    assert_eq!(payload.elements, 1);
    assert_eq!(payload.bytes, "Aé𐀀".as_bytes());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn write_char_payload_encodes_ascii() {
    let host = TcpTestHost::new();
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).unwrap_or_default();
        buf
    });

    let client = host.connect(port, 1.0, RuntimeTcpByteOrder::Little);
    let chars = CharArray::from_unicode_scalars("RunMat".chars().collect(), 1, 6).unwrap();
    let result = run_write(
        &host,
        client.clone(),
        Value::CharArray(chars),
        vec![Value::from("char")],
    )
    .expect("write");
    match result {
        Value::Num(count) => assert_eq!(count, 6.0),
        other => panic!("expected numeric count, got {other:?}"),
    }
    host.close(&client);
    let received = handle.join().expect("join");
    assert_eq!(received, b"RunMat");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn write_errors_when_client_disconnected() {
    let host = TcpTestHost::new();
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        drop(stream);
    });
    let client = host.connect(port, 1.0, RuntimeTcpByteOrder::Little);
    handle.join().expect("peer closed");
    host.wait_disconnected(&client);

    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let err =
        run_write(&host, client.clone(), Value::Tensor(tensor), Vec::new()).expect_err("write");
    assert_error_identifier(err, WRITE_ERROR_NOT_CONNECTED.identifier.unwrap());

    host.close(&client);
}
