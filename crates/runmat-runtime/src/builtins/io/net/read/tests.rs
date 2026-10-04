use super::*;
use crate::builtins::io::net::client_state::{insert_client, remove_client_for_test};
use crate::builtins::io::net::control::configure_stream;
use crate::builtins::io::net::test_support::CountingDownloadProvider;
use runmat_builtins::READ_DESCRIPTOR;
use runmat_value::{IntValue, IntegerStorage, StructValue, Tensor, Value};
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::thread;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_invalid_count_or_datatype_rejects_before_resident_control_download() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = Box::leak(Box::new(CountingDownloadProvider::new()));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let resident_client = Value::GpuTensor(owner.resident());
    let resident_count = Value::GpuTensor(owner.resident());
    let invalid_datatype = futures::executor::block_on(read_builtin(
        resident_client.clone(),
        vec![resident_count.clone(), Value::from("uint128")],
    ))
    .expect_err("known invalid datatype must reject first");
    assert_eq!(
        invalid_datatype.identifier(),
        Some("RunMat:read:InvalidDataType")
    );
    assert_eq!(owner.downloads(), 0);
    let invalid_count = futures::executor::block_on(read_builtin(
        resident_client,
        vec![Value::Num(0.0), resident_count],
    ))
    .expect_err("known invalid count must reject first");
    assert_eq!(invalid_count.identifier(), Some("RunMat:read:InvalidCount"));
    assert_eq!(owner.downloads(), 0);
}
use std::time::Duration;

fn make_client(stream: TcpStream, timeout: f64) -> Value {
    let peer_addr = stream.peer_addr().expect("peer addr");
    configure_stream(&stream, timeout).expect("configure stream");
    let client_id = insert_client(stream, 0, peer_addr, timeout, "little-endian".to_string());
    let mut st = StructValue::new();
    st.fields.insert(
        CLIENT_HANDLE_FIELD.to_string(),
        Value::Int(IntValue::U64(client_id)),
    );
    Value::Struct(st)
}

fn client_id(client: &Value) -> u64 {
    match client {
        Value::Struct(st) => match st.fields.get(CLIENT_HANDLE_FIELD) {
            Some(Value::Int(IntValue::U64(id))) => *id,
            Some(Value::Int(iv)) => iv.to_i64() as u64,
            other => panic!("unexpected id field {other:?}"),
        },
        other => panic!("expected struct, got {other:?}"),
    }
}

fn assert_error_identifier(err: RuntimeError, expected: &str) {
    assert_eq!(err.identifier(), Some(expected));
}

#[test]
fn typed_read_count_and_handle_parsers_are_exact() {
    assert_eq!(parse_count(&Value::Int(IntValue::U64(42))).unwrap(), 42);
    assert!(parse_count(&Value::Int(IntValue::U8(0))).is_err());
    assert!(parse_count(&Value::Num(2.5)).is_err());
    assert!(parse_count(&Value::Int(IntValue::I8(-1))).is_err());
    let maximum = parse_count(&Value::Int(IntValue::U64(u64::MAX)));
    if usize::BITS == 64 {
        assert_eq!(maximum.unwrap(), usize::MAX);
    } else {
        assert!(maximum.is_err());
    }

    let typed_count =
        Tensor::new_integer(IntegerStorage::U64(vec![42]), vec![1, 1]).expect("count");
    assert_eq!(parse_count(&Value::Tensor(typed_count)).unwrap(), 42);

    let boundary = parse_count(&Value::Num(usize::MAX as f64));
    if usize::BITS == 64 {
        assert!(boundary.is_err());
    } else {
        assert_eq!(boundary.unwrap(), usize::MAX);
    }
    assert!(parse_count(&Value::Num((usize::MAX as f64) + 1.0)).is_err());

    let typed_negative =
        Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("count");
    assert!(parse_count(&Value::Tensor(typed_negative)).is_err());

    let typed_max =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("count");
    let maximum = parse_count(&Value::Tensor(typed_max));
    if usize::BITS == 64 {
        assert_eq!(maximum.unwrap(), usize::MAX);
    } else {
        assert!(maximum.is_err());
    }

    let mut client = StructValue::new();
    client.fields.insert(
        CLIENT_HANDLE_FIELD.to_string(),
        Value::Int(IntValue::I8(-1)),
    );
    let err = extract_client_id(&client).unwrap_err();
    assert_error_identifier(err, READ_ERROR_INVALID_CLIENT.identifier.unwrap());
}

fn run_read(client: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(read_builtin(client, rest))
}

fn net_guard() -> std::sync::MutexGuard<'static, ()> {
    crate::builtins::io::net::client_state::test_guard()
}

fn integer_bytes(storage: &IntegerStorage, order: ByteOrder) -> Vec<u8> {
    let mut bytes = Vec::new();
    match storage {
        IntegerStorage::I8(values) => bytes.extend(values.iter().map(|value| *value as u8)),
        IntegerStorage::U8(values) => bytes.extend_from_slice(values),
        IntegerStorage::I16(values) => {
            for value in values {
                bytes.extend_from_slice(&match order {
                    ByteOrder::Little => value.to_le_bytes(),
                    ByteOrder::Big => value.to_be_bytes(),
                });
            }
        }
        IntegerStorage::U16(values) => {
            for value in values {
                bytes.extend_from_slice(&match order {
                    ByteOrder::Little => value.to_le_bytes(),
                    ByteOrder::Big => value.to_be_bytes(),
                });
            }
        }
        IntegerStorage::I32(values) => {
            for value in values {
                bytes.extend_from_slice(&match order {
                    ByteOrder::Little => value.to_le_bytes(),
                    ByteOrder::Big => value.to_be_bytes(),
                });
            }
        }
        IntegerStorage::U32(values) => {
            for value in values {
                bytes.extend_from_slice(&match order {
                    ByteOrder::Little => value.to_le_bytes(),
                    ByteOrder::Big => value.to_be_bytes(),
                });
            }
        }
        IntegerStorage::I64(values) => {
            for value in values {
                bytes.extend_from_slice(&match order {
                    ByteOrder::Little => value.to_le_bytes(),
                    ByteOrder::Big => value.to_be_bytes(),
                });
            }
        }
        IntegerStorage::U64(values) => {
            for value in values {
                bytes.extend_from_slice(&match order {
                    ByteOrder::Little => value.to_le_bytes(),
                    ByteOrder::Big => value.to_be_bytes(),
                });
            }
        }
    }
    bytes
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn read_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = READ_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"data = read(client)"));
    assert!(labels.contains(&"data = read(client, count)"));
    assert!(labels.contains(&"data = read(client, count, datatype)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn read_reads_requested_uint8_values() {
    let _guard = net_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let payload: Vec<u8> = (1..=10).collect();
        stream.write_all(&payload).expect("write");
    });

    let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let client = make_client(stream, 1.0);

    let data = run_read(client.clone(), vec![Value::Num(6.0)]).expect("read");
    let tensor = match data {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };
    assert_eq!(tensor.shape, vec![1, 6]);
    assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    handle.join().expect("server thread");
    remove_client_for_test(client_id(&client));
}

#[test]
fn bytes_to_value_preserves_all_integer_classes_in_both_byte_orders() {
    let cases = [
        (DataType::Int8, IntegerStorage::I8(vec![i8::MIN, i8::MAX])),
        (DataType::UInt8, IntegerStorage::U8(vec![0, u8::MAX])),
        (
            DataType::Int16,
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        ),
        (DataType::UInt16, IntegerStorage::U16(vec![0, u16::MAX])),
        (
            DataType::Int32,
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        ),
        (DataType::UInt32, IntegerStorage::U32(vec![0, u32::MAX])),
        (
            DataType::Int64,
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        ),
        (
            DataType::UInt64,
            IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
        ),
    ];

    for order in [ByteOrder::Little, ByteOrder::Big] {
        for (datatype, expected) in &cases {
            let decoded = bytes_to_value(&integer_bytes(expected, order), *datatype, order)
                .expect("typed decode");
            let Value::Tensor(tensor) = decoded else {
                panic!("expected tensor");
            };
            assert_eq!(tensor.integer_storage(), Some(expected));
        }
    }
}

#[test]
fn read_default_uint8_and_explicit_single_output_classes_are_truthful() {
    let default =
        bytes_to_value(&[1, 2, 255], DataType::UInt8, ByteOrder::Little).expect("default decode");
    let Value::Tensor(default) = default else {
        panic!("expected default tensor");
    };
    assert_eq!(
        default.integer_storage(),
        Some(&IntegerStorage::U8(vec![1, 2, 255]))
    );

    let bytes = [1.25_f32, -2.5_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let single =
        bytes_to_value(&bytes, DataType::Single, ByteOrder::Little).expect("single decode");
    let Value::Tensor(single) = single else {
        panic!("expected single tensor");
    };
    assert_eq!(single.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(single.materialize_f64(), vec![1.25, -2.5]);
}

#[test]
fn read_requested_uint64_preserves_exact_socket_values() {
    let _guard = net_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        stream
            .write_all(
                &[1_u64 << 63, u64::MAX]
                    .into_iter()
                    .flat_map(u64::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .expect("write");
    });

    let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let client = make_client(stream, 1.0);
    let data =
        run_read(client.clone(), vec![Value::Num(2.0), Value::from("uint64")]).expect("read");
    let Value::Tensor(tensor) = data else {
        panic!("expected tensor");
    };
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]))
    );

    handle.join().expect("server thread");
    remove_client_for_test(client_id(&client));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn read_without_count_drains_available_bytes() {
    let _guard = net_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        std::thread::sleep(Duration::from_millis(50));
        let payload: Vec<u8> = vec![42, 43, 44];
        stream.write_all(&payload).expect("write");
    });

    let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let client = make_client(stream, 1.0);

    let data = run_read(client.clone(), Vec::new()).expect("read");
    let tensor = match data {
        Value::Tensor(t) => t,
        other => panic!("expected tensor result, got {other:?}"),
    };
    assert_eq!(tensor.shape, vec![1, 3]);
    assert_eq!(tensor.materialize_f64(), vec![42.0, 43.0, 44.0]);

    handle.join().expect("server thread");
    remove_client_for_test(client_id(&client));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn read_respects_timeout() {
    let _guard = net_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let port = listener.local_addr().unwrap().port();
    let _handle = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        // Keep the connection open well beyond the client timeout so the
        // read call experiences a timeout rather than a clean closure.
        std::thread::sleep(Duration::from_millis(500));
        drop(stream);
    });

    let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let client = make_client(stream, 0.1);

    let err = run_read(client.clone(), vec![Value::Num(4.0)]).unwrap_err();
    assert_eq!(err.identifier(), READ_ERROR_TIMEOUT.identifier);
    assert_eq!(err.message(), READ_ERROR_TIMEOUT.message);
    assert_eq!(err.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    let source = std::error::Error::source(&err)
        .and_then(|source| source.downcast_ref::<io::Error>())
        .expect("timed-out socket error remains the cause");
    assert!(matches!(
        source.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    ));

    remove_client_for_test(client_id(&client));
}
