use super::*;
#[cfg(not(target_arch = "wasm32"))]
use crate::builtins::io::http::transport::NativeHttpService;
#[cfg(not(target_arch = "wasm32"))]
use crate::context::{RuntimeContext, RuntimeServicePorts};
use image::{ImageBuffer, ImageOutputFormat, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use runmat_builtins::IMREAD_DESCRIPTOR;
use runmat_value::IntegerStorage;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
#[cfg(not(target_arch = "wasm32"))]
use std::rc::Rc;
use std::sync::Arc;

fn encode_image(image: DynamicImage, format: ImageOutputFormat) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    image.write_to(&mut cursor, format).expect("encode image");
    cursor.into_inner()
}

fn rgb_png() -> Vec<u8> {
    let image = RgbImage::from_fn(2, 2, |x, y| match (x, y) {
        (0, 0) => Rgb([10, 20, 30]),
        (1, 0) => Rgb([40, 50, 60]),
        (0, 1) => Rgb([70, 80, 90]),
        (1, 1) => Rgb([100, 110, 120]),
        _ => unreachable!(),
    });
    encode_image(DynamicImage::ImageRgb8(image), ImageOutputFormat::Png)
}

fn rgba_png() -> Vec<u8> {
    let image = RgbaImage::from_fn(2, 1, |x, _| match x {
        0 => Rgba([1, 2, 3, 4]),
        1 => Rgba([5, 6, 7, 8]),
        _ => unreachable!(),
    });
    encode_image(DynamicImage::ImageRgba8(image), ImageOutputFormat::Png)
}

fn run_imread(bytes: &[u8], extension: &str, rest: Vec<Value>) -> Value {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(format!("image.{extension}"));
    std::fs::write(&path, bytes).expect("write image");
    futures::executor::block_on(imread_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
    .expect("imread")
}

#[test]
fn imread_decodes_rgb_png_as_column_major_truecolor_uint8() {
    let result = run_imread(&rgb_png(), "png", Vec::new());
    let Value::Tensor(tensor) = result else {
        panic!("expected tensor, got {result:?}");
    };
    assert_eq!(tensor.shape(), &[2, 2, 3]);
    assert_eq!(tensor.numeric_dtype(), NumericDType::U8);
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U8(vec![
            10, 70, 40, 100, 20, 80, 50, 110, 30, 90, 60, 120
        ]))
    );
}

#[test]
fn imread_returns_alpha_as_third_output_for_rgba_png() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("alpha.png");
    std::fs::write(&path, rgba_png()).expect("write image");
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let result = futures::executor::block_on(imread_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("imread");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list, got {result:?}");
    };
    assert_eq!(outputs.len(), 3);
    match &outputs[0] {
        Value::Tensor(rgb) => {
            assert_eq!(rgb.shape(), &[1, 2, 3]);
            assert_eq!(rgb.numeric_dtype(), NumericDType::U8);
            assert_eq!(
                rgb.integer_storage(),
                Some(&IntegerStorage::U8(vec![1, 5, 2, 6, 3, 7]))
            );
        }
        other => panic!("expected rgb tensor, got {other:?}"),
    }
    match &outputs[1] {
        Value::Tensor(map) => assert_eq!(map.shape(), &[0, 0]),
        other => panic!("expected empty map tensor, got {other:?}"),
    }
    match &outputs[2] {
        Value::Tensor(alpha) => {
            assert_eq!(alpha.shape(), &[1, 2]);
            assert_eq!(alpha.numeric_dtype(), NumericDType::U8);
            assert_eq!(
                alpha.integer_storage(),
                Some(&IntegerStorage::U8(vec![4, 8]))
            );
        }
        other => panic!("expected alpha tensor, got {other:?}"),
    }
}

#[test]
fn imread_reads_local_file_path() {
    let result = run_imread(&rgb_png(), "png", Vec::new());
    assert!(matches!(result, Value::Tensor(_)));
}

#[test]
fn imread_windows_drive_letter_path_is_not_treated_as_url_scheme() {
    // Url::parse("C:/foo.png") succeeds with scheme "c", which must not be
    // treated as an unsupported URL scheme — it should fall through to the
    // local path reader and produce a file-not-found error, not a scheme error.
    let err = futures::executor::block_on(imread_builtin(
        None,
        Value::from("C:/nonexistent/photo.png"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("expected error for missing file");
    assert_ne!(
        err.identifier(),
        IMREAD_ERROR_UNSUPPORTED_SCHEME.identifier,
        "drive-letter path incorrectly rejected as unsupported URL scheme"
    );
}

#[test]
fn parsed_opaque_and_nonlocal_file_urls_reject_before_filesystem_access() {
    for source in ["mailto:alice@example.com", "ftp://example.test/pixels.png"] {
        let err = futures::executor::block_on(imread_builtin(
            None,
            Value::from(source),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("unsupported URL scheme");
        assert_eq!(err.identifier(), IMREAD_ERROR_UNSUPPORTED_SCHEME.identifier);
    }
    let err = futures::executor::block_on(imread_builtin(
        None,
        Value::from("file://remote.example/pixels.png"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("nonlocal file URL");
    assert_eq!(err.identifier(), IMREAD_ERROR_INVALID_FILE_URL.identifier);
}

#[test]
fn imread_respects_explicit_format_hint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("image-no-extension");
    std::fs::write(&path, rgb_png()).expect("write image");
    let result = futures::executor::block_on(imread_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("png")]),
    ))
    .expect("imread");
    assert!(matches!(result, Value::Tensor(_)));
}

#[test]
fn imread_rejects_unknown_format_hint() {
    let err = futures::executor::block_on(imread_builtin(
        None,
        Value::from("missing"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("not-a-format")]),
    ))
    .expect_err("expected error");
    assert_eq!(err.identifier(), IMREAD_ERROR_UNSUPPORTED_FORMAT.identifier);
}

#[test]
fn output_and_input_arity_reject_before_missing_file_or_provider_work() {
    let _outputs = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let err = futures::executor::block_on(imread_builtin(
        None,
        Value::from("missing.png"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("excess outputs must fail before reading");
    assert_eq!(err.identifier(), IMREAD_ERROR_TOO_MANY_OUTPUTS.identifier);
    drop(_outputs);
    let err = futures::executor::block_on(imread_builtin(
        None,
        Value::from("missing.png"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("png"),
            Value::Num(1.0),
        ]),
    ))
    .expect_err("excess inputs must fail before reading");
    assert_eq!(err.identifier(), IMREAD_ERROR_TOO_MANY_INPUTS.identifier);
}

#[test]
fn numeric_filename_does_not_coerce_to_filesystem_path() {
    let err = futures::executor::block_on(imread_builtin(
        None,
        Value::Num(42.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("numeric filename must reject before reading");
    assert_eq!(err.identifier(), IMREAD_ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn imread_dispatcher_reports_builtin_error_directly() {
    let driver = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("test HTTP driver");
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_http(Rc::new(NativeHttpService::new(driver.handle().clone()))),
        );
    let url = spawn_repeating_server(
        2,
        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_vec(),
    );
    let err = futures::executor::block_on(crate::call_builtin_async_in_context(
        &runtime,
        "imread",
        &[Value::from(format!("{url}/missing.jpg"))],
    ))
    .expect_err("expected 404");
    assert_eq!(err.identifier(), IMREAD_ERROR_HTTP_STATUS.identifier);
    assert!(err.message().contains("HTTP status 404"));
    assert!(!err.message().contains("No matching overload"));
}

#[test]
fn imread_descriptor_signatures_cover_surface() {
    let labels: Vec<&str> = IMREAD_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert_eq!(
        labels,
        vec![
            "I = imread(filename)",
            "I = imread(filename, fmt)",
            "[I, map] = imread(filename)",
            "[I, map] = imread(filename, fmt)",
            "[I, map, alpha] = imread(filename)",
            "[I, map, alpha] = imread(filename, fmt)",
        ]
    );
}

#[test]
fn imread_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = IMREAD_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert_eq!(
        codes,
        vec![
            "RM.IMREAD.INVALID_ARGUMENT",
            "RM.IMREAD.INVALID_FILENAME",
            "RM.IMREAD.INVALID_FORMAT",
            "RM.IMREAD.UNSUPPORTED_FORMAT",
            "RM.IMREAD.TOO_MANY_INPUTS",
            "RM.IMREAD.TOO_MANY_OUTPUTS",
            "RM.IMREAD.UNSUPPORTED_SCHEME",
            "RM.IMREAD.FILE_READ",
            "RM.IMREAD.INVALID_FILE_URL",
            "RM.IMREAD.TIMEOUT",
            "RM.IMREAD.NETWORK",
            "RM.IMREAD.HTTP_STATUS",
            "RM.IMREAD.INVALID_HEADER",
            "RM.IMREAD.DECODE",
            "RM.IMREAD.SHAPE",
        ]
    );
}

#[test]
fn imread_materializes_multi_outputs_with_empty_colormap() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("rgb.png");
    std::fs::write(&path, rgb_png()).expect("write image");
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = futures::executor::block_on(imread_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("imread");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list, got {result:?}");
    };
    assert_eq!(outputs.len(), 2);
    assert!(matches!(&outputs[0], Value::Tensor(_)));
    match &outputs[1] {
        Value::Tensor(map) => assert_eq!(map.shape(), &[0, 0]),
        other => panic!("expected map tensor, got {other:?}"),
    }
}

#[test]
fn imread_decodes_16_bit_grayscale() {
    let image: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_fn(2, 2, |x, y| {
        let value = match (x, y) {
            (0, 0) => 1,
            (1, 0) => 2,
            (0, 1) => 300,
            (1, 1) => 65535,
            _ => unreachable!(),
        };
        Luma([value])
    });
    let bytes = encode_image(DynamicImage::ImageLuma16(image), ImageOutputFormat::Png);
    let result = run_imread(&bytes, "png", Vec::new());
    let Value::Tensor(tensor) = result else {
        panic!("expected tensor, got {result:?}");
    };
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.numeric_dtype(), NumericDType::U16);
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U16(vec![1, 300, 2, 65535]))
    );
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn imread_fetches_http_url() {
    let driver = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("test HTTP driver");
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_http(Rc::new(NativeHttpService::new(driver.handle().clone()))),
        );
    let body = rgb_png();
    let response = http_response(200, "OK", "image/png", &body);
    let url = spawn_server(response);
    let result = futures::executor::block_on(imread_builtin(
        Some(runtime),
        Value::from(format!("{url}/image.png")),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("imread");
    let Value::Tensor(tensor) = result else {
        panic!("expected tensor, got {result:?}");
    };
    assert_eq!(tensor.shape(), &[2, 2, 3]);
    assert_eq!(tensor.numeric_dtype(), NumericDType::U8);
}

#[cfg(not(target_arch = "wasm32"))]
fn http_response(status: u16, reason: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
    let mut response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    response.extend_from_slice(body);
    response
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_server(response: Vec<u8>) -> String {
    spawn_repeating_server(1, response)
}

fn spawn_repeating_server(limit: usize, response: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let response = Arc::new(response);
    std::thread::spawn(move || {
        for stream in listener.incoming().take(limit) {
            let Ok(mut stream) = stream else {
                continue;
            };
            write_response(&mut stream, &response);
        }
    });
    format!("http://{addr}")
}

fn write_response(stream: &mut TcpStream, response: &[u8]) {
    let mut buffer = [0u8; 1024];
    let _ = stream.read(&mut buffer);
    stream.write_all(response).expect("write response");
    stream.flush().expect("flush response");
}
