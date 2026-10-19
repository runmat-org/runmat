#![cfg(target_arch = "wasm32")]

use std::cell::Cell;
use std::io::{Cursor, Write};
use std::rc::Rc;
use std::sync::Arc;

use js_sys::{Function, Object, Promise, Reflect, Uint8Array};
use runmat_filesystem::{self as vfs, MemoryFsProvider};
use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
use runmat_runtime::http::BrowserHttpService;
use runmat_value::Value;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
use web_sys::{ReadableStream, Request, Response, WorkerGlobalScope};
use zip::write::SimpleFileOptions;

wasm_bindgen_test_configure!(run_in_worker);

const ARCHIVE_URL: &str = "https://fixture.invalid/archive.zip";
const OUTPUT_ROOT: &str = "/worker/unpacked";

fn archive_bytes() -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .start_file(
            "bundle/stored.txt",
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored),
        )
        .expect("stored ZIP entry");
    writer.write_all(b"stored Worker bytes").unwrap();
    writer
        .start_file(
            "bundle/deflated.txt",
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated),
        )
        .expect("deflated ZIP entry");
    writer.write_all(b"deflated Worker bytes").unwrap();
    writer.finish().expect("complete ZIP").into_inner()
}

fn worker_scope() -> WorkerGlobalScope {
    assert!(web_sys::window().is_none(), "must execute in a Worker");
    js_sys::global()
        .dyn_into::<WorkerGlobalScope>()
        .expect("WorkerGlobalScope")
}

/// Keep the original own-vs-inherited Worker Fetch property through unwinding.
struct FetchOverride {
    global: Object,
    original: JsValue,
    had_own: bool,
    _callback: Closure<dyn FnMut(Request) -> Promise>,
}

impl FetchOverride {
    fn install(callback: Closure<dyn FnMut(Request) -> Promise>) -> Self {
        let global = worker_scope().unchecked_into::<Object>();
        let key = JsValue::from_str("fetch");
        let original = Reflect::get(&global, &key).expect("original Worker Fetch");
        let had_own = Object::has_own(&global, &key);
        assert!(
            Reflect::set(&global, &key, callback.as_ref()).expect("replace Worker Fetch"),
            "Worker Fetch must be replaceable"
        );
        Self {
            global,
            original,
            had_own,
            _callback: callback,
        }
    }
}

impl Drop for FetchOverride {
    fn drop(&mut self) {
        let key = JsValue::from_str("fetch");
        let restored = if self.had_own {
            Reflect::set(&self.global, &key, &self.original)
        } else {
            Reflect::delete_property(&self.global, &key)
        };
        assert!(matches!(restored, Ok(true)), "restore Worker Fetch");
    }
}

fn streamed_response(bytes: &[u8]) -> Response {
    let middle = bytes.len() / 2;
    let first = Uint8Array::from(&bytes[..middle]);
    let second = Uint8Array::from(&bytes[middle..]);
    let source = Object::new();
    let start = Closure::<dyn FnMut(JsValue)>::new(move |controller| {
        let enqueue = Reflect::get(&controller, &JsValue::from_str("enqueue"))
            .expect("ReadableStream.enqueue")
            .dyn_into::<Function>()
            .expect("enqueue function");
        enqueue.call1(&controller, &first).expect("first ZIP chunk");
        enqueue
            .call1(&controller, &second)
            .expect("second ZIP chunk");
        let close = Reflect::get(&controller, &JsValue::from_str("close"))
            .expect("ReadableStream.close")
            .dyn_into::<Function>()
            .expect("close function");
        close.call0(&controller).expect("end ZIP stream");
    });
    Reflect::set(&source, &JsValue::from_str("start"), start.as_ref()).expect("set stream start");
    let stream = ReadableStream::new_with_underlying_source(&source).expect("ZIP stream");
    drop(start); // The stream constructor invokes `start` synchronously.
    let response = Response::new_with_opt_readable_stream(Some(&stream)).expect("ZIP response");
    response
        .headers()
        .set("Content-Type", "application/zip")
        .expect("ZIP content type");
    response
}

#[wasm_bindgen_test(async)]
async fn worker_unzip_streams_real_zip_into_captured_memory_filesystem() {
    let _scope = worker_scope();
    let selected = Arc::new(MemoryFsProvider::with_current_dir("/worker"));
    let decoy = Arc::new(MemoryFsProvider::with_current_dir("/decoy"));
    let _filesystem = vfs::replace_provider(selected.clone());
    let calls = Rc::new(Cell::new(0usize));
    let calls_for_fetch = Rc::clone(&calls);
    let decoy_for_fetch = Arc::clone(&decoy);
    let bytes = archive_bytes();
    let _fetch = FetchOverride::install(Closure::new(move |request: Request| {
        calls_for_fetch.set(calls_for_fetch.get() + 1);
        assert_eq!(request.url(), ARCHIVE_URL);
        assert_eq!(request.method(), "GET");
        // The URL suspension must not retarget the already selected output.
        vfs::set_provider(decoy_for_fetch.clone());
        let response: JsValue = streamed_response(&bytes).into();
        Promise::resolve(&response)
    }));

    // No parallel/placement quota is installed: this is an execution witness,
    // not a claim about browser transport or ZIP peak-memory admission.
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_http(Rc::new(BrowserHttpService::new())),
    );
    let result = runmat_runtime::call_builtin_async_in_context(
        &context,
        "unzip",
        &[
            Value::String(ARCHIVE_URL.into()),
            Value::String(OUTPUT_ROOT.into()),
        ],
    )
    .await
    .expect("Worker unzip");

    assert_eq!(calls.get(), 1, "one invocation has one Fetch");
    let Value::Cell(names) = result.value() else {
        panic!("unzip must return a filename cell")
    };
    assert_eq!(names.shape, vec![2, 1]);
    let paths: Vec<String> = names
        .data
        .iter()
        .map(|value| match value {
            Value::CharArray(chars) => {
                assert_eq!(chars.rows, 1);
                chars.data.iter().collect()
            }
            other => panic!("filename must be a character row, got {other:?}"),
        })
        .collect();
    assert_eq!(
        paths,
        vec![
            "/worker/unpacked/bundle/stored.txt".to_string(),
            "/worker/unpacked/bundle/deflated.txt".to_string(),
        ]
    );
    assert_eq!(
        selected.read_project_path(&paths[0]).unwrap(),
        b"stored Worker bytes"
    );
    assert_eq!(
        selected.read_project_path(&paths[1]).unwrap(),
        b"deflated Worker bytes"
    );
    assert!(decoy.read_project_path(&paths[0]).is_err());
    assert!(decoy.read_project_path(&paths[1]).is_err());
}

#[wasm_bindgen_test(async)]
async fn worker_crc_failure_removes_unpublished_partial_output() {
    let _scope = worker_scope();
    let selected = Arc::new(MemoryFsProvider::with_current_dir("/worker"));
    let _filesystem = vfs::replace_provider(selected.clone());
    let calls = Rc::new(Cell::new(0usize));
    let calls_for_fetch = Rc::clone(&calls);
    let mut bytes = archive_bytes();
    let original = b"stored Worker bytes";
    let offset = bytes
        .windows(original.len())
        .position(|window| window == original)
        .expect("stored entry payload in ZIP fixture");
    bytes[offset] ^= 1;
    let _fetch = FetchOverride::install(Closure::new(move |request: Request| {
        calls_for_fetch.set(calls_for_fetch.get() + 1);
        assert_eq!(request.url(), ARCHIVE_URL);
        let response: JsValue = streamed_response(&bytes).into();
        Promise::resolve(&response)
    }));
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_http(Rc::new(BrowserHttpService::new())),
    );
    let error = runmat_runtime::call_builtin_async_in_context(
        &context,
        "unzip",
        &[
            Value::String(ARCHIVE_URL.into()),
            Value::String(OUTPUT_ROOT.into()),
        ],
    )
    .await
    .expect_err("corrupt stored entry CRC must fail");
    assert_eq!(error.identifier(), Some("RunMat:unzip:Zip"));
    assert_eq!(calls.get(), 1);
    assert!(selected
        .read_project_path("/worker/unpacked/bundle/stored.txt")
        .is_err());

    // PartialEntry's Worker drop schedules its provider-owned removal on the
    // next microtask; do not mistake its asynchronous cleanup for publication.
    for _ in 0..4 {
        if selected
            .list_project_path("/worker/unpacked/bundle")
            .expect("created output folder")
            .is_empty()
        {
            break;
        }
        wasm_bindgen_futures::JsFuture::from(Promise::resolve(&JsValue::UNDEFINED))
            .await
            .expect("advance cleanup microtask");
    }
    assert!(selected
        .list_project_path("/worker/unpacked/bundle")
        .expect("created output folder")
        .is_empty());
}
