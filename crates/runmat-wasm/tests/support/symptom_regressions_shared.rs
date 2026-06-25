#![cfg(target_arch = "wasm32")]

use runmat_wasm::{init_runmat, RunMatWasm};
use serde::Deserialize;
use wasm_bindgen::JsValue;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExecPayload {
    pub(crate) error: Option<ExecError>,
    pub(crate) stdout: Vec<ExecStream>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExecError {
    pub(crate) message: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExecStream {
    pub(crate) stream: String,
    pub(crate) text: String,
}

fn init_options(enable_gpu: bool) -> JsValue {
    let options = js_sys::Object::new();
    js_sys::Reflect::set(
        &options,
        &JsValue::from_str("enableGpu"),
        &JsValue::from_bool(enable_gpu),
    )
    .expect("set enableGpu");
    js_sys::Reflect::set(
        &options,
        &JsValue::from_str("fsProvider"),
        &test_fs_provider(),
    )
    .expect("set fsProvider");
    options.into()
}

pub(crate) fn test_fs_provider() -> JsValue {
    js_sys::eval(
        r#"
(function () {
  const files = new Map();
  const dirs = new Set(["/"]);
  const now = () => Date.now();
  const modified = new Map([["/", now()]]);
  const normalize = (path) => {
    let text = String(path || "/").replace(/\\/g, "/");
    if (!text.startsWith("/")) text = "/" + text;
    const parts = [];
    for (const part of text.split("/")) {
      if (!part || part === ".") continue;
      if (part === "..") parts.pop();
      else parts.push(part);
    }
    return "/" + parts.join("/");
  };
  const parentOf = (path) => {
    const normalized = normalize(path);
    const index = normalized.lastIndexOf("/");
    return index <= 0 ? "/" : normalized.slice(0, index);
  };
  const baseName = (path) => normalize(path).split("/").pop() || "";
  const notFound = (path) => {
    const err = new Error(`NotFound: ${normalize(path)}`);
    err.name = "NotFoundError";
    err.code = "NotFound";
    return err;
  };
  const ensureDir = (path) => {
    const normalized = normalize(path);
    if (!dirs.has(normalized)) throw notFound(normalized);
  };
  const createDirAll = (path) => {
    const normalized = normalize(path);
    let current = "";
    for (const part of normalized.split("/").filter(Boolean)) {
      current = `${current}/${part}`;
      dirs.add(current);
      modified.set(current, now());
    }
  };
  const bytesOf = (data) => {
    if (data instanceof Uint8Array) return data.slice();
    if (ArrayBuffer.isView(data)) {
      return new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
    }
    return new Uint8Array(data).slice();
  };
  const writeFile = async (path, data) => {
    await Promise.resolve();
    const normalized = normalize(path);
    createDirAll(parentOf(normalized));
    files.set(normalized, bytesOf(data));
    modified.set(normalized, now());
  };
  const readFile = async (path) => {
    await Promise.resolve();
    const normalized = normalize(path);
    const data = files.get(normalized);
    if (!data) throw notFound(normalized);
    return data.slice();
  };
  const removeFile = (path) => {
    const normalized = normalize(path);
    if (!files.delete(normalized)) throw notFound(normalized);
    modified.delete(normalized);
  };
  const metadata = (path) => {
    const normalized = normalize(path);
    if (dirs.has(normalized)) {
      return { fileType: "directory", len: 0, modified: modified.get(normalized) || now(), readonly: false };
    }
    const data = files.get(normalized);
    if (!data) throw notFound(normalized);
    return { fileType: "file", len: data.length, modified: modified.get(normalized) || now(), readonly: false };
  };
  const readDir = (path) => {
    const normalized = normalize(path);
    ensureDir(normalized);
    const prefix = normalized === "/" ? "/" : `${normalized}/`;
    const names = new Set();
    for (const dir of dirs) {
      if (dir !== normalized && dir.startsWith(prefix)) {
        const rest = dir.slice(prefix.length);
        if (rest && !rest.includes("/")) names.add(rest);
      }
    }
    for (const file of files.keys()) {
      if (file.startsWith(prefix)) {
        const rest = file.slice(prefix.length);
        if (rest && !rest.includes("/")) names.add(rest);
      }
    }
    return Array.from(names).sort().map((name) => {
      const child = normalized === "/" ? `/${name}` : `${normalized}/${name}`;
      return { path: child, fileName: name, fileType: dirs.has(child) ? "directory" : "file" };
    });
  };
  const createDir = (path) => {
    const normalized = normalize(path);
    ensureDir(parentOf(normalized));
    dirs.add(normalized);
    modified.set(normalized, now());
  };
  const removeDir = (path) => {
    const normalized = normalize(path);
    if (normalized === "/" || !dirs.has(normalized)) throw notFound(normalized);
    if (readDir(normalized).length > 0) throw new Error(`DirectoryNotEmpty: ${normalized}`);
    dirs.delete(normalized);
    modified.delete(normalized);
  };
  const removeDirAll = (path) => {
    const normalized = normalize(path);
    if (normalized === "/") throw new Error("Cannot remove root");
    for (const dir of Array.from(dirs)) {
      if (dir === normalized || dir.startsWith(`${normalized}/`)) {
        dirs.delete(dir);
        modified.delete(dir);
      }
    }
    for (const file of Array.from(files.keys())) {
      if (file === normalized || file.startsWith(`${normalized}/`)) {
        files.delete(file);
        modified.delete(file);
      }
    }
  };
  const rename = (from, to) => {
    const src = normalize(from);
    const dst = normalize(to);
    if (files.has(src)) {
      writeFile(dst, files.get(src));
      files.delete(src);
      modified.delete(src);
      return;
    }
    if (!dirs.has(src)) throw notFound(src);
    createDirAll(dst);
    for (const file of Array.from(files.keys())) {
      if (file.startsWith(`${src}/`)) {
        const next = `${dst}${file.slice(src.length)}`;
        writeFile(next, files.get(file));
        files.delete(file);
      }
    }
    removeDirAll(src);
  };
  return {
    readFile,
    writeFile,
    removeFile,
    metadata,
    symlinkMetadata: metadata,
    readDir,
    canonicalize: normalize,
    createDir,
    createDirAll,
    removeDir,
    removeDirAll,
    rename,
    setReadonly: function () {}
  };
})()
"#,
    )
    .expect("create wasm test filesystem provider")
}

pub(crate) async fn execute_script_with_runtime(runtime: &RunMatWasm, script: &str) -> ExecPayload {
    let request = text_execute_request("symptom_regression.m", script);
    serde_wasm_bindgen::from_value(
        runtime
            .execute_request_js(request)
            .await
            .expect("execute script"),
    )
    .expect("deserialize execution payload")
}

async fn execute_script(script: &str) -> ExecPayload {
    let runtime = init_runmat(init_options(false))
        .await
        .expect("initialize wasm runtime");
    execute_script_with_runtime(&runtime, script).await
}

fn text_execute_request(name: &str, script: &str) -> JsValue {
    let source = js_sys::Object::new();
    js_sys::Reflect::set(
        &source,
        &JsValue::from_str("kind"),
        &JsValue::from_str("text"),
    )
    .expect("set source kind");
    js_sys::Reflect::set(
        &source,
        &JsValue::from_str("name"),
        &JsValue::from_str(name),
    )
    .expect("set source name");
    js_sys::Reflect::set(
        &source,
        &JsValue::from_str("text"),
        &JsValue::from_str(script),
    )
    .expect("set source text");

    let request = js_sys::Object::new();
    js_sys::Reflect::set(&request, &JsValue::from_str("source"), source.as_ref())
        .expect("set request source");
    request.into()
}

pub(crate) fn stdout_text(payload: &ExecPayload) -> String {
    payload
        .stdout
        .iter()
        .filter(|entry| entry.stream == "stdout")
        .map(|entry| entry.text.as_str())
        .collect::<String>()
}

fn finite_stdout_numbers(stdout_text: &str) -> Vec<f64> {
    stdout_text
        .split(|ch: char| ch.is_whitespace() || matches!(ch, '[' | ']' | ',' | ';'))
        .filter_map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                None
            } else {
                trimmed.parse::<f64>().ok()
            }
        })
        .filter(|value| value.is_finite())
        .collect()
}

pub(crate) async fn assert_signal_compatibility_harness_executes_without_runtime_error() {
    let script =
        include_str!("../../../runmat-runtime/tests/fixtures/signal_compatibility_harness.m");

    let payload = execute_script(script).await;
    if let Some(err) = payload.error {
        panic!(
            "signal compatibility harness wasm execution failed: {}",
            err.message
        );
    }
    let stdout_text = stdout_text(&payload);
    assert!(
        stdout_text.contains("RESULT_signal_compat csv=4 fft=2.0 conv=-1.0 mat=1.0"),
        "signal compatibility harness produced unexpected stdout: {stdout_text:?}"
    );
}

pub(crate) async fn assert_impedance_loop_executes_without_runtime_error() {
    let script = r#"
% Given values
f = 50;
I = 0.4;
V_target = 500e3;

% Reactance
Xc = V_target / I;

% Resistance
R = 25e3;

L = 3000:200:5000;

Vc = zeros(size(L));
Vin = zeros(size(L));

for k = 1:length(L)
    XL = 2*3.141592653589793*f*L(k);

    % Total impedance
    Z = sqrt(R^2 + (XL - Xc)^2);

    % Input voltage
    Vin(k) = I * Z;

    % Cable voltage
    Vc(k) = I * Xc;
end

disp(Vin);
disp(Vc);
"#;

    let payload = execute_script(script).await;
    if let Some(err) = payload.error {
        panic!("impedance loop wasm execution failed: {}", err.message);
    }
    let stdout_text = stdout_text(&payload);
    assert!(
        stdout_text.contains("500000"),
        "unexpected impedance loop stdout: {stdout_text:?}"
    );
}

pub(crate) async fn assert_slice_end_arithmetic_executes_without_runtime_error() {
    let script = r#"
Z = reshape(1:625, 25, 25);

hx = 1;
hy = 1;

[rows, cols] = size(Z);
DX = zeros(rows, cols);
DY = zeros(rows, cols);

% Central differences for interior
DX(:, 2:end-1) = (Z(:, 3:end) - Z(:, 1:end-2)) / (2*hx);
DY(2:end-1, :) = (Z(3:end, :) - Z(1:end-2, :)) / (2*hy);

% Forward/backward differences for edges
DX(:, 1) = (Z(:, 2) - Z(:, 1)) / hx;
DX(:, end) = (Z(:, end) - Z(:, end-1)) / hx;
DY(1, :) = (Z(2, :) - Z(1, :)) / hy;
DY(end, :) = (Z(end, :) - Z(end-1, :)) / hy;

probe = DX(13, 13) + DY(13, 13);
disp(probe);
"#;

    let payload = execute_script(script).await;
    if let Some(err) = payload.error {
        panic!("slice arithmetic wasm execution failed: {}", err.message);
    }
    let stdout_text = stdout_text(&payload);
    assert!(
        !stdout_text.trim().is_empty(),
        "slice arithmetic produced empty stdout unexpectedly"
    );
}

pub(crate) async fn assert_tic_toc_loop_executes_without_runtime_error() {
    let script = r#"
tic();
for k = 1:1e5
    sqrt(k);
end
elapsedStack = toc();

tic
for k = 1:1e5
    sqrt(k);
end
elapsedBare = toc;

timerVal = tic();
for k = 1:1e5
    sqrt(k);
end
elapsedHandle = toc(timerVal);

disp(elapsedStack);
disp(elapsedBare);
disp(elapsedHandle);
"#;

    let payload = execute_script(script).await;
    if let Some(err) = payload.error {
        panic!("tic/toc loop wasm execution failed: {}", err.message);
    }
    let stdout_text = stdout_text(&payload);
    let elapsed_values = finite_stdout_numbers(&stdout_text);
    assert!(
        elapsed_values.len() >= 3,
        "tic/toc loop produced unexpected stdout: {stdout_text:?}"
    );
    assert!(
        elapsed_values.iter().take(3).all(|value| *value >= 0.0),
        "tic/toc loop produced negative elapsed values: {elapsed_values:?}"
    );
}

pub(crate) async fn assert_symbolic_limit_workflow_executes_without_runtime_error() {
    let script = r#"
syms x h
f1 = limit(sin(x)/x, x, 0);
f2 = limit((cos(x+h) - cos(x))/h, h, 0);

disp(f1);
disp(f2);
"#;

    let payload = execute_script(script).await;
    if let Some(err) = payload.error {
        panic!("symbolic limit wasm execution failed: {}", err.message);
    }
    let stdout_text = stdout_text(&payload);
    assert!(
        stdout_text.split_whitespace().any(|token| token == "1"),
        "symbolic limit workflow did not print sinc limit: {stdout_text:?}"
    );
    assert!(
        stdout_text.contains("-sin(x)"),
        "symbolic limit workflow did not print derivative limit: {stdout_text:?}"
    );
}
