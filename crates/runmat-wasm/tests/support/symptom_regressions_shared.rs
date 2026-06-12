#![cfg(target_arch = "wasm32")]

use runmat_wasm::init_runmat;
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
    options.into()
}

async fn execute_script(script: &str) -> ExecPayload {
    let runtime = init_runmat(init_options(false))
        .await
        .expect("initialize wasm runtime");
    let request = text_execute_request("symptom_regression.m", script);
    serde_wasm_bindgen::from_value(
        runtime
            .execute_request_js(request)
            .await
            .expect("execute script"),
    )
    .expect("deserialize execution payload")
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

fn stdout_text(payload: &ExecPayload) -> String {
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
