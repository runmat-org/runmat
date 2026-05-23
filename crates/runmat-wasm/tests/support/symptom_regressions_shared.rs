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
    serde_wasm_bindgen::from_value(
        runtime
            .execute(script.to_string())
            .await
            .expect("execute script"),
    )
    .expect("deserialize execution payload")
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
    let stdout_text = payload
        .stdout
        .iter()
        .filter(|entry| entry.stream == "stdout")
        .map(|entry| entry.text.as_str())
        .collect::<String>();
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
    let stdout_text = payload
        .stdout
        .iter()
        .filter(|entry| entry.stream == "stdout")
        .map(|entry| entry.text.as_str())
        .collect::<String>();
    assert!(
        !stdout_text.trim().is_empty(),
        "slice arithmetic produced empty stdout unexpectedly"
    );
}
