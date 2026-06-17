#![cfg(target_arch = "wasm32")]

use serde::Deserialize;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{Clamped, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

#[path = "support/symptom_regressions_shared.rs"]
mod shared;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuStatusPayload {
    requested: bool,
    active: bool,
    error: Option<String>,
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

fn create_canvas(width: u32, height: u32) -> HtmlCanvasElement {
    let document = web_sys::window()
        .expect("browser window")
        .document()
        .expect("browser document");
    let canvas = document
        .create_element("canvas")
        .expect("create canvas element")
        .dyn_into::<HtmlCanvasElement>()
        .expect("created element is a canvas");
    canvas.set_width(width);
    canvas.set_height(height);
    canvas
}

async fn wait_for_animation_frame() {
    let window = web_sys::window().expect("browser window");
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let callback = Closure::once_into_js(move |_timestamp: f64| {
            let _ = resolve.call0(&JsValue::UNDEFINED);
        });
        if let Err(err) = window.request_animation_frame(callback.unchecked_ref()) {
            let _ = reject.call1(&JsValue::UNDEFINED, &err);
        }
    });
    JsFuture::from(promise)
        .await
        .expect("wait for animation frame");
}

fn rendered_pixel_stats(canvas: &HtmlCanvasElement) -> (usize, usize) {
    let document = web_sys::window()
        .expect("browser window")
        .document()
        .expect("browser document");
    let readback_canvas = document
        .create_element("canvas")
        .expect("create readback canvas")
        .dyn_into::<HtmlCanvasElement>()
        .expect("created element is a canvas");
    readback_canvas.set_width(canvas.width());
    readback_canvas.set_height(canvas.height());
    let context = readback_canvas
        .get_context("2d")
        .expect("get 2d context result")
        .expect("2d context is available")
        .dyn_into::<CanvasRenderingContext2d>()
        .expect("context is 2d");
    context
        .draw_image_with_html_canvas_element(canvas, 0.0, 0.0)
        .expect("copy rendered canvas into readback canvas");
    let image_data = context
        .get_image_data(0.0, 0.0, canvas.width() as f64, canvas.height() as f64)
        .expect("read rendered pixels");
    let Clamped(data) = image_data.data();
    let nonzero_pixels = data
        .chunks_exact(4)
        .filter(|pixel| pixel.iter().any(|component| *component != 0))
        .count();
    let mut sampled_colors = std::collections::BTreeSet::new();
    for (idx, pixel) in data.chunks_exact(4).enumerate() {
        if idx % 64 == 0 {
            sampled_colors.insert([pixel[0], pixel[1], pixel[2], pixel[3]]);
        }
    }
    (nonzero_pixels, sampled_colors.len())
}

fn js_error_text(value: &JsValue) -> String {
    if let Some(message) = value.as_string() {
        return message;
    }
    if let Some(error) = value.dyn_ref::<js_sys::Error>() {
        return String::from(error.message());
    }
    js_sys::JSON::stringify(value)
        .ok()
        .and_then(|json| json.as_string())
        .unwrap_or_else(|| format!("{value:?}"))
}

fn is_canvas_webgpu_unavailable(message: &str) -> bool {
    message.contains("no compatible WebGPU adapter was found")
}

#[wasm_bindgen_test(async)]
async fn impedance_loop_executes_without_runtime_error() {
    shared::assert_impedance_loop_executes_without_runtime_error().await;
}

#[wasm_bindgen_test(async)]
async fn slice_end_arithmetic_executes_without_runtime_error() {
    shared::assert_slice_end_arithmetic_executes_without_runtime_error().await;
}

#[wasm_bindgen_test(async)]
async fn tic_toc_loop_executes_without_runtime_error() {
    shared::assert_tic_toc_loop_executes_without_runtime_error().await;
}

#[wasm_bindgen_test(async)]
async fn symbolic_limit_workflow_executes_without_runtime_error() {
    shared::assert_symbolic_limit_workflow_executes_without_runtime_error().await;
}

#[wasm_bindgen_test(async)]
async fn image_normalize_vecdim_workload_completes_with_webgpu() {
    let runtime = runmat_wasm::init_runmat(init_options(true))
        .await
        .expect("initialize wasm runtime");
    let gpu_status: GpuStatusPayload =
        serde_wasm_bindgen::from_value(runtime.gpu_status().expect("gpu status"))
            .expect("deserialize gpu status");
    if !gpu_status.active {
        web_sys::console::warn_1(
            &format!(
                "Skipping image normalize vecdim WebGPU check: requested={}, active={}, error={:?}",
                gpu_status.requested, gpu_status.active, gpu_status.error
            )
            .into(),
        );
        return;
    }

    let script = r#"
rng(0); B=2; H=4; W=5;
gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0=single(1e-6);

imgs = rand(B, H, W, 'single');
mu = mean(imgs, [2 3]);
sigma = sqrt(mean((imgs - mu).^2, [2 3]) + eps0);
out = ((imgs - mu) ./ sigma) * gain + bias;
out = out .^ gamma;
mse = mean((out - imgs).^2, 'all');

fprintf('RESULT_ok MSE=%.6e\n', double(mse));
"#;

    let payload = shared::execute_script_with_runtime(&runtime, script).await;
    if let Some(err) = payload.error {
        panic!(
            "image normalize vecdim wasm execution failed: {} (gpu requested={}, active={}, gpu error={:?})",
            err.message, gpu_status.requested, gpu_status.active, gpu_status.error
        );
    }
    let stdout_text = shared::stdout_text(&payload);
    assert!(
        stdout_text.contains("RESULT_ok MSE="),
        "image normalize vecdim workload produced unexpected stdout: {stdout_text:?}"
    );
}

#[wasm_bindgen_test(async)]
async fn image_normalize_vecdim_clamped_benchmark_shape_completes_with_webgpu() {
    let runtime = runmat_wasm::init_runmat(init_options(true))
        .await
        .expect("initialize wasm runtime");
    let gpu_status: GpuStatusPayload =
        serde_wasm_bindgen::from_value(runtime.gpu_status().expect("gpu status"))
            .expect("deserialize gpu status");
    if !gpu_status.active {
        web_sys::console::warn_1(
            &format!(
                "Skipping clamped image normalize vecdim WebGPU check: requested={}, active={}, error={:?}",
                gpu_status.requested, gpu_status.active, gpu_status.error
            )
            .into(),
        );
        return;
    }

    let script = r#"
rng(0); B=2; H=4; W=5;
gain=single(1.0123); bias=single(-0.02); gamma=single(1.8); eps0=single(1e-6);

imgs = rand(B, H, W, 'single');
mu = single(mean(imgs, [2 3], 'native'));
sigma = single(sqrt(mean((imgs - mu).^2, [2 3], 'native') + eps0));
out = single(((imgs - mu) ./ sigma) * gain + bias);
out = max(out, single(0));
out = single(out .^ gamma);
err = out - imgs;
mse = mean(err .* err, 'all');

fprintf('RESULT_ok MSE=%.6e\n', double(mse));
"#;

    let payload = shared::execute_script_with_runtime(&runtime, script).await;
    if let Some(err) = payload.error {
        panic!(
            "clamped image normalize vecdim wasm execution failed: {} (gpu requested={}, active={}, gpu error={:?})",
            err.message, gpu_status.requested, gpu_status.active, gpu_status.error
        );
    }
    let stdout_text = shared::stdout_text(&payload);
    assert!(
        stdout_text.contains("RESULT_ok MSE="),
        "clamped image normalize vecdim workload produced unexpected stdout: {stdout_text:?}"
    );
    assert!(
        !stdout_text.contains("MSE=NaN"),
        "clamped image normalize vecdim workload should produce finite stdout: {stdout_text:?}"
    );
}

#[wasm_bindgen_test(async)]
async fn surface_animation_loop_presents_without_runtime_error() {
    let runtime = runmat_wasm::init_runmat(init_options(true))
        .await
        .expect("initialize wasm runtime");
    let canvas = create_canvas(640, 480);
    let (blank_nonzero, blank_colors) = rendered_pixel_stats(&canvas);
    assert_eq!(blank_nonzero, 0, "new canvas should start transparent");
    assert_eq!(
        blank_colors, 1,
        "new canvas should have one transparent color"
    );
    let surface_id = match runmat_wasm::create_plot_surface(canvas.clone().into()).await {
        Ok(surface_id) => surface_id,
        Err(err) => {
            let message = js_error_text(&err);
            if is_canvas_webgpu_unavailable(&message) {
                web_sys::console::warn_1(
                    &format!(
                        "Skipping surface animation presentation check: browser has no canvas-compatible WebGPU adapter ({message})"
                    )
                    .into(),
                );
                return;
            }
            panic!("create plot surface: {message}");
        }
    };

    let script = r#"
XRange = -2:0.02:2;
YRange = -2:0.02:2;
T = 2/30;
FPS = 30;
dT = 1/FPS;
noise = 1.0;

[X, Y] = meshgrid(XRange, YRange);

for t = 0:dT:T
    R = sqrt(X.^2 + Y.^2);
    Z = sin(t*R) ./ R + rand(R) * noise;
    surf(X, Y, Z);
    pause(dT);
end

disp(1);
"#;

    let payload = shared::execute_script_with_runtime(&runtime, script).await;
    if let Some(err) = payload.error {
        panic!("surf animation loop wasm execution failed: {}", err.message);
    }
    let stdout_text = shared::stdout_text(&payload);
    assert!(
        stdout_text.split_whitespace().any(|token| token == "1"),
        "surf animation loop produced unexpected stdout: {stdout_text:?}"
    );
    runmat_wasm::present_surface(surface_id).expect("present bound plot surface");
    wait_for_animation_frame().await;
    let (nonzero_pixels, sampled_colors) = rendered_pixel_stats(&canvas);
    assert!(
        nonzero_pixels > (canvas.width() as usize * canvas.height() as usize) / 2,
        "surf animation rendered too few nonzero pixels: {nonzero_pixels}"
    );
    assert!(
        sampled_colors >= 8,
        "surf animation rendered insufficient color variation: {sampled_colors}"
    );
}
