#![cfg(target_arch = "wasm32")]

use js_sys::{Object, Reflect};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

#[path = "support/browser_parallel.rs"]
mod browser_parallel;
use browser_parallel::{execution_host, source_request};

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn browser_scheduler_executes_compiler_bound_parfor_regions() {
    let (host, _closures) = execution_host();
    let options = Object::new();
    Reflect::set(&options, &JsValue::from_str("executionHost"), &host).unwrap();
    Reflect::set(&options, &JsValue::from_str("enableGpu"), &JsValue::FALSE).unwrap();
    let runtime = runmat_wasm::init_runmat(options.into()).await.unwrap();
    let source = r#"
pool = parpool(2);
values = zeros(1, 2);
task_seen = zeros(1, 2);
worker_seen = zeros(1, 2);
parallel_random = zeros(1, 2);
total = 0;
rng(41);
parfor (index = 1:2, 2)
  values(index) = index * 2;
  parallel_random(index) = rand();
  task = getCurrentTask();
  worker = getCurrentWorker();
  task_seen(index) = task.ID == task.ID;
  worker_seen(index) = worker.ID == worker.ID;
  total = total + index;
end
rng(41);
serial_random = zeros(1, 2);
for index = 1:2
  serial_random(index) = rand();
end
if sum(values) ~= 6 || total ~= 3 || sum(task_seen) ~= 2 || sum(worker_seen) ~= 2 || any(parallel_random ~= serial_random)
  error("RunMat:parallel:BrowserDifferential", "browser parfor result mismatch");
end
answer = pool.NumWorkers;
"#;
    runtime
        .execute_request_js(source_request("browser-parfor.m", source, 1))
        .await
        .expect("browser parfor executes through the worker host boundary");
}

#[wasm_bindgen_test]
async fn browser_runtime_executes_typed_spmd_cooperatively() {
    let (host, _closures) = execution_host();
    let options = Object::new();
    Reflect::set(&options, &JsValue::from_str("executionHost"), &host).unwrap();
    Reflect::set(&options, &JsValue::from_str("enableGpu"), &JsValue::FALSE).unwrap();
    let runtime = runmat_wasm::init_runmat(options.into()).await.unwrap();
    let source = r#"
pool = parpool(2);
spmd(2)
  exact = uint64(0x0020000000000001u64) + uint64(spmdIndex);
  total = spmdPlus(uint16(spmdIndex));
end
values = exact{:};
sums = total{[1, 2]};
if values(1) ~= uint64(0x0020000000000002u64) || values(2) ~= uint64(0x0020000000000003u64)
  error("RunMat:parallel:BrowserSpmdValue", "browser SPMD lost an exact rank value");
end
if sums(1) ~= uint16(3) || sums(2) ~= uint16(3)
  error("RunMat:parallel:BrowserSpmdCollective", "browser SPMD collective result mismatch");
end
answer = pool.NumWorkers;
"#;
    runtime
        .execute_request_js(source_request("browser-spmd.m", source, 1))
        .await
        .expect("browser SPMD executes through the cooperative typed runtime");
}
