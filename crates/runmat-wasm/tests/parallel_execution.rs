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
