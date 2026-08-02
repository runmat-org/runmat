#![cfg(target_arch = "wasm32")]

use js_sys::{Object, Promise, Reflect};
use runmat_test::descriptor::{
    ProcedureDescriptor, ProcedureKind, SourceDescriptor, SourceSpan, TestDescriptor,
};
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
use runmat_test::identity::{FixtureGroupId, SuiteId, TestId, TestIdentityInput};
use runmat_test::plan::{FixtureGroupPlan, SuitePlan, TestPlanBuilder};
use runmat_test::result::{AttemptResult, ResultState};
use runmat_test_runner::worker::WorkerExecution;
use serde::Serialize;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
async fn portable_coordinator_runs_through_the_javascript_backend_port() {
    let (plan, snapshot, test_id) = fixture();
    let input = to_js(&serde_json::json!({
        "plan": plan,
        "snapshot": snapshot,
        "options": {
            "isolation": "session",
            "jobs": 1,
            "reports": ["json"]
        }
    }));
    let backend = fake_backend(test_id);
    let output = runmat_wasm::run_tests(input, backend).await.unwrap();
    let value: serde_json::Value = serde_wasm_bindgen::from_value(output).unwrap();

    assert_eq!(value["result"]["state"]["disposition"], "passed");
    assert_eq!(value["result"]["tests"].as_array().unwrap().len(), 1);
    assert_eq!(value["reports"].as_array().unwrap().len(), 1);
    assert_eq!(value["isolation"], "session");
}

fn fake_backend(test_id: TestId) -> JsValue {
    let backend = Object::new();
    set_function(
        &backend,
        "capabilities",
        Closure::wrap(Box::new(move || {
            to_js(&serde_json::json!({
                "isolation": ["session", "none"],
                "maxWorkers": 2
            }))
        }) as Box<dyn Fn() -> JsValue>),
    );
    set_function(
        &backend,
        "spawn",
        Closure::wrap(Box::new(move |_input: JsValue| {
            Promise::resolve(&to_js(&serde_json::json!({"id": "fake-1"})))
        }) as Box<dyn Fn(JsValue) -> Promise>),
    );
    let completed_id = test_id.clone();
    set_function(
        &backend,
        "execute",
        Closure::wrap(Box::new(move |_session: JsValue, _request: JsValue| {
            let execution = WorkerExecution {
                result: AttemptResult {
                    test_id: completed_id.clone(),
                    attempt: 1,
                    state: ResultState::PASSED,
                    diagnostics: Vec::new(),
                    artifacts: Vec::new(),
                    output: String::new(),
                    abort_run: false,
                },
                events: Vec::new(),
            };
            Promise::resolve(&to_js(&execution))
        }) as Box<dyn Fn(JsValue, JsValue) -> Promise>),
    );
    for name in ["terminate", "shutdown"] {
        set_function(
            &backend,
            name,
            Closure::wrap(
                Box::new(move |_session: JsValue| Promise::resolve(&JsValue::UNDEFINED))
                    as Box<dyn Fn(JsValue) -> Promise>,
            ),
        );
    }
    set_function(
        &backend,
        "cancel",
        Closure::wrap(Box::new(move |_session: JsValue, _request: JsValue| {
            Promise::resolve(&JsValue::NULL)
        }) as Box<dyn Fn(JsValue, JsValue) -> Promise>),
    );
    set_function(
        &backend,
        "isCancelled",
        Closure::wrap(Box::new(move || JsValue::FALSE) as Box<dyn Fn() -> JsValue>),
    );
    set_function(
        &backend,
        "cancellationReason",
        Closure::wrap(Box::new(move || JsValue::UNDEFINED) as Box<dyn Fn() -> JsValue>),
    );
    set_function(
        &backend,
        "waitForCancellation",
        Closure::wrap(
            Box::new(move || Promise::new(&mut |_resolve, _reject| {})) as Box<dyn Fn() -> Promise>
        ),
    );
    backend.into()
}

fn set_function<T: ?Sized + wasm_bindgen::closure::WasmClosure + 'static>(
    backend: &Object,
    name: &str,
    function: Closure<T>,
) {
    Reflect::set(backend, &JsValue::from_str(name), function.as_ref()).unwrap();
    function.forget();
}

fn to_js(value: &impl Serialize) -> JsValue {
    value
        .serialize(&serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true))
        .unwrap()
}

fn fixture() -> (runmat_test::plan::TestPlan, FrozenTestRunSnapshot, TestId) {
    let snapshot = FrozenTestRunSnapshot::freeze(
        "graph",
        "source",
        1,
        1,
        "test",
        vec![SavedRunSource {
            owner_identity: "root".into(),
            relative_path: "tests/sample.m".into(),
            content: "%% sample\n".into(),
        }],
        Vec::new(),
    )
    .unwrap();
    let revision = snapshot.program_revision.clone();
    let suite_id = SuiteId::derive(&revision.canonical_identity(), "suite");
    let group_id = FixtureGroupId::derive(suite_id.as_str(), "group");
    let test_id = TestId::derive(&TestIdentityInput {
        owner_identity: "root",
        relative_source_identity: "tests/sample.m",
        semantic_scheme: "section",
        semantic_item_path: "sample",
        parameter_identity: "",
        fixture_identity: group_id.as_str(),
    });
    let test = TestDescriptor {
        id: test_id.clone(),
        suite_id: suite_id.clone(),
        fixture_group_id: group_id.clone(),
        display_name: "sample".into(),
        procedure: ProcedureDescriptor {
            semantic_path: "sample".into(),
            display_name: "sample".into(),
            kind: ProcedureKind::ScriptSection,
            source: SourceDescriptor {
                owner_identity: "root".into(),
                relative_path: "tests/sample.m".into(),
                semantic_path: "sample".into(),
                span: SourceSpan {
                    start_byte: 0,
                    end_byte: 10,
                    start_line: 1,
                    start_column: 1,
                    end_line: 1,
                    end_column: 10,
                },
            },
        },
        parameters: Vec::new(),
        tags: Vec::new(),
        requirements: Default::default(),
    };
    let plan = TestPlanBuilder::new(revision, "wasm-coordinator")
        .add_suite(SuitePlan {
            id: suite_id,
            display_name: "suite".into(),
            fixture_groups: vec![FixtureGroupPlan {
                id: group_id,
                fixtures: Vec::new(),
                tests: vec![test],
            }],
        })
        .build()
        .unwrap();
    (plan, snapshot, test_id)
}
