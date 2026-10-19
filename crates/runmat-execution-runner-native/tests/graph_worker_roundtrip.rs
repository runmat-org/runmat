//! Real stdio worker boundary for a selected V3 graph. The test binary is its
//! own execution worker so no installed CLI path or in-process worker stub is
//! needed.

use std::ffi::OsStr;
use std::rc::Rc;

use runmat_core::abi::{ExecutionRequest, HostExecutionPolicy, SourceInput};
use runmat_core::RunMatSession;
use runmat_execution_runner_native::{NativeExecutionConfig, NativeExecutionService};
use runmat_runtime::execution::{ExecutionServiceError, RuntimeExecutionServices};
use runmat_value::Value;

fn main() {
    if std::env::args_os().nth(1).as_deref()
        == Some(OsStr::new(
            runmat_process_host::HiddenMode::ExecutionWorker.marker(),
        ))
    {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("worker runtime")
            .block_on(runmat_execution_runner_native::run_worker_stdio())
            .expect("stdio execution worker");
        return;
    }
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
        .block_on(roundtrip());
}

async fn execute(session: &mut RunMatSession, name: &str, source: &str) {
    let request = ExecutionRequest::for_source(
        SourceInput::Text {
            name: name.into(),
            text: source.into(),
        },
        session.compat_mode(),
        HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    let response = session.execute_request(request).await;
    let outcome = response.result.expect("graph session execution");
    assert!(
        outcome
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.severity != runmat_core::abi::DiagnosticSeverity::Error),
        "unexpected graph diagnostics: {:?}",
        outcome.diagnostics
    );
}

async fn roundtrip() {
    let temporary = tempfile::tempdir().expect("isolated worker state");
    let mut config = NativeExecutionConfig::for_current_executable().expect("host inventory");
    config.executable = std::env::current_exe().expect("test executable");
    config.worker_arguments = vec![runmat_process_host::HiddenMode::ExecutionWorker
        .marker()
        .into()];
    config.store_root = temporary.path().join("native-session");
    config.max_workers = 1;
    let service = Rc::new(NativeExecutionService::new(config).expect("native worker service"));
    let mut session = RunMatSession::with_options(false, false).expect("session");
    session.install_execution_services(service.clone());

    execute(
        &mut session,
        "graph_submit.m",
        "pool = parpool(1);\nh = makeCounter();\nfuture = parfeval(pool, @returnHandle, 1, h);\nfunction h = makeCounter()\nshared = 40;\nfunction y = bump(delta)\nshared = shared + delta;\ny = shared;\nend\nh = @bump;\nend\nfunction out = returnHandle(input)\nout = input;\nend",
    )
    .await;

    let future_object = session
        .get_variables()
        .expect("session variables")
        .get("future")
        .expect("submitted future")
        .clone();
    let task = runmat_runtime::parallel::future::execution_value(&future_object)
        .expect("future object carries an execution handle")
        .clone();
    assert!(matches!(&task, Value::Task(_)));
    let foreign = runmat_runtime::context::RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    assert!(matches!(
        service.begin_await_in_context(task, &foreign),
        Err(ExecutionServiceError::ForeignScope)
    ));

    execute(
        &mut session,
        "graph_fetch.m",
        "first = fetchOutputs(future); second = fetchOutputs(future); firstValue = feval(first, 1); secondValue = feval(second, 1); clientValue = feval(h, 1);",
    )
    .await;
    let variables = session.get_variables().expect("fetched variables");
    assert_eq!(variables.get("firstValue"), Some(&Value::Num(41.0)));
    assert_eq!(variables.get("secondValue"), Some(&Value::Num(42.0)));
    assert_eq!(variables.get("clientValue"), Some(&Value::Num(41.0)));

    retained_constructorless_class_uses_selected_worker_products(&mut session, temporary.path())
        .await;
}

async fn retained_constructorless_class_uses_selected_worker_products(
    session: &mut RunMatSession,
    root: &std::path::Path,
) {
    let base_dir = root.join("base_revision");
    let derived_dir = root.join("derived_revision");
    std::fs::create_dir(&base_dir).expect("Base source directory");
    std::fs::create_dir(&derived_dir).expect("Derived source directory");
    let base_source = base_dir.join("RevisionBase.m");
    std::fs::write(
        &base_source,
        "classdef RevisionBase < dynamicprops\n properties\n seed = 11\n end\n methods\n function v = marker(obj)\n v = obj.seed + 100;\n end\n end\nend",
    )
    .expect("old constructorless Base source");
    std::fs::write(
        derived_dir.join("RevisionDerived.m"),
        "classdef RevisionDerived < RevisionBase\n methods\n function obj = RevisionDerived()\n obj@RevisionBase();\n obj.seed = obj.seed + 1;\n end\n end\nend",
    )
    .expect("Derived source");
    let base_path = base_dir.to_string_lossy().replace('\'', "''");
    let derived_path = derived_dir.to_string_lossy().replace('\'', "''");
    execute(
        session,
        "class_old.m",
        &format!(
            "addpath('{base_path}'); addpath('{derived_path}'); old_obj = RevisionDerived(); old_ctor = @RevisionDerived; old_seed = old_obj.seed; old_client_marker = old_obj.marker(); old_metadata = addprop(old_obj, 'tag'); old_obj.tag = 7; old_pair = {{old_obj, old_metadata}}; old_empty = old_obj([]);"
        ),
    )
    .await;
    assert_eq!(
        session.get_variables().unwrap().get("old_seed"),
        Some(&Value::Num(12.0))
    );
    assert_eq!(
        session.get_variables().unwrap().get("old_client_marker"),
        Some(&Value::Num(112.0))
    );
    std::fs::write(
        &base_source,
        "classdef RevisionBase < dynamicprops\n properties\n seed = 31\n end\n methods\n function v = marker(obj)\n v = obj.seed + 200;\n end\n end\nend",
    )
    .expect("new constructorless Base source");
    execute(
        session,
        "class_new.m",
        "new_obj = RevisionDerived(); new_ctor = @RevisionDerived; new_seed = new_obj.seed; new_client_marker = new_obj.marker(); old_client_marker_after = old_obj.marker(); new_empty = new_obj([]);",
    )
    .await;
    assert_eq!(
        session.get_variables().unwrap().get("new_seed"),
        Some(&Value::Num(32.0))
    );
    assert_eq!(
        session.get_variables().unwrap().get("new_client_marker"),
        Some(&Value::Num(232.0))
    );
    assert_eq!(
        session
            .get_variables()
            .unwrap()
            .get("old_client_marker_after"),
        Some(&Value::Num(112.0))
    );
    execute(
        session,
        "class_worker_submit.m",
        "oldFuture = parfeval(pool, @constructSeed, 1, old_ctor); newFuture = parfeval(pool, @constructSeed, 1, new_ctor); oldMethodFuture = parfeval(pool, @readMarker, 1, old_obj); newMethodFuture = parfeval(pool, @readMarker, 1, new_obj); oldEchoFuture = parfeval(pool, @echoObject, 1, old_obj); oldAliasFuture = parfeval(pool, @probeAlias, 1, old_obj, old_obj); oldPairFuture = parfeval(pool, @echoObject, 1, old_pair); oldEmptyFuture = parfeval(pool, @echoObject, 1, old_empty); newEmptyFuture = parfeval(pool, @echoObject, 1, new_empty);\nfunction seed = constructSeed(ctor)\n obj = feval(ctor);\n seed = obj.seed;\nend\nfunction marker = readMarker(obj)\n marker = obj.marker();\nend\nfunction obj = echoObject(input)\n obj = input;\nend\nfunction observed = probeAlias(left, right)\n left.seed = 99;\n observed = right.seed;\nend",
    )
    .await;
    execute(
        session,
        "class_worker_fetch.m",
        "old_worker_seed = fetchOutputs(oldFuture); new_worker_seed = fetchOutputs(newFuture); old_worker_marker = fetchOutputs(oldMethodFuture); new_worker_marker = fetchOutputs(newMethodFuture); old_returned = fetchOutputs(oldEchoFuture); old_returned_marker = old_returned.marker(); old_worker_alias = fetchOutputs(oldAliasFuture); returned_pair = fetchOutputs(oldPairFuture); returned_obj = returned_pair{1}; returned_metadata = returned_pair{2}; returned_findprop = findprop(returned_obj, 'tag'); returned_metadata.Hidden = true; returned_hidden = returned_findprop.Hidden; returned_obj.tag = 13; returned_tag = returned_obj.tag; returned_old_empty = fetchOutputs(oldEmptyFuture); returned_new_empty = fetchOutputs(newEmptyFuture);",
    )
    .await;
    let variables = session.get_variables().expect("scalar worker results");
    assert_eq!(variables.get("old_worker_seed"), Some(&Value::Num(12.0)));
    assert_eq!(variables.get("new_worker_seed"), Some(&Value::Num(32.0)));
    assert_eq!(variables.get("old_worker_marker"), Some(&Value::Num(112.0)));
    assert_eq!(variables.get("new_worker_marker"), Some(&Value::Num(232.0)));
    assert_eq!(
        variables.get("old_returned_marker"),
        Some(&Value::Num(112.0))
    );
    assert_eq!(variables.get("old_worker_alias"), Some(&Value::Num(99.0)));
    assert_eq!(variables.get("returned_hidden"), Some(&Value::Bool(true)));
    assert_eq!(variables.get("returned_tag"), Some(&Value::Num(13.0)));
    let (Some(Value::HandleObject(returned_metadata)), Some(Value::HandleObject(found_metadata))) = (
        variables.get("returned_metadata"),
        variables.get("returned_findprop"),
    ) else {
        panic!("returned dynamic property metadata is a handle");
    };
    assert_eq!(returned_metadata.target, found_metadata.target);
    for (original, returned) in [
        ("old_empty", "returned_old_empty"),
        ("new_empty", "returned_new_empty"),
    ] {
        let (Some(Value::ObjectArray(original)), Some(Value::ObjectArray(returned))) =
            (variables.get(original), variables.get(returned))
        else {
            panic!("selected zero-length object slice is an object array");
        };
        assert!(original.is_empty() && returned.is_empty());
        assert_eq!(original.shape(), returned.shape());
        assert_eq!(original.selected_provenance().len(), 1);
        assert_eq!(returned.selected_provenance().len(), 1);
        assert_eq!(
            original.selected_provenance()[0].constructor(),
            returned.selected_provenance()[0].constructor()
        );
    }
}
