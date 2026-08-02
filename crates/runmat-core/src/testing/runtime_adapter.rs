use std::collections::{BTreeMap, HashMap};

use runmat_builtins::Value;
use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};

use crate::{CompatMode, RunMatSession};

pub(crate) fn services(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
) -> runmat_runtime::testing::RuntimeTestServices {
    let run_project = project.clone();
    let run = move |args: Vec<Value>| {
        let project = run_project.clone();
        Box::pin(async move { run_arguments(compat, project, args).await })
            as runmat_runtime::testing::RunSuiteFuture
    };
    let suite_run = run.clone();
    let discover_project = project;
    runmat_runtime::testing::RuntimeTestServices::new(
        move |suite| {
            let args = suite_targets(suite);
            suite_run(args)
        },
        run,
        move |args| {
            let project = discover_project.clone();
            Box::pin(async move { discover_arguments(compat, project, args).await })
        },
    )
}

async fn run_arguments(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
    args: Vec<Value>,
) -> runmat_runtime::BuiltinResult<Value> {
    let (snapshot, selected_names) = snapshot_from_arguments(project.as_ref(), args).await?;
    let mut session = configured_session(compat, project)?;
    let run = session
        .run_test_snapshot(
            &snapshot,
            &TestSelector {
                names: selected_names,
                ..TestSelector::default()
            },
        )
        .await
        .map_err(domain_error)?;
    result_value(&run)
}

async fn discover_arguments(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
    args: Vec<Value>,
) -> runmat_runtime::BuiltinResult<Value> {
    let (snapshot, selected_names) = snapshot_from_arguments(project.as_ref(), args).await?;
    let session = configured_session(compat, project)?;
    let discovery = session
        .discover_tests(&snapshot)
        .map_err(domain_error)?
        .select(&TestSelector {
            names: selected_names,
            ..TestSelector::default()
        });
    let values = discovery
        .suites
        .into_iter()
        .flat_map(|suite| suite.tests)
        .map(|test| {
            let mut value = runmat_runtime::testing::test_suite_object(&test);
            if let Value::Object(object) = &mut value {
                object.properties.insert(
                    "TestFile".into(),
                    Value::String(display_source_path(&test.procedure.source)),
                );
            }
            value
        })
        .collect();
    runmat_runtime::testing::object_array_or_scalar(
        runmat_runtime::testing::TEST_SUITE_CLASS,
        values,
    )
    .map_err(|error| runtime_error("RunMat:Testing:SuiteProjection", error))
}

async fn snapshot_from_arguments(
    project: Option<&runmat_package::FrozenProjectHandoff>,
    args: Vec<Value>,
) -> runmat_runtime::BuiltinResult<(FrozenTestRunSnapshot, Vec<String>)> {
    let resolved =
        runmat_runtime::builtins::diagnostics::runtests::resolve_runtests_targets(args).await?;
    let mut sources = BTreeMap::new();
    let mut selected_names = Vec::new();
    for case in resolved.targets {
        selected_names.push(case.name);
        let source = runmat_filesystem::read_to_string_async(&case.source_path)
            .await
            .map_err(|error| {
                runtime_error(
                    "RunMat:runtests:FileReadFailed",
                    format!("{} ({error})", case.source_path.display()),
                )
            })?;
        let file_name = case
            .source_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                runtime_error(
                    "RunMat:runtests:InvalidSourcePath",
                    format!("invalid test source path '{}'", case.source_path.display()),
                )
            })?
            .to_owned();
        let owner = case
            .source_path
            .parent()
            .map(|path| format!("path:{}", path.to_string_lossy()))
            .unwrap_or_else(|| "path:runtime".into());
        sources.insert(
            (owner.clone(), file_name.clone()),
            SavedRunSource {
                owner_identity: owner,
                relative_path: file_name,
                content: source,
            },
        );
    }

    let (graph_digest, source_digest) = project
        .map(|handoff| {
            let revision = handoff.revision();
            (
                revision.graph_digest.to_string(),
                revision.source_revision.to_string(),
            )
        })
        .unwrap_or_else(|| ("runtime-live".into(), "runtime-live".into()));
    let snapshot = FrozenTestRunSnapshot::freeze(
        graph_digest,
        source_digest,
        1,
        1,
        "runtime-default",
        sources.into_values().collect(),
        Vec::new(),
    )
    .map_err(domain_error)?;
    Ok((snapshot, selected_names))
}

fn configured_session(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
) -> runmat_runtime::BuiltinResult<RunMatSession> {
    let mut session = RunMatSession::with_options(false, false).map_err(|error| {
        runtime_error("RunMat:Testing:SessionInitialization", error.to_string())
    })?;
    session.set_compat_mode(compat);
    if let Some(project) = project {
        session
            .install_project_handoff(project)
            .map_err(|error| runtime_error("RunMat:Testing:ProjectHandoff", error.to_string()))?;
    }
    Ok(session)
}

fn result_value(run: &super::CoreTestRun) -> runmat_runtime::BuiltinResult<Value> {
    let tests = run
        .plan
        .tests()
        .map(|test| (test.id.clone(), test))
        .collect::<HashMap<_, _>>();
    let values = run
        .results
        .iter()
        .map(|result| {
            let descriptor = tests.get(&result.test_id).ok_or_else(|| {
                runtime_error(
                    "RunMat:Testing:ResultProjection",
                    "result has no matching test descriptor",
                )
            })?;
            let mut value =
                runmat_runtime::testing::test_result_object(&descriptor.display_name, result);
            if let Value::Object(object) = &mut value {
                object.properties.insert(
                    "TestFile".into(),
                    Value::String(display_source_path(&descriptor.procedure.source)),
                );
                object.properties.insert("Duration".into(), Value::Num(0.0));
                let details = result
                    .attempts
                    .last()
                    .and_then(|attempt| attempt.diagnostics.first())
                    .map(|diagnostic| diagnostic.message.clone())
                    .unwrap_or_default();
                object
                    .properties
                    .insert("Details".into(), Value::String(details));
            }
            Ok(value)
        })
        .collect::<runmat_runtime::BuiltinResult<Vec<_>>>()?;
    runmat_runtime::testing::object_array_or_scalar(
        runmat_runtime::testing::TEST_RESULT_CLASS,
        values,
    )
    .map_err(|error| runtime_error("RunMat:Testing:ResultProjection", error))
}

fn display_source_path(source: &runmat_test::descriptor::SourceDescriptor) -> String {
    if let Some(parent) = source.owner_identity.strip_prefix("path:") {
        return std::path::Path::new(parent)
            .join(&source.relative_path)
            .to_string_lossy()
            .into_owned();
    }
    source.relative_path.clone()
}

fn suite_targets(suite: Value) -> Vec<Value> {
    let objects = match suite {
        Value::Object(object) => vec![Value::Object(object)],
        Value::ObjectArray(array) => array.data().to_vec(),
        value => return vec![value],
    };
    let mut files = Vec::new();
    let mut procedures = Vec::new();
    for value in objects {
        let Value::Object(object) = value else {
            continue;
        };
        if let Some(file) = object.properties.get("TestFile") {
            files.push(file.clone());
        }
        if let Some(procedure) = object.properties.get("ProcedureName") {
            procedures.push(procedure.clone());
        }
    }
    let mut args = vec![cell_value(files)];
    if !procedures.is_empty() {
        args.push(Value::String("Name".into()));
        args.push(cell_value(procedures));
    }
    args
}

fn cell_value(values: Vec<Value>) -> Value {
    let count = values.len();
    Value::Cell(
        runmat_builtins::CellArray::new(values, 1, count)
            .expect("runtime test target rows have valid dimensions"),
    )
}

fn domain_error(error: runmat_test::TestDomainError) -> runmat_runtime::RuntimeError {
    runtime_error("RunMat:Testing:Domain", error.to_string())
}

fn runtime_error(
    identifier: &'static str,
    message: impl Into<String>,
) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(message)
        .with_identifier(identifier)
        .build()
}
