use std::collections::{BTreeMap, HashMap};

use runmat_builtins::Value;
use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};

use crate::{CompatMode, RunMatSession};

pub(crate) fn services(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
) -> runmat_runtime::testing::RuntimeTestServices {
    let run = move |args: Vec<Value>| {
        let project = project.clone();
        Box::pin(async move { run_arguments(compat, project, args).await })
            as runmat_runtime::testing::RunSuiteFuture
    };
    let suite_run = run.clone();
    runmat_runtime::testing::RuntimeTestServices::new(
        move |suite| {
            let args = suite_targets(suite);
            suite_run(args)
        },
        run,
    )
}

async fn run_arguments(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
    args: Vec<Value>,
) -> runmat_runtime::BuiltinResult<Value> {
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
        .as_ref()
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
    let mut session = RunMatSession::with_options(false, false).map_err(|error| {
        runtime_error("RunMat:Testing:SessionInitialization", error.to_string())
    })?;
    session.set_compat_mode(compat);
    if let Some(project) = project {
        session
            .install_project_handoff(project)
            .map_err(|error| runtime_error("RunMat:Testing:ProjectHandoff", error.to_string()))?;
    }
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
    match suite {
        Value::Object(object) => object
            .properties
            .get("TestFile")
            .cloned()
            .into_iter()
            .collect(),
        Value::ObjectArray(array) => array
            .data()
            .iter()
            .filter_map(|value| match value {
                Value::Object(object) => object.properties.get("TestFile").cloned(),
                _ => None,
            })
            .collect(),
        value => vec![value],
    }
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
