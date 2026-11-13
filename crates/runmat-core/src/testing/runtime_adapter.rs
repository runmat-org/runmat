use std::collections::{BTreeMap, HashMap};

use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
use runmat_value::Value;

use crate::{CompatMode, RunMatSession};

pub(crate) fn services(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
    #[cfg(not(target_arch = "wasm32"))]
    child_tasks: runmat_runtime::context::RuntimeLocalTaskChildFactory,
    #[cfg(not(target_arch = "wasm32"))] parent_mex_lane: Option<
        runmat_runtime::foreign::MexNativeLaneChildFactory,
    >,
) -> runmat_runtime::testing::RuntimeTestServices {
    let run_project = project.clone();
    #[cfg(not(target_arch = "wasm32"))]
    let run_child_tasks = child_tasks.clone();
    #[cfg(not(target_arch = "wasm32"))]
    let run_parent_mex_lane = parent_mex_lane.clone();
    let run = move |caller: Option<runmat_runtime::context::RuntimeContext>,
                    args: runmat_runtime::call::BuiltinArguments,
                    coverage: Option<MatlabCoverageSelection>| {
        let project = run_project.clone();
        #[cfg(not(target_arch = "wasm32"))]
        let child_tasks = run_child_tasks.clone();
        #[cfg(not(target_arch = "wasm32"))]
        let parent_mex_lane = run_parent_mex_lane.clone();
        run_arguments_continuation(
            compat,
            project,
            caller,
            args,
            coverage,
            #[cfg(not(target_arch = "wasm32"))]
            child_tasks,
            #[cfg(not(target_arch = "wasm32"))]
            parent_mex_lane,
        )
    };
    let suite_run = run.clone();
    let discover_project = project;
    #[cfg(not(target_arch = "wasm32"))]
    let discover_child_tasks = child_tasks;
    #[cfg(not(target_arch = "wasm32"))]
    let discover_parent_mex_lane = parent_mex_lane;
    runmat_runtime::testing::RuntimeTestServices::new(
        move |caller, suite, plugins| {
            let args = suite_targets(suite);
            let coverage = plugins.iter().find_map(coverage_plugin_selection);
            let args =
                match runmat_runtime::call::BuiltinArguments::from_borrowed(&args, caller.as_ref())
                {
                    Ok(args) => args,
                    Err(error) => return Box::pin(async move { Err(error) }),
                };
            suite_run(caller, args, coverage)
        },
        move |caller, args| run(caller, args, None),
        move |caller, args| {
            let project = discover_project.clone();
            #[cfg(not(target_arch = "wasm32"))]
            let child_tasks = discover_child_tasks.clone();
            #[cfg(not(target_arch = "wasm32"))]
            let parent_mex_lane = discover_parent_mex_lane.clone();
            Box::pin(discover_arguments(
                compat,
                project,
                caller,
                args,
                #[cfg(not(target_arch = "wasm32"))]
                child_tasks,
                #[cfg(not(target_arch = "wasm32"))]
                parent_mex_lane,
            ))
        },
    )
}

/// The service request owns its inputs and the complete child session. Queue
/// it on the caller's already-active driver before entering another lifecycle
/// or interpreter, rather than merely boxing a still-nested borrowed poll.
fn run_arguments_continuation(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
    caller: Option<runmat_runtime::context::RuntimeContext>,
    args: runmat_runtime::call::BuiltinArguments,
    coverage: Option<MatlabCoverageSelection>,
    #[cfg(not(target_arch = "wasm32"))]
    child_tasks: runmat_runtime::context::RuntimeLocalTaskChildFactory,
    #[cfg(not(target_arch = "wasm32"))] parent_mex_lane: Option<
        runmat_runtime::foreign::MexNativeLaneChildFactory,
    >,
) -> runmat_runtime::testing::RunSuiteFuture {
    let Some(runtime) = caller.as_ref().cloned() else {
        // Preserve the existing context-free error/argument-validation path;
        // there is no active parent supervisor to participate in here.
        return Box::pin(run_arguments(
            compat,
            project,
            caller,
            args,
            coverage,
            #[cfg(not(target_arch = "wasm32"))]
            child_tasks,
            #[cfg(not(target_arch = "wasm32"))]
            parent_mex_lane,
        ));
    };
    let publication = runtime.clone();
    // Factory admission accounts for the concrete owned request before it is
    // constructed. Its completion stays rooted across sibling scheduling.
    let admitted =
        runmat_runtime::execution::admit_rooted_result_with(&runtime, move || async move {
            let result = run_arguments(
                compat,
                project,
                caller,
                args,
                coverage,
                #[cfg(not(target_arch = "wasm32"))]
                child_tasks,
                #[cfg(not(target_arch = "wasm32"))]
                parent_mex_lane,
            )
            .await?;
            runmat_runtime::execution::RootedValue::in_context(result, &publication)
        });
    Box::pin(async move {
        let rooted =
            runmat_runtime::context::run_owned_request_continuation(&runtime, admitted?).await?;
        Ok(rooted.with_owned(|value| value))
    })
}

async fn run_arguments(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
    caller: Option<runmat_runtime::context::RuntimeContext>,
    args: runmat_runtime::call::BuiltinArguments,
    coverage: Option<MatlabCoverageSelection>,
    #[cfg(not(target_arch = "wasm32"))]
    child_tasks: runmat_runtime::context::RuntimeLocalTaskChildFactory,
    #[cfg(not(target_arch = "wasm32"))] parent_mex_lane: Option<
        runmat_runtime::foreign::MexNativeLaneChildFactory,
    >,
) -> runmat_runtime::BuiltinResult<Value> {
    let (snapshot, selected_names, requested_coverage) =
        snapshot_from_arguments(compat, project.as_ref(), args, caller.as_ref()).await?;
    let parent = caller.as_ref().map(|runtime| runtime.cancellation());
    let mut session = construct_child_session(
        compat,
        caller,
        #[cfg(not(target_arch = "wasm32"))]
        child_tasks,
        #[cfg(not(target_arch = "wasm32"))]
        parent_mex_lane,
    )?;
    // The child owns the outer continuation receipt before test procedures
    // create nested invocations. Retire only after report publication.
    let _admission = session.admit_execution(parent.as_ref());
    if let Some(project) = project {
        #[cfg(not(target_arch = "wasm32"))]
        install_child_project(&mut session, project)?.await?;
        #[cfg(target_arch = "wasm32")]
        session
            .install_project_handoff(project)
            .map_err(|error| runtime_error("RunMat:Testing:ProjectHandoff", error.to_string()))?;
    }
    let selector = TestSelector {
        names: selected_names,
        ..TestSelector::default()
    };
    let runtime = session
        .runtime_context()
        .clone()
        .with_cancellation_token(session.runtime_context().cancellation());
    let test_session = &mut session;
    let test_snapshot = &snapshot;
    let test_selector = &selector;
    let mut completed_run = None;
    let completed_report = &mut completed_run;
    // The nested lifecycle/child-execution future is large. Construct its
    // one exact-admitted continuation inside the admission boundary, after
    // the caller's poll frame has passed only these small references. A
    // by-value future argument would still reserve its large temporary in
    // this frame during every nested VM poll. Core owns the metadata report;
    // the shared continuation boundary returns only unit, not raw Values.
    runmat_runtime::execution::admit_domain_continuation_with(&runtime, move || async move {
        *completed_report = Some(
            test_session
                .run_test_snapshot(test_snapshot, test_selector)
                .await
                .map_err(domain_error)?,
        );
        Ok(())
    })?
    .await?;
    let run = completed_run.expect("successful test continuation produces its report");
    let coverage = coverage.or_else(|| requested_coverage.then(MatlabCoverageSelection::default));
    result_value(&runtime, &run, coverage.as_ref())
}

async fn discover_arguments(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
    caller: Option<runmat_runtime::context::RuntimeContext>,
    args: runmat_runtime::call::BuiltinArguments,
    #[cfg(not(target_arch = "wasm32"))]
    child_tasks: runmat_runtime::context::RuntimeLocalTaskChildFactory,
    #[cfg(not(target_arch = "wasm32"))] parent_mex_lane: Option<
        runmat_runtime::foreign::MexNativeLaneChildFactory,
    >,
) -> runmat_runtime::BuiltinResult<Value> {
    let (snapshot, selected_names, _) =
        snapshot_from_arguments(compat, project.as_ref(), args, caller.as_ref())
            .await
            .map_err(|source| {
                let identifier = source
                    .identifier()
                    .map(|value| value.replacen("RunMat:runtests:", "RunMat:testsuite:", 1));
                let mut error = runmat_runtime::build_runtime_error(source.message().replacen(
                    "runtests:",
                    "testsuite:",
                    1,
                ))
                .with_builtin("testsuite")
                .with_source(source);
                if let Some(identifier) = identifier {
                    error = error.with_identifier(identifier);
                }
                error.build()
            })?;
    let parent = caller.as_ref().map(|runtime| runtime.cancellation());
    let mut session = construct_child_session(
        compat,
        caller,
        #[cfg(not(target_arch = "wasm32"))]
        child_tasks,
        #[cfg(not(target_arch = "wasm32"))]
        parent_mex_lane,
    )?;
    let _admission = session.admit_execution(parent.as_ref());
    if let Some(project) = project {
        #[cfg(not(target_arch = "wasm32"))]
        install_child_project(&mut session, project)?.await?;
        #[cfg(target_arch = "wasm32")]
        session
            .install_project_handoff(project)
            .map_err(|error| runtime_error("RunMat:Testing:ProjectHandoff", error.to_string()))?;
    }
    let discovery = session
        .discover_tests(&snapshot)
        .map_err(domain_error)?
        .select(&TestSelector {
            names: selected_names,
            ..TestSelector::default()
        });
    let runtime = session.runtime_context();
    let count = discovery.suites.iter().map(|suite| suite.tests.len()).sum();
    let mut tests = discovery.suites.into_iter().flat_map(|suite| suite.tests);
    // These projected objects contain text/scalar children only; there are no
    // unregistered GC identities between initializer calls.
    let values =
        runmat_runtime::execution::RootedValues::try_initialize_in_context(count, runtime, |_| {
            let test = tests.next().expect("discovery count matches its tests");
            let value = runmat_runtime::testing::test_suite_object_in_context(runtime, &test)?;
            let file = display_source_path(&test.procedure.source);
            let updates = [("TestFile", projected_text(runtime, &file)?)];
            runmat_runtime::testing::replace_object_properties_in_context(
                runtime,
                value,
                updates.iter().map(|(key, value)| (*key, value)),
            )
            .map(|value| value.with_owned(|value| value))
        })?;
    runmat_runtime::testing::object_array_or_scalar_in_context(
        runtime,
        runmat_runtime::testing::TEST_SUITE_CLASS,
        values,
    )
    .map(|value| value.with_owned(|value| value))
}

async fn snapshot_from_arguments(
    compat: CompatMode,
    project: Option<&runmat_package::FrozenProjectHandoff>,
    args: runmat_runtime::call::BuiltinArguments,
    runtime: Option<&runmat_runtime::context::RuntimeContext>,
) -> runmat_runtime::BuiltinResult<(FrozenTestRunSnapshot, Vec<String>, bool)> {
    let resolved =
        runmat_runtime::builtins::diagnostics::runtests::resolve_runtests_targets(args, runtime)
            .await?;
    let mut sources = BTreeMap::new();
    let selected_names = resolved.filters;
    let coverage = resolved.coverage;
    for case in resolved.targets {
        // Runtime read each selected source once. Freeze those exact bytes;
        // semantic discovery and execution must not reopen a mutable path.
        let source = case.source;
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
        .unwrap_or_else(|| {
            (
                runmat_execution::Digest::sha256(b"runtime-live").to_string(),
                "runtime-live".into(),
            )
        });
    let snapshot = FrozenTestRunSnapshot::freeze(
        graph_digest,
        source_digest,
        crate::program_environment(compat),
        runmat_execution::Digest::sha256(b"runtime-default").to_string(),
        sources.into_values().collect(),
        Vec::new(),
    )
    .map_err(domain_error)?;
    Ok((snapshot, selected_names, coverage))
}

/// Construct the selected project-install future inside its admitted owner.
/// Keeping this cold branch out of the test driver's inline state avoids
/// reserving its preparation temporaries during every nested test poll.
#[cfg(not(target_arch = "wasm32"))]
fn install_child_project<'a>(
    session: &'a mut RunMatSession,
    project: runmat_package::FrozenProjectHandoff,
) -> runmat_runtime::BuiltinResult<runmat_runtime::execution::RuntimeContinuationFuture<'a, ()>> {
    let runtime = session
        .runtime_context()
        .clone()
        .with_cancellation_token(session.runtime_context().cancellation());
    runmat_runtime::execution::admit_domain_continuation_with(&runtime, move || async move {
        session
            .install_project_handoff(project)
            .await
            .map(|_| ())
            .map_err(|error| runtime_error("RunMat:Testing:ProjectHandoff", error.to_string()))
    })
}

fn construct_child_session(
    compat: CompatMode,
    caller: Option<runmat_runtime::context::RuntimeContext>,
    #[cfg(not(target_arch = "wasm32"))]
    child_tasks: runmat_runtime::context::RuntimeLocalTaskChildFactory,
    #[cfg(not(target_arch = "wasm32"))] parent_mex_lane: Option<
        runmat_runtime::foreign::MexNativeLaneChildFactory,
    >,
) -> runmat_runtime::BuiltinResult<RunMatSession> {
    let caller = caller.ok_or_else(|| {
        runtime_error(
            "RunMat:Testing:RequiresExecutor",
            "test child requires the originating runtime context",
        )
    })?;
    #[cfg(not(target_arch = "wasm32"))]
    let mut session = RunMatSession::with_options_on_parent_driver_with_mex(
        false,
        false,
        &child_tasks,
        &caller,
        parent_mex_lane,
    )
    .map_err(|error| runtime_error("RunMat:Testing:SessionInitialization", error.to_string()))?;
    #[cfg(target_arch = "wasm32")]
    let mut session =
        RunMatSession::with_options_for_test_child(false, false, &caller).map_err(|error| {
            runtime_error("RunMat:Testing:SessionInitialization", error.to_string())
        })?;
    session.set_compat_mode(compat);
    Ok(session)
}

fn result_value(
    runtime: &runmat_runtime::context::RuntimeContext,
    run: &super::CoreTestRun,
    coverage_selection: Option<&MatlabCoverageSelection>,
) -> runmat_runtime::BuiltinResult<Value> {
    let coverage = if let Some(selection) = coverage_selection {
        let aggregate = runmat_test::coverage::merge_coverage(run.coverage.clone())
            .map_err(|error| runtime_error("RunMat:Testing:Coverage", error.to_string()))?;
        Some(filter_matlab_coverage(aggregate, selection))
    } else {
        None
    };
    let tests = run
        .plan
        .tests()
        .map(|test| (test.id.clone(), test))
        .collect::<HashMap<_, _>>();
    let values = runmat_runtime::execution::RootedValues::try_initialize_in_context(
        run.results.len(),
        runtime,
        |index| {
            let result = &run.results[index];
            let descriptor = tests.get(&result.test_id).ok_or_else(|| {
                runtime_error(
                    "RunMat:Testing:ResultProjection",
                    "result has no matching test descriptor",
                )
            })?;
            let value = runmat_runtime::testing::test_result_object_in_context(
                runtime,
                &descriptor.display_name,
                result,
            )?;
            let file = display_source_path(&descriptor.procedure.source);
            let details = result
                .attempts
                .last()
                .and_then(|attempt| attempt.diagnostics.first())
                .map(|diagnostic| diagnostic.message.as_str())
                .unwrap_or_default();
            let updates = [
                ("TestFile", projected_text(runtime, &file)?),
                ("Duration", Value::Num(0.0)),
                ("Details", projected_text(runtime, details)?),
            ];
            let value = runmat_runtime::testing::replace_object_properties_in_context(
                runtime,
                value,
                updates.iter().map(|(key, value)| (*key, value)),
            )?;
            let value = if let Some(coverage) = &coverage {
                let statements = coverage.summary(runmat_test::coverage::CoverageMetric::Statement);
                let functions = coverage.summary(runmat_test::coverage::CoverageMetric::Function);
                let fields = [
                    ("StatementsCovered", Value::Num(statements.covered as f64)),
                    (
                        "StatementsTotal",
                        Value::Num(statements.instrumented as f64),
                    ),
                    ("FunctionsCovered", Value::Num(functions.covered as f64)),
                    ("FunctionsTotal", Value::Num(functions.instrumented as f64)),
                ];
                let summary = runmat_runtime::testing::object_from_fields_in_context(
                    runtime,
                    "RunMatCoverageResult",
                    fields.iter().map(|(key, value)| (*key, value)),
                )?;
                let updates = [("Coverage", summary.value().clone())];
                runmat_runtime::testing::replace_object_properties_in_context(
                    runtime,
                    value,
                    updates.iter().map(|(key, value)| (*key, value)),
                )?
            } else {
                value
            };
            // This report schema has no GC edges; exact object/text owners move
            // into the final admitted result slots before their roots retire.
            Ok::<_, runmat_runtime::RuntimeError>(value.with_owned(|value| value))
        },
    )?;
    runmat_runtime::testing::object_array_or_scalar_in_context(
        runtime,
        runmat_runtime::testing::TEST_RESULT_CLASS,
        values,
    )
    .map(|value| value.with_owned(|value| value))
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct MatlabCoverageSelection {
    folders: Vec<String>,
    including_subfolders: bool,
}

fn coverage_plugin_selection(value: &Value) -> Option<MatlabCoverageSelection> {
    let Value::Object(object) = value else {
        return None;
    };
    if !object
        .class_name()
        .is(runmat_types::standard::UNIT_TEST_CODE_COVERAGE_PLUGIN)
    {
        return None;
    }
    let folders = match object.properties().get("Folders") {
        Some(Value::StringArray(folders)) => folders
            .data()
            .iter()
            .map(|text| text.as_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>()?,
        _ => Vec::new(),
    };
    let including_subfolders = matches!(
        object.properties().get("IncludingSubfolders"),
        Some(Value::Bool(true))
    );
    Some(MatlabCoverageSelection {
        folders,
        including_subfolders,
    })
}

fn filter_matlab_coverage(
    mut coverage: runmat_test::coverage::CoverageAggregate,
    selection: &MatlabCoverageSelection,
) -> runmat_test::coverage::CoverageAggregate {
    if selection.folders.is_empty() {
        return coverage;
    }
    coverage.sites.retain(|site| selection.includes(site));
    let retained = coverage
        .sites
        .iter()
        .map(|site| site.id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    coverage
        .counts
        .retain(|site_id, _| retained.contains(site_id.as_str()));
    coverage
}

impl MatlabCoverageSelection {
    fn includes(&self, site: &runmat_test::coverage::CoverageSite) -> bool {
        let relative = normalize_coverage_path(&site.relative_path);
        let mut candidates = vec![relative.clone()];
        if let Some(owner) = site.owner_identity.strip_prefix("path:") {
            let owner = normalize_coverage_path(owner);
            candidates.push(if owner.is_empty() {
                relative
            } else {
                format!("{owner}/{relative}")
            });
        }
        self.folders.iter().any(|folder| {
            let root = normalize_coverage_path(folder);
            candidates.iter().any(|candidate| {
                let parent = candidate.rsplit_once('/').map_or("", |(parent, _)| parent);
                if self.including_subfolders {
                    parent == root || parent.starts_with(&(root.clone() + "/"))
                } else {
                    parent == root
                }
            })
        })
    }
}

fn normalize_coverage_path(path: &str) -> String {
    path.replace('\\', "/")
        .trim_start_matches("./")
        .trim_end_matches('/')
        .to_owned()
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
        if let Some(file) = object.properties().get("TestFile") {
            files.push(file.clone());
        }
        if let Some(procedure) = object.properties().get("ProcedureName") {
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
        runmat_value::CellArray::new(values, 1, count)
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

fn projected_text(
    runtime: &runmat_runtime::context::RuntimeContext,
    source: &str,
) -> runmat_runtime::BuiltinResult<Value> {
    runmat_runtime::call::argument_text::copy_str(source, Some(runtime))
        .map(|text| Value::String(text.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_test::coverage::{CoverageAggregate, CoverageMetric, CoverageSite};

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn owned_runtime_test_request_yields_before_loading_child_source() {
        use runmat_runtime::context::{
            RuntimeContext, RuntimeLocalTaskSupervisor, RuntimeServicePorts,
        };
        use std::{rc::Rc, sync::Arc, task::Poll};
        let _serial = runmat_filesystem::provider_override_lock();
        let _filesystem = runmat_filesystem::replace_provider(Arc::new(
            runmat_filesystem::MemoryFsProvider::with_current_dir("/suite"),
        ));
        let supervisor = RuntimeLocalTaskSupervisor::new();
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let ports = RuntimeServicePorts::default()
            .with_local_tasks(supervisor.submission_port(runtime.local_identity()));
        let runtime = runtime.with_service_ports(ports);
        let request = || {
            run_arguments_continuation(
                CompatMode::RunMat,
                None,
                Some(runtime.clone()),
                runmat_runtime::call::BuiltinArguments::from_borrowed(
                    &[Value::String("/suite/missingTests.m".into())],
                    Some(&runtime),
                )
                .unwrap(),
                None,
                supervisor.child_scope_factory(),
                None,
            )
        };
        // Idle-host behavior remains direct and supplies the independent
        // original source error for the active-driver request below.
        let host = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let direct = host.block_on(request()).unwrap_err();
        let queued = host
            .block_on(supervisor.drive_until(async {
                let mut future = request();
                let first =
                    std::future::poll_fn(|context| Poll::Ready(future.as_mut().poll(context)))
                        .await;
                assert!(
                    first.is_pending(),
                    "child source/lifecycle must not poll on the parent's request stack"
                );
                future.await
            }))
            .unwrap()
            .unwrap_err();
        assert_eq!(queued.identifier, direct.identifier);
        assert_eq!(queued.message(), direct.message());
        assert_eq!(queued.failure_origin, direct.failure_origin);
    }

    #[test]
    fn runtests_freezes_original_source_and_leaves_procedure_selection_to_core() {
        use runmat_filesystem::FsProvider;
        use std::path::Path;
        use std::sync::Arc;
        let _serial = runmat_filesystem::provider_override_lock();
        let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
            "/suite",
        ));
        let source = "function helper()\nend\nfunction testAlpha()\n assert(true);\nend\nfunction betaTest()\n assert(true);\nend\n";
        futures::executor::block_on(
            provider.write(Path::new("/suite/functionTests.m"), source.as_bytes()),
        )
        .unwrap();
        let _selected = runmat_filesystem::replace_provider(provider.clone());
        let args = runmat_runtime::call::BuiltinArguments::from_borrowed(
            &[
                Value::String("/suite/functionTests.m".into()),
                Value::String("Name".into()),
                Value::String("Alpha".into()),
            ],
            None,
        )
        .unwrap();
        let (snapshot, filters, _) = futures::executor::block_on(snapshot_from_arguments(
            CompatMode::RunMat,
            None,
            args,
            None,
        ))
        .unwrap();
        assert_eq!(snapshot.sources.len(), 1);
        assert_eq!(snapshot.sources[0].content, source);
        assert_eq!(filters, vec!["Alpha"]);
        futures::executor::block_on(provider.write(
            Path::new("/suite/functionTests.m"),
            b"error('source replaced');",
        ))
        .unwrap();
        assert_eq!(
            snapshot.sources[0].content, source,
            "freeze must not reopen the source"
        );
    }

    fn site(id: &str, owner: &str, path: &str) -> CoverageSite {
        CoverageSite {
            id: id.into(),
            counter_key: id.len() as u64,
            metric: CoverageMetric::Statement,
            owner_identity: owner.into(),
            relative_path: path.into(),
            semantic_path: "covered".into(),
            source_id: 0,
            start_byte: 0,
            end_byte: 1,
            start_line: 1,
            start_column: 1,
            end_line: 1,
            end_column: 2,
            instrumented: true,
            unsupported_reason: None,
        }
    }

    #[test]
    fn matlab_coverage_folder_policy_controls_projected_sites() {
        let immediate = site("immediate", "path:/workspace/src", "covered.m");
        let nested = site("nested", "path:/workspace/src/nested", "other.m");
        let unrelated = site("unrelated", "path:/workspace/tests", "coveredTest.m");
        let coverage = CoverageAggregate {
            program_revision: Some("program".into()),
            sites: vec![immediate.clone(), nested.clone(), unrelated],
            counts: BTreeMap::from([
                (immediate.id.clone(), 1),
                (nested.id.clone(), 1),
                ("unrelated".into(), 1),
            ]),
        };

        let immediate_only = filter_matlab_coverage(
            coverage.clone(),
            &MatlabCoverageSelection {
                folders: vec!["/workspace/src".into()],
                including_subfolders: false,
            },
        );
        assert_eq!(immediate_only.sites, vec![immediate]);
        assert_eq!(immediate_only.counts.len(), 1);

        let recursive = filter_matlab_coverage(
            coverage,
            &MatlabCoverageSelection {
                folders: vec!["/workspace/src".into()],
                including_subfolders: true,
            },
        );
        assert_eq!(
            recursive.sites,
            vec![
                site("immediate", "path:/workspace/src", "covered.m"),
                nested
            ]
        );
        assert_eq!(recursive.counts.len(), 2);
    }
}
