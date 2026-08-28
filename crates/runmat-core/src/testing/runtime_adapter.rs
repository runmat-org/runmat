use std::collections::{BTreeMap, HashMap};

use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
use runmat_value::Value;

use crate::{CompatMode, RunMatSession};

pub(crate) fn services(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
) -> runmat_runtime::testing::RuntimeTestServices {
    let run_project = project.clone();
    let run = move |args: Vec<Value>, coverage: Option<MatlabCoverageSelection>| {
        let project = run_project.clone();
        Box::pin(async move { run_arguments(compat, project, args, coverage).await })
            as runmat_runtime::testing::RunSuiteFuture
    };
    let suite_run = run.clone();
    let discover_project = project;
    runmat_runtime::testing::RuntimeTestServices::new(
        move |suite, plugins| {
            let args = suite_targets(suite);
            let coverage = plugins.iter().find_map(coverage_plugin_selection);
            suite_run(args, coverage)
        },
        move |args| run(args, None),
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
    coverage: Option<MatlabCoverageSelection>,
) -> runmat_runtime::BuiltinResult<Value> {
    let (snapshot, selected_names, requested_coverage) =
        snapshot_from_arguments(compat, project.as_ref(), args).await?;
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
    let coverage = coverage.or_else(|| requested_coverage.then(MatlabCoverageSelection::default));
    result_value(&run, coverage.as_ref())
}

async fn discover_arguments(
    compat: CompatMode,
    project: Option<runmat_package::FrozenProjectHandoff>,
    args: Vec<Value>,
) -> runmat_runtime::BuiltinResult<Value> {
    let (snapshot, selected_names, _) =
        snapshot_from_arguments(compat, project.as_ref(), args).await?;
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
    compat: CompatMode,
    project: Option<&runmat_package::FrozenProjectHandoff>,
    args: Vec<Value>,
) -> runmat_runtime::BuiltinResult<(FrozenTestRunSnapshot, Vec<String>, bool)> {
    let resolved =
        runmat_runtime::builtins::diagnostics::runtests::resolve_runtests_targets(args).await?;
    let mut sources = BTreeMap::new();
    let mut selected_names = Vec::new();
    let coverage = resolved.coverage;
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

fn result_value(
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
                if let Some(coverage) = &coverage {
                    let statements =
                        coverage.summary(runmat_test::coverage::CoverageMetric::Statement);
                    let functions =
                        coverage.summary(runmat_test::coverage::CoverageMetric::Function);
                    let mut summary =
                        runmat_value::ObjectInstance::new("RunMatCoverageResult".into());
                    summary.properties.insert(
                        "StatementsCovered".into(),
                        Value::Num(statements.covered as f64),
                    );
                    summary.properties.insert(
                        "StatementsTotal".into(),
                        Value::Num(statements.instrumented as f64),
                    );
                    summary.properties.insert(
                        "FunctionsCovered".into(),
                        Value::Num(functions.covered as f64),
                    );
                    summary.properties.insert(
                        "FunctionsTotal".into(),
                        Value::Num(functions.instrumented as f64),
                    );
                    object
                        .properties
                        .insert("Coverage".into(), Value::Object(summary));
                }
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct MatlabCoverageSelection {
    folders: Vec<String>,
    including_subfolders: bool,
}

fn coverage_plugin_selection(value: &Value) -> Option<MatlabCoverageSelection> {
    let Value::Object(object) = value else {
        return None;
    };
    if object.class_name != "matlab.unittest.plugins.CodeCoveragePlugin" {
        return None;
    }
    let folders = match object.properties.get("Folders") {
        Some(Value::StringArray(folders)) => folders.data.clone(),
        _ => Vec::new(),
    };
    let including_subfolders = matches!(
        object.properties.get("IncludingSubfolders"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_test::coverage::{CoverageAggregate, CoverageMetric, CoverageSite};

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
