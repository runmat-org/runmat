//! MATLAB-compatible `runtests` discovery and result helpers.
//!
//! Runtime owns MATLAB argument parsing, filesystem target resolution, and
//! result shaping. Core owns semantic discovery and execution.

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::common::path_search::{
    file_candidates, find_file_with_extensions, path_is_directory, path_is_file,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{BuiltinErrorDescriptor, RuntestsOption};
pub use runmat_builtins::{
    RUNTESTS_DESCRIPTOR, RUNTESTS_ERROR_FILE_READ, RUNTESTS_ERROR_INSUFFICIENT_RESOURCE,
    RUNTESTS_ERROR_INVALID_INPUT, RUNTESTS_ERROR_REQUIRES_EXECUTOR,
    RUNTESTS_ERROR_TARGET_NOT_FOUND, RUNTESTS_ERROR_UNSUPPORTED_OPTION,
    RUNTESTS_INTEGER_CAPABILITIES,
};
use runmat_macros::runtime_builtin;
use runmat_types::RUNTESTS_BUILTIN_NAME;
use runmat_value::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTestTarget {
    pub source_path: PathBuf,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTestTargets {
    pub targets: Vec<ResolvedTestTarget>,
    pub coverage: bool,
    pub filters: Vec<String>,
}

#[derive(Debug, Default)]
struct RunTestsOptions {
    include_subfolders: bool,
    targets: Vec<String>,
    base_folders: Vec<String>,
    filters: Vec<String>,
    coverage: bool,
}

#[runtime_builtin(
    name = "runtests",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::diagnostics::runtests"
)]
pub async fn runtests_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &RUNTESTS_DESCRIPTOR,
        &runmat_builtins::RUNTESTS_ERROR_TOO_MANY_OUTPUTS,
        "runtests",
    )?;
    crate::testing::run_tests_in_context(runtime, args).await
}

pub async fn resolve_runtests_targets(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<ResolvedTestTargets> {
    if runtime
        .and_then(|runtime| runtime.service_ports().parallel())
        .is_some_and(|port| port.placement_resources().memory_available_bytes.is_some())
    {
        return Err(runtests_error_detail(
            &RUNTESTS_ERROR_INSUFFICIENT_RESOURCE,
            "",
        ));
    }
    let options = parse_options_retained(&args)?;
    let mut paths = BTreeSet::new();
    let targets = if options.targets.is_empty() {
        if !options.base_folders.is_empty() {
            options.base_folders.clone()
        } else {
            vec![path_to_string(&runmat_filesystem::current_dir().map_err(
                |err| runtests_error_detail(&RUNTESTS_ERROR_TARGET_NOT_FOUND, err.to_string()),
            )?)]
        }
    } else {
        options.targets.clone()
    };

    let base_folders = if options.base_folders.is_empty() || options.targets.is_empty() {
        vec![None]
    } else {
        options
            .base_folders
            .iter()
            .map(|folder| Some(folder.as_str()))
            .collect()
    };

    for target in targets {
        for base_folder in &base_folders {
            for path in resolve_target(&target, *base_folder, options.include_subfolders).await? {
                let path =
                    runmat_filesystem::canonicalize_async(&path)
                        .await
                        .map_err(|source| {
                            runtests_io_error(
                                &RUNTESTS_ERROR_TARGET_NOT_FOUND,
                                path.display().to_string(),
                                source,
                            )
                        })?;
                paths.insert(path);
            }
        }
    }

    let mut cases = Vec::new();
    for path in paths {
        let source = runmat_filesystem::read_to_string_async(&path)
            .await
            .map_err(|err| {
                runtests_io_error(&RUNTESTS_ERROR_FILE_READ, path.display().to_string(), err)
            })?;
        let display_path = path;
        cases.push(ResolvedTestTarget {
            source_path: display_path,
            source,
        });
    }

    Ok(ResolvedTestTargets {
        targets: cases,
        coverage: options.coverage,
        filters: options.filters,
    })
}

#[cfg(test)]
fn parse_options(args: Vec<Value>) -> BuiltinResult<RunTestsOptions> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    parse_options_retained(&args)
}

fn parse_options_retained(args: &crate::call::BuiltinArguments) -> BuiltinResult<RunTestsOptions> {
    let mut options = RunTestsOptions::default();
    let mut idx = 0usize;

    if let Some(first) = args.first() {
        if !is_option_name(first)? {
            options.targets.extend(value_to_strings(first)?);
            idx = 1;
        }
    }

    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(runtests_error_detail(
                &RUNTESTS_ERROR_INVALID_INPUT,
                "name-value options must appear in pairs",
            ));
        }
        let name = value_to_string_scalar(&args[idx])?.to_ascii_lowercase();
        let value = &args[idx + 1];
        match RuntestsOption::parse(&name) {
            Some(RuntestsOption::IncludeSubfolders) => {
                options.include_subfolders = value_to_bool(value)?
            }
            Some(RuntestsOption::UseParallel) => {
                if value_to_bool(value)? {
                    return Err(runtests_error_detail(
                        &RUNTESTS_ERROR_UNSUPPORTED_OPTION,
                        "UseParallel=true is deferred to the parallel execution effort",
                    ));
                }
            }
            Some(RuntestsOption::BaseFolder) => {
                options.base_folders.extend(value_to_strings(value)?)
            }
            Some(RuntestsOption::Name) | Some(RuntestsOption::ProcedureName) => {
                options.filters.extend(value_to_strings(value)?)
            }
            Some(RuntestsOption::OutputDetail) | Some(RuntestsOption::LoggingLevel) => {
                let _ = value_to_string_scalar(value)?;
            }
            Some(RuntestsOption::Tag) => {
                let tags = value_to_strings(value)?;
                if !tags.iter().all(|tag| tag.is_empty()) {
                    return Err(runtests_error_detail(
                        &RUNTESTS_ERROR_UNSUPPORTED_OPTION,
                        "tag filtering requires matlab.unittest metadata support",
                    ));
                }
            }
            Some(RuntestsOption::Coverage) => {
                options.coverage = value_to_bool(value)?;
            }
            None => {
                return Err(runtests_error_detail(
                    &RUNTESTS_ERROR_INVALID_INPUT,
                    format!("unknown option '{name}'"),
                ));
            }
        }
        idx += 2;
    }

    Ok(options)
}

fn is_option_name(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::String(_) => Ok(RuntestsOption::parse(&value_to_string_scalar(value)?).is_some()),
        Value::CharArray(array) if array.rows == 1 => {
            Ok(RuntestsOption::parse(&value_to_string_scalar(value)?).is_some())
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(RuntestsOption::parse(&value_to_string_scalar(value)?).is_some())
        }
        _ => Ok(false),
    }
}

fn value_to_string_scalar(value: &Value) -> BuiltinResult<runmat_value::Utf8Text> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => {
            runmat_value::decode_utf16_code_units(array.data())
                .map(runmat_value::Utf8Text::from)
                .map_err(|error| {
                    runtests_error_detail(&RUNTESTS_ERROR_INVALID_INPUT, error.to_string())
                })
        }
        Value::StringArray(array) if array.data().len() == 1 => {
            array.data()[0].text_ref().cloned().ok_or_else(|| {
                runtests_error_detail(
                    &RUNTESTS_ERROR_INVALID_INPUT,
                    "expected a string scalar or character row",
                )
            })
        }
        other => Err(runtests_error_detail(
            &RUNTESTS_ERROR_INVALID_INPUT,
            format!("expected a string scalar or character row, got {other:?}"),
        )),
    }
}

fn value_to_strings(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => Ok(vec![text.to_string()]),
        Value::CharArray(array) if array.rows == 1 => {
            runmat_value::decode_utf16_code_units(array.data())
                .map(|text| vec![text])
                .map_err(|error| {
                    runtests_error_detail(&RUNTESTS_ERROR_INVALID_INPUT, error.to_string())
                })
        }
        Value::StringArray(array) => array
            .data()
            .iter()
            .map(|text| {
                text.as_str().map(str::to_owned).ok_or_else(|| {
                    runtests_error_detail(
                        &RUNTESTS_ERROR_INVALID_INPUT,
                        "expected a string scalar or character row",
                    )
                })
            })
            .collect(),
        Value::Cell(cell) => cell
            .data()
            .iter()
            .map(|value| {
                value_to_string_scalar(value).map(runmat_value::Utf8Text::into_string_unbounded)
            })
            .collect(),
        other => Err(runtests_error_detail(
            &RUNTESTS_ERROR_INVALID_INPUT,
            format!("expected a string, string array, or cell array of strings, got {other:?}"),
        )),
    }
}

fn value_to_bool(value: &Value) -> BuiltinResult<bool> {
    if let Some(integer) = crate::builtins::common::tensor::scalar_integer_value(value) {
        return match integer.try_to_u64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(runtests_error_detail(
                &RUNTESTS_ERROR_INVALID_INPUT,
                "expected an exact numeric or logical scalar value of 0 or 1",
            )),
        };
    }
    match value {
        Value::Bool(v) => Ok(*v),
        Value::Num(v) if *v == 0.0 || *v == 1.0 => Ok(*v != 0.0),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(runtests_error_detail(
            &RUNTESTS_ERROR_INVALID_INPUT,
            format!("expected a logical scalar, got {other:?}"),
        )),
    }
}

async fn resolve_target(
    target: &str,
    base_folder: Option<&str>,
    include_subfolders: bool,
) -> BuiltinResult<Vec<PathBuf>> {
    let expanded = expand_user_path(target, RUNTESTS_BUILTIN_NAME)
        .map_err(|err| runtests_error_detail(&RUNTESTS_ERROR_TARGET_NOT_FOUND, err))?;
    let direct = target_path_in_base(&expanded, base_folder)?;
    if path_is_directory(&direct)
        .await
        .map_err(|error| error.into_runtime_error(RUNTESTS_BUILTIN_NAME))?
    {
        return discover_test_files(&direct, include_subfolders).await;
    }
    if path_is_file(&direct)
        .await
        .map_err(|error| error.into_runtime_error(RUNTESTS_BUILTIN_NAME))?
    {
        return Ok(vec![direct]);
    }
    if base_folder.is_none() {
        if let Some(path) = find_file_with_extensions(&expanded, &[".m"], RUNTESTS_BUILTIN_NAME)
            .await
            .map_err(|error| error.into_runtime_error(RUNTESTS_BUILTIN_NAME))?
        {
            return Ok(vec![path]);
        }

        for candidate in file_candidates(&expanded, &[".m"], RUNTESTS_BUILTIN_NAME)
            .map_err(|error| error.into_runtime_error(RUNTESTS_BUILTIN_NAME))?
        {
            if path_is_directory(&candidate)
                .await
                .map_err(|error| error.into_runtime_error(RUNTESTS_BUILTIN_NAME))?
            {
                return discover_test_files(&candidate, include_subfolders).await;
            }
        }
    } else if direct.extension().is_none() {
        let candidate = direct.with_extension("m");
        if path_is_file(&candidate)
            .await
            .map_err(|error| error.into_runtime_error(RUNTESTS_BUILTIN_NAME))?
        {
            return Ok(vec![candidate]);
        }
    }

    Err(runtests_error_detail(
        &RUNTESTS_ERROR_TARGET_NOT_FOUND,
        format!("'{target}'"),
    ))
}

fn target_path_in_base(target: &str, base_folder: Option<&str>) -> BuiltinResult<PathBuf> {
    let target = PathBuf::from(target);
    let Some(base_folder) = base_folder else {
        return Ok(target);
    };
    if runmat_filesystem::current_provider()
        .path_namespace()
        .is_absolute(&target)
    {
        return Ok(target);
    }
    let expanded = expand_user_path(base_folder, RUNTESTS_BUILTIN_NAME)
        .map_err(|err| runtests_error_detail(&RUNTESTS_ERROR_TARGET_NOT_FOUND, err))?;
    Ok(PathBuf::from(expanded).join(target))
}

async fn discover_test_files(dir: &Path, include_subfolders: bool) -> BuiltinResult<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = runmat_filesystem::read_dir_async(&current)
            .await
            .map_err(|err| {
                runtests_io_error(
                    &RUNTESTS_ERROR_TARGET_NOT_FOUND,
                    current.display().to_string(),
                    err,
                )
            })?;
        for entry in entries {
            let path = entry.path().to_path_buf();
            if entry.is_dir() {
                if include_subfolders {
                    stack.push(path);
                }
                continue;
            }
            if is_test_file(&path) {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn is_test_file(path: &Path) -> bool {
    if !path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("m"))
    {
        return false;
    }
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };
    let lower = stem.to_ascii_lowercase();
    lower.starts_with("test") || lower.ends_with("test") || lower.ends_with("tests")
}

fn runtests_error_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(RUNTESTS_BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn runtests_io_error(
    error: &'static BuiltinErrorDescriptor,
    detail: String,
    source: std::io::Error,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {detail} ({source})", error.message))
        .with_builtin(RUNTESTS_BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{
        CellArray, CharArray, IntegerStorage, LogicalArray, StringArray, Tensor, Utf16CodeUnit,
    };

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn first_argument_detection_preserves_target_arrays_and_decode_errors() {
        let targets = Value::StringArray(
            StringArray::new(vec!["one".into(), "two".into()], vec![1, 2])
                .expect("valid target array"),
        );
        assert!(!is_option_name(&targets).expect("target arrays remain targets"));

        let malformed = Value::CharArray(
            CharArray::from_code_units(vec![Utf16CodeUnit::from_raw(0xd83d)], 1, 1)
                .expect("valid character geometry"),
        );
        let error = is_option_name(&malformed).expect_err("invalid UTF-16 is not an option");
        assert_eq!(error.identifier(), RUNTESTS_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn finite_test_discovery_refuses_before_missing_target_resolution() {
        use crate::context::{
            ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
            RuntimeServicePorts,
        };
        use std::rc::Rc;
        struct Finite;
        impl RuntimeParallelService for Finite {
            fn supports(&self, _: ParallelCapability) -> bool {
                false
            }
            fn placement_resources(&self) -> RuntimeParallelResources {
                RuntimeParallelResources {
                    memory_available_bytes: Some(1),
                    ..Default::default()
                }
            }
        }
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(RuntimeServicePorts::default().with_parallel(Rc::new(Finite)));
        let args = crate::call::BuiltinArguments::from_borrowed(
            &[Value::String("/absent/testMissing.m".into())],
            None,
        )
        .unwrap();
        let error = futures::executor::block_on(resolve_runtests_targets(args, Some(&runtime)))
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
    }

    #[test]
    fn runtime_identity_is_the_single_canonical_default_binding() {
        let entry =
            runmat_builtins::builtin_catalog_entry_by_name("runtests").expect("canonical runtests");
        let bindings = crate::builtin::runtime_builtin_bindings_by_name("runtests");
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].identity,
            entry.binding_identity(&entry.bindings[0])
        );
    }

    #[test]
    fn parse_accepts_target_and_include_subfolders() {
        let opts = parse_options(vec![
            Value::String("tests".to_string().into()),
            Value::String("IncludeSubfolders".to_string().into()),
            Value::Bool(true),
        ])
        .expect("parse options");
        assert_eq!(opts.targets, vec!["tests"]);
        assert!(opts.include_subfolders);
    }

    #[test]
    fn parse_accepts_coverage_collection() {
        let options =
            parse_options(vec![Value::String("Coverage".into()), Value::Bool(true)]).unwrap();
        assert!(options.coverage);
    }

    #[test]
    fn parse_rejects_integer_flags_other_than_exact_zero_or_one() {
        let flag = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(vec![2]), vec![1, 1])
                .expect("scalar uint8 flag"),
        );
        let error = parse_options(vec![
            Value::String("IncludeSubfolders".to_string().into()),
            flag,
        ])
        .expect_err("invalid integer flag");
        assert_eq!(error.identifier(), Some("RunMat:runtests:InvalidInput"));
        assert!(error.message().contains("0 or 1"));
    }

    #[test]
    fn parse_rejects_parallel_execution() {
        let err = parse_options(vec![
            Value::String("UseParallel".to_string().into()),
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
        ])
        .unwrap_err();
        assert_eq!(
            err.identifier().map(str::to_string),
            Some("RunMat:runtests:UnsupportedOption".to_string())
        );
    }

    #[test]
    fn string_collection_accepts_cell_targets() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("testOne")),
                Value::String("testTwo".to_string().into()),
            ],
            1,
            2,
        )
        .unwrap();
        assert_eq!(
            value_to_strings(&Value::Cell(cell)).unwrap(),
            vec!["testOne".to_string(), "testTwo".to_string()]
        );
    }

    #[test]
    fn discovers_matlab_test_file_names() {
        assert!(is_test_file(Path::new("testSmoke.m")));
        assert!(is_test_file(Path::new("SmokeTest.m")));
        assert!(!is_test_file(Path::new("helper.m")));
        assert!(!is_test_file(Path::new("testSmoke.txt")));
    }

    #[test]
    fn string_array_targets_are_flattened() {
        let array = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        assert_eq!(
            value_to_strings(&Value::StringArray(array)).unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
    }
}
