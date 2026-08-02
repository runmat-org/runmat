//! MATLAB-compatible `runtests` discovery and result helpers.
//!
//! Runtime owns MATLAB argument parsing, filesystem target resolution, and
//! result shaping. Core owns semantic discovery and execution.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type, Value,
};
use runmat_hir::RUNTESTS_BUILTIN_NAME;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::common::path_search::{
    file_candidates, find_file_with_extensions, path_is_directory, path_is_file,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const RUNTESTS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "tests",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("current folder"),
        description: "Test file, folder, function name, string array, or cell array of targets.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Common options such as IncludeSubfolders, BaseFolder, Name, ProcedureName, and UseParallel.",
    },
];

const RUNTESTS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "results",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Scalar TestResult object or homogeneous TestResult object row.",
}];

const RUNTESTS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "results = runtests",
        inputs: &[],
        outputs: &RUNTESTS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "results = runtests(tests, Name, Value, ...)",
        inputs: &RUNTESTS_INPUTS,
        outputs: &RUNTESTS_OUTPUT,
    },
];

pub const RUNTESTS_ERROR_REQUIRES_EXECUTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.REQUIRES_EXECUTOR",
    identifier: Some("RunMat:Testing:RequiresExecutor"),
    when: "`runtests` is dispatched outside an active Core test executor.",
    message: "runtests: requires an active Core test executor",
};

pub const RUNTESTS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.INVALID_INPUT",
    identifier: Some("RunMat:runtests:InvalidInput"),
    when: "A target or option value has an unsupported type or value.",
    message: "runtests: invalid input",
};

pub const RUNTESTS_ERROR_UNSUPPORTED_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.UNSUPPORTED_OPTION",
    identifier: Some("RunMat:runtests:UnsupportedOption"),
    when: "A documented option requires an execution mode not available through the in-program adapter.",
    message: "runtests: unsupported option",
};

pub const RUNTESTS_ERROR_TARGET_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.TARGET_NOT_FOUND",
    identifier: Some("RunMat:runtests:TargetNotFound"),
    when: "A requested test target cannot be resolved to a file or folder.",
    message: "runtests: test target not found",
};

pub const RUNTESTS_ERROR_FILE_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUNTESTS.FILE_READ",
    identifier: Some("RunMat:runtests:FileReadFailed"),
    when: "A discovered test file cannot be read as source text.",
    message: "runtests: failed to read test file",
};

pub const RUNTESTS_ERRORS: [BuiltinErrorDescriptor; 5] = [
    RUNTESTS_ERROR_REQUIRES_EXECUTOR,
    RUNTESTS_ERROR_INVALID_INPUT,
    RUNTESTS_ERROR_UNSUPPORTED_OPTION,
    RUNTESTS_ERROR_TARGET_NOT_FOUND,
    RUNTESTS_ERROR_FILE_READ,
];

pub const RUNTESTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RUNTESTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RUNTESTS_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::diagnostics::runtests")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "runtests",
    op_kind: GpuOpKind::Custom("testing"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Test discovery and execution are host control-flow operations. Test bodies may call GPU-capable builtins normally, but runtests itself has no device kernel.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::diagnostics::runtests")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "runtests",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Test execution is a Core service and filesystem boundary and is excluded from fusion.",
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTestTarget {
    pub name: String,
    pub source_path: PathBuf,
    pub display_name: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTestTargets {
    pub targets: Vec<ResolvedTestTarget>,
    pub coverage: bool,
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
    category = "diagnostics",
    summary = "Discover and run MATLAB-style test files.",
    keywords = "test,unit testing,runtests,diagnostics,developer tools",
    descriptor(self::RUNTESTS_DESCRIPTOR),
    type_resolver(runtests_type),
    builtin_path = "crate::builtins::diagnostics::runtests"
)]
pub async fn runtests_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::testing::run_tests(args).await
}

fn runtests_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Object {
        class_name: Some(crate::testing::TEST_RESULT_CLASS.into()),
        shape: None,
    }
}

pub async fn resolve_runtests_targets(args: Vec<Value>) -> BuiltinResult<ResolvedTestTargets> {
    let gathered = gather_values(args).await?;
    let options = parse_options(gathered)?;
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
                paths.insert(path);
            }
        }
    }

    let mut cases = Vec::new();
    for path in paths {
        let source = runmat_filesystem::read_to_string_async(&path)
            .await
            .map_err(|err| {
                runtests_error_detail(
                    &RUNTESTS_ERROR_FILE_READ,
                    format!("{} ({err})", path.display()),
                )
            })?;
        let display_path = runmat_filesystem::canonicalize_async(&path)
            .await
            .unwrap_or_else(|_| path.clone());
        let file_name = test_name_for_path(&display_path);
        let function_tests = function_test_names(&source);
        if function_tests.is_empty() {
            if !matches_filters(&file_name, &options.filters) {
                continue;
            }
            cases.push(ResolvedTestTarget {
                name: file_name,
                source_path: display_path.clone(),
                display_name: path_to_string(&display_path),
                source,
            });
        } else {
            for function_name in function_tests {
                let name = format!("{file_name}/{function_name}");
                if !matches_filters(&name, &options.filters)
                    && !matches_filters(&function_name, &options.filters)
                {
                    continue;
                }
                cases.push(ResolvedTestTarget {
                    name,
                    source_path: display_path.clone(),
                    display_name: path_to_string(&display_path),
                    source: format!("{source}\n{function_name}();\n"),
                });
            }
        }
    }

    Ok(ResolvedTestTargets {
        targets: cases,
        coverage: options.coverage,
    })
}

async fn gather_values(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(gather_if_needed_async(&arg).await.map_err(runtests_flow)?);
    }
    Ok(out)
}

fn parse_options(args: Vec<Value>) -> BuiltinResult<RunTestsOptions> {
    let mut options = RunTestsOptions::default();
    let mut idx = 0usize;

    if let Some(first) = args.first() {
        if !is_option_name(first) {
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
        match normalize_option_name(&name).as_str() {
            "includesubfolders" => options.include_subfolders = value_to_bool(value)?,
            "useparallel" => {
                if value_to_bool(value)? {
                    return Err(runtests_error_detail(
                        &RUNTESTS_ERROR_UNSUPPORTED_OPTION,
                        "UseParallel=true is deferred to the parallel execution effort",
                    ));
                }
            }
            "basefolder" => options.base_folders.extend(value_to_strings(value)?),
            "name" | "procedurename" => options.filters.extend(value_to_strings(value)?),
            "outputdetail" | "logginglevel" => {
                let _ = value_to_string_scalar(value)?;
            }
            "tag" => {
                let tags = value_to_strings(value)?;
                if !tags.iter().all(|tag| tag.is_empty()) {
                    return Err(runtests_error_detail(
                        &RUNTESTS_ERROR_UNSUPPORTED_OPTION,
                        "tag filtering requires matlab.unittest metadata support",
                    ));
                }
            }
            "coverage" => {
                options.coverage = value_to_bool(value)?;
            }
            other => {
                return Err(runtests_error_detail(
                    &RUNTESTS_ERROR_INVALID_INPUT,
                    format!("unknown option '{other}'"),
                ));
            }
        }
        idx += 2;
    }

    Ok(options)
}

fn normalize_option_name(name: &str) -> String {
    name.chars()
        .filter(|ch| !ch.is_ascii_whitespace() && *ch != '_' && *ch != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_option_name(value: &Value) -> bool {
    let Ok(text) = value_to_string_scalar(value) else {
        return false;
    };
    matches!(
        normalize_option_name(&text).as_str(),
        "includesubfolders"
            | "useparallel"
            | "basefolder"
            | "name"
            | "procedurename"
            | "outputdetail"
            | "logginglevel"
            | "tag"
            | "coverage"
    )
}

fn value_to_string_scalar(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(runtests_error_detail(
            &RUNTESTS_ERROR_INVALID_INPUT,
            format!("expected a string scalar or character row, got {other:?}"),
        )),
    }
}

fn value_to_strings(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::CharArray(array) if array.rows == 1 => Ok(vec![array.data.iter().collect()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(cell) => cell.data.iter().map(value_to_string_scalar).collect(),
        other => Err(runtests_error_detail(
            &RUNTESTS_ERROR_INVALID_INPUT,
            format!("expected a string, string array, or cell array of strings, got {other:?}"),
        )),
    }
}

fn value_to_bool(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(v) => Ok(*v),
        Value::Num(v) if *v == 0.0 || *v == 1.0 => Ok(*v != 0.0),
        Value::Int(v) => Ok(v.to_f64() != 0.0),
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
    if path_is_directory(&direct).await {
        return discover_test_files(&direct, include_subfolders).await;
    }
    if path_is_file(&direct).await {
        return Ok(vec![direct]);
    }
    if base_folder.is_none() {
        if let Some(path) = find_file_with_extensions(&expanded, &[".m"], RUNTESTS_BUILTIN_NAME)
            .await
            .map_err(|err| runtests_error_detail(&RUNTESTS_ERROR_TARGET_NOT_FOUND, err))?
        {
            return Ok(vec![path]);
        }

        for candidate in file_candidates(&expanded, &[".m"], RUNTESTS_BUILTIN_NAME)
            .map_err(|err| runtests_error_detail(&RUNTESTS_ERROR_TARGET_NOT_FOUND, err))?
        {
            if path_is_directory(&candidate).await {
                return discover_test_files(&candidate, include_subfolders).await;
            }
        }
    } else if direct.extension().is_none() {
        let candidate = direct.with_extension("m");
        if path_is_file(&candidate).await {
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
    if target.is_absolute() {
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
                runtests_error_detail(
                    &RUNTESTS_ERROR_TARGET_NOT_FOUND,
                    format!("{} ({err})", current.display()),
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

fn test_name_for_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("unnamed")
        .to_string()
}

fn matches_filters(name: &str, filters: &[String]) -> bool {
    filters.is_empty() || filters.iter().any(|filter| name.contains(filter))
}

fn function_test_names(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim_start();
        let lowered = trimmed.to_ascii_lowercase();
        if !lowered.starts_with("function") {
            continue;
        }
        let rest = trimmed["function".len()..].trim_start();
        let after_outputs = rest
            .split_once('=')
            .map(|(_, rhs)| rhs.trim_start())
            .unwrap_or(rest);
        let name = after_outputs
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            .collect::<String>();
        if name.is_empty() {
            continue;
        }
        let lower_name = name.to_ascii_lowercase();
        if lower_name.starts_with("test") || lower_name.ends_with("test") {
            names.push(name);
        }
    }
    names.sort();
    names.dedup();
    names
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

fn runtests_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(err.message().to_string())
        .with_builtin(RUNTESTS_BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{CellArray, CharArray, LogicalArray, StringArray};

    #[test]
    fn parse_accepts_target_and_include_subfolders() {
        let opts = parse_options(vec![
            Value::String("tests".to_string()),
            Value::String("IncludeSubfolders".to_string()),
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
    fn parse_rejects_parallel_execution() {
        let err = parse_options(vec![
            Value::String("UseParallel".to_string()),
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
                Value::String("testTwo".to_string()),
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
    fn discovers_function_test_names() {
        let names = function_test_names(
            r#"
function helper()
end
function testAlpha()
end
function out = betaTest()
end
"#,
        );
        assert_eq!(names, vec!["betaTest".to_string(), "testAlpha".to_string()]);
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
