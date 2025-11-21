//! MATLAB-compatible `who` builtin for RunMat.

use glob::Pattern;
use regex::Regex;
use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;
use std::path::PathBuf;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::io::mat::load::read_mat_file;
use crate::{gather_if_needed, make_cell, register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "who"
category: "introspection"
keywords: ["who", "workspace", "variables", "introspection", "mat-file"]
summary: "List the names of variables in the workspace or MAT-files (MATLAB-compatible)."
references:
  - https://www.mathworks.com/help/matlab/ref/who.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host. GPU arrays remain on the device; RunMat inspects metadata without gathering buffers."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::introspection::who::tests"
  integration: null
---

# What does the `who` function do in MATLAB / RunMat?
`who` returns the names of variables that are visible in the active workspace. You can filter the result using wildcard patterns, regular expressions, or by reading directly from MAT-files without loading the data.

## How does the `who` function behave in MATLAB / RunMat?
- `who` with no arguments lists every variable in the current workspace, sorted alphabetically.
- `who pattern1 pattern2 ...` accepts character vectors or string scalars with MATLAB wildcard syntax (`*`, `?`, `[abc]`). Any variable name that matches at least one pattern is returned.
- `who('-regexp', expr1, expr2, ...)` keeps names that match any of the supplied regular expressions.
- `who('global')` lists only global variables in the active workspace.
- `who('-file', filename, ...)` inspects the variables stored in a MAT-file. You can combine this option with explicit names or `-regexp` selectors.
- The result is a column cell array of character vectors (consistent with MATLAB). Empty results return a `0×1` cell array so that idioms like `isempty(who(...))` work as expected.

## `who` Function GPU Execution Behaviour
`who` is a pure introspection builtin that runs on the CPU. When a variable is a `gpuArray`, RunMat leaves it resident on the device and reports its name without triggering any device-to-host copies. Only scalar selector arguments are gathered if they are stored on the GPU.

## Examples of using the `who` function in MATLAB / RunMat

### List All Workspace Variables
```matlab
a = 42;
b = rand(3, 2);
names = who;
```
Expected output (example):
```matlab
names =
  2×1 cell array
    {"a"}
    {"b"}
```

### Filter With Wildcard Patterns
```matlab
alpha = 1;
beta = 2;
names = who("a*");
```
Expected output:
```matlab
names =
  1×1 cell array
    {"alpha"}
```

### Use Regular Expressions
```matlab
x1 = rand;
x2 = rand;
matches = who('-regexp', '^x\d$');
```
Expected output (example):
```matlab
matches =
  2×1 cell array
    {"x1"}
    {"x2"}
```

### Inspect Variables Stored In A MAT-File
```matlab
save('snapshot.mat', 'alpha', 'beta')
file_names = who('-file', 'snapshot.mat');
```
Expected output (example):
```matlab
file_names =
  2×1 cell array
    {"alpha"}
    {"beta"}
```

### List Only Global Variables
```matlab
global shared;
local = 1;
globals = who('global');
```
Expected output (example):
```matlab
globals =
  1×1 cell array
    {"shared"}
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `who` never requires you to gather data or move arrays between the host and GPU. It simply reports variable names, regardless of residency. Use `gpuArray` or `gather` only when you explicitly need to control where data lives.

## FAQ
### What type does `who` return?
A column cell array of character vectors, matching MATLAB behaviour.

### Are the names sorted?
Yes. Results are sorted alphabetically so that diffs are deterministic.

### Can I mix wildcard patterns and `-regexp`?
Yes. The final result includes any name matched by either the wildcard selectors or the regular expressions.

### What happens if no variables match?
You receive a `0×1` cell array. You can call `isempty` on it to check for an empty result.

### Can I call `who('-file', ...)` on large MAT-files?
Yes. The builtin reads just enough metadata to enumerate variable names; it does not load the data into the workspace.

## See Also
[whos](./whos), [which](./which), [class](./class), [size](../array/introspection/size), [load](../io/mat/load), [save](../io/mat/save), [gpuArray](../acceleration/gpu/gpuArray), [gather](../acceleration/gpu/gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/introspection/who.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/who.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal reproduction.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "who",
    op_kind: GpuOpKind::Custom("introspection"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only builtin. Arguments are gathered from the GPU if necessary; no kernels are launched.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "who",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Introspection builtin; registered for diagnostics only.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("who", DOC_MD);

#[runtime_builtin(
    name = "who",
    category = "introspection",
    summary = "List the names of variables in the workspace or MAT-files (MATLAB-compatible).",
    keywords = "who,workspace,variables,introspection",
    accel = "cpu"
)]
fn who_builtin(args: Vec<Value>) -> Result<Value, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        // Only install the WGPU provider if none is currently registered to avoid clobbering
        // the in-process provider used by non-WGPU tests.
        if runmat_accelerate_api::provider().is_none() {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(gather_if_needed(&arg)?);
    }
    let request = parse_request(&gathered)?;

    let mut entries = match &request.source {
        WhoSource::Workspace => crate::workspace::snapshot().unwrap_or_default(),
        WhoSource::File(path) => {
            read_mat_file(path).map_err(|err| err.replacen("load:", "who:", 1))?
        }
    };

    if matches!(request.source, WhoSource::File(_)) {
        entries.sort_by(|a, b| a.0.cmp(&b.0));
    }

    let global_names: std::collections::HashSet<String> =
        if matches!(request.source, WhoSource::Workspace) {
            crate::workspace::global_names().into_iter().collect()
        } else {
            std::collections::HashSet::new()
        };

    let mut names = Vec::new();
    for (name, _value) in entries {
        if !matches_filters(&name, &request.selectors, &request.regex_patterns) {
            continue;
        }
        let is_global = global_names.contains(&name);
        if request.only_global && !is_global {
            continue;
        }
        names.push(name);
    }

    names.sort();

    let mut cells = Vec::with_capacity(names.len());
    for name in names.into_iter() {
        cells.push(Value::String(name));
    }
    let rows = cells.len();
    make_cell(cells, rows, 1)
}

#[derive(Debug)]
struct WhoRequest {
    source: WhoSource,
    selectors: Vec<NameSelector>,
    regex_patterns: Vec<Regex>,
    only_global: bool,
}

#[derive(Debug)]
enum WhoSource {
    Workspace,
    File(PathBuf),
}

#[derive(Debug)]
enum NameSelector {
    Exact(String),
    Wildcard(Pattern),
}

fn parse_request(values: &[Value]) -> Result<WhoRequest, String> {
    let mut idx = 0usize;
    let mut path_value: Option<Value> = None;
    let mut names: Vec<String> = Vec::new();
    let mut regex_patterns = Vec::new();
    let mut only_global = false;

    while idx < values.len() {
        if let Some(token) = option_token(&values[idx])? {
            match token.as_str() {
                "-file" => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err("who: '-file' requires a filename".to_string());
                    }
                    if path_value.is_some() {
                        return Err("who: '-file' may only be specified once".to_string());
                    }
                    path_value = Some(values[idx].clone());
                    idx += 1;
                    continue;
                }
                "-regexp" => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err("who: '-regexp' requires at least one pattern".to_string());
                    }
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let candidates = extract_name_list(&values[idx])?;
                        if candidates.is_empty() {
                            return Err(
                                "who: '-regexp' requires non-empty pattern strings".to_string()
                            );
                        }
                        for pattern in candidates {
                            let regex = Regex::new(&pattern).map_err(|err| {
                                format!("who: invalid regular expression '{pattern}': {err}")
                            })?;
                            regex_patterns.push(regex);
                        }
                        idx += 1;
                    }
                    continue;
                }
                other => {
                    return Err(format!("who: unsupported option '{other}'"));
                }
            }
        }

        let extracted = extract_name_list(&values[idx])?;
        if extracted.is_empty() {
            idx += 1;
            continue;
        }
        if extracted.len() == 1
            && extracted[0].eq_ignore_ascii_case("global")
            && names.is_empty()
            && regex_patterns.is_empty()
            && path_value.is_none()
        {
            only_global = true;
        } else {
            names.extend(extracted);
        }
        idx += 1;
    }

    let source = if let Some(path_value) = path_value {
        let path = parse_file_path(&path_value)?;
        WhoSource::File(path)
    } else {
        WhoSource::Workspace
    };

    let selectors = build_selectors(&names)?;

    Ok(WhoRequest {
        source,
        selectors,
        regex_patterns,
        only_global,
    })
}

fn matches_filters(name: &str, selectors: &[NameSelector], regex_patterns: &[Regex]) -> bool {
    if selectors.is_empty() && regex_patterns.is_empty() {
        return true;
    }
    if selectors.iter().any(|selector| match selector {
        NameSelector::Exact(expected) => name == expected,
        NameSelector::Wildcard(pattern) => pattern.matches(name),
    }) {
        return true;
    }
    regex_patterns.iter().any(|regex| regex.is_match(name))
}

fn build_selectors(names: &[String]) -> Result<Vec<NameSelector>, String> {
    let mut selectors = Vec::with_capacity(names.len());
    for name in names {
        if contains_wildcards(name) {
            let pattern = Pattern::new(name)
                .map_err(|err| format!("who: invalid pattern '{name}': {err}"))?;
            selectors.push(NameSelector::Wildcard(pattern));
        } else {
            selectors.push(NameSelector::Exact(name.clone()));
        }
    }
    Ok(selectors)
}

fn contains_wildcards(text: &str) -> bool {
    text.chars().any(|ch| matches!(ch, '*' | '?' | '['))
}

fn parse_file_path(value: &Value) -> Result<PathBuf, String> {
    let text = value_to_string_scalar(value)
        .ok_or_else(|| "who: filename must be a character vector or string scalar".to_string())?;
    let mut path = PathBuf::from(text);
    if path.extension().is_none() {
        path.set_extension("mat");
    }
    Ok(path)
}

fn option_token(value: &Value) -> Result<Option<String>, String> {
    if let Some(token) = value_to_string_scalar(value) {
        if token.starts_with('-') {
            return Ok(Some(token.to_ascii_lowercase()));
        }
    }
    Ok(None)
}

fn extract_name_list(value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::CharArray(ca) => Ok(char_array_rows_as_strings(ca)),
        Value::StringArray(sa) => Ok(sa.data.clone()),
        Value::Cell(ca) => {
            let mut names = Vec::with_capacity(ca.data.len());
            for handle in &ca.data {
                let inner = unsafe { &*handle.as_raw() };
                if let Some(text) = value_to_string_scalar(inner) {
                    names.push(text);
                    continue;
                }
                let gathered = gather_if_needed(inner)?;
                if let Some(text) = value_to_string_scalar(&gathered) {
                    names.push(text);
                } else {
                    return Err(
                        "who: selection cells must contain string or character scalars".to_string(),
                    );
                }
            }
            Ok(names)
        }
        Value::GpuTensor(_) => {
            let gathered = gather_if_needed(value)?;
            extract_name_list(&gathered)
        }
        _ => Err(
            "who: selections must be character vectors, string scalars, string arrays, or cell arrays of those types"
                .to_string(),
        ),
    }
}

fn value_to_string_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => char_array_rows_as_strings(ca).into_iter().next(),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}

fn char_array_rows_as_strings(ca: &CharArray) -> Vec<String> {
    let mut rows = Vec::with_capacity(ca.rows);
    for r in 0..ca.rows {
        let mut row = String::with_capacity(ca.cols);
        for c in 0..ca.cols {
            let idx = r * ca.cols + c;
            row.push(ca.data[idx]);
        }
        rows.push(row.trim_end_matches([' ', '\0']).to_string());
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::super::whos::tests::{
        char_array_from_rows as shared_char_array_from_rows,
        ensure_test_resolver as ensure_shared_resolver, set_workspace as shared_set_workspace,
    };
    use super::*;
    use crate::builtins::common::test_support;
    use crate::call_builtin;
    use runmat_builtins::{CellArray, CharArray, StringArray, Tensor};
    use tempfile::tempdir;

    fn names_from_value(value: Value) -> Vec<String> {
        match value {
            Value::Cell(cell) => cell
                .data
                .iter()
                .map(|ptr| unsafe { &*ptr.as_raw() })
                .map(|value| match value {
                    Value::String(s) => s.clone(),
                    Value::CharArray(ca) if ca.rows == 1 => ca
                        .data
                        .iter()
                        .collect::<String>()
                        .trim_end_matches([' ', '\0'])
                        .to_string(),
                    other => panic!("expected string entry, got {other:?}"),
                })
                .collect(),
            Value::String(s) => vec![s],
            Value::StringArray(sa) => sa.data,
            Value::CharArray(ca) => char_array_rows_as_strings(&ca),
            other => panic!("expected cell array result, got {other:?}"),
        }
    }

    #[test]
    fn who_lists_workspace_variables() {
        ensure_shared_resolver();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        shared_set_workspace(
            &[("alpha", Value::Num(1.0)), ("beta", Value::Tensor(tensor))],
            &[],
        );

        let value = who_builtin(Vec::new()).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn who_filters_with_wildcard() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[("alpha", Value::Num(1.0)), ("beta", Value::Num(2.0))],
            &[],
        );

        let value = who_builtin(vec![Value::from("a*")]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string()]);
    }

    #[test]
    fn who_filters_with_regex() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[
                ("foo", Value::Num(1.0)),
                ("bar", Value::Num(2.0)),
                ("baz", Value::Num(3.0)),
            ],
            &[],
        );

        let value = who_builtin(vec![Value::from("-regexp"), Value::from("^ba")]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["bar".to_string(), "baz".to_string()]);
    }

    #[test]
    fn who_combines_wildcard_and_regex_filters() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("beta", Value::Num(2.0)),
                ("gamma", Value::Num(3.0)),
                ("delta", Value::Num(4.0)),
            ],
            &[],
        );

        let value = who_builtin(vec![
            Value::from("a*"),
            Value::from("-regexp"),
            Value::from("ma$"),
        ])
        .expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[test]
    fn who_filters_global_only() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[("shared", Value::Num(1.0)), ("local", Value::Num(2.0))],
            &["shared"],
        );

        let value = who_builtin(vec![Value::from("global")]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["shared".to_string()]);
    }

    #[test]
    fn who_global_option_is_case_insensitive() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[("Shared", Value::Num(1.0)), ("local", Value::Num(2.0))],
            &["Shared"],
        );

        let value = who_builtin(vec![Value::from("GLOBAL")]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["Shared".to_string()]);
    }

    #[test]
    fn who_accepts_char_array_arguments() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("gamma", Value::Num(3.0)),
                ("omega", Value::Num(4.0)),
            ],
            &[],
        );

        let arg = Value::CharArray(shared_char_array_from_rows(&["alpha", "gamma"]));
        let value = who_builtin(vec![arg]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[test]
    fn who_accepts_string_array_arguments() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("gamma", Value::Num(3.0)),
                ("omega", Value::Num(4.0)),
            ],
            &[],
        );

        let array =
            StringArray::new(vec!["gamma".to_string(), "alpha".to_string()], vec![2, 1]).unwrap();
        let value = who_builtin(vec![Value::StringArray(array)]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[test]
    fn who_accepts_cell_array_arguments() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("gamma", Value::Num(3.0)),
                ("omega", Value::Num(4.0)),
            ],
            &[],
        );

        let cell = CellArray::new(vec![Value::from("gamma"), Value::from("alpha")], 2, 1).unwrap();
        let value = who_builtin(vec![Value::Cell(cell)]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[test]
    fn who_rejects_numeric_selection() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let err = who_builtin(vec![Value::Num(7.0)]).expect_err("who should error");
        assert!(
            err.contains("who: selections must"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn who_rejects_unknown_option() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let err = who_builtin(vec![Value::from("-bogus")]).expect_err("who should error");
        assert!(
            err.contains("unsupported option"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn who_requires_filename_for_file_option() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let err = who_builtin(vec![Value::from("-file")]).expect_err("who should error");
        assert!(err.contains("'-file' requires a filename"), "error: {err}");
    }

    #[test]
    fn who_requires_pattern_for_regexp() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let err = who_builtin(vec![Value::from("-regexp")]).expect_err("who should error");
        assert!(
            err.contains("'-regexp' requires at least one pattern"),
            "error: {err}"
        );
    }

    #[test]
    fn who_rejects_invalid_regex() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let err = who_builtin(vec![Value::from("-regexp"), Value::from("[")])
            .expect_err("who should error");
        assert!(err.contains("invalid regular expression"), "error: {err}");
    }

    #[test]
    fn who_returns_empty_column_cell_when_no_match() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let value = who_builtin(vec![Value::from("nothing")]).expect("who");
        match value {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 0);
                assert_eq!(cell.cols, 1);
                assert!(cell.data.is_empty());
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn who_file_option_reads_mat_file() {
        ensure_shared_resolver();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("beta", Value::Tensor(tensor.clone())),
            ],
            &[],
        );

        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("snapshot.mat");
        let path_str = file_path.to_string_lossy().to_string();
        call_builtin(
            "save",
            &[
                Value::from(path_str.clone()),
                Value::from("alpha"),
                Value::from("beta"),
            ],
        )
        .expect("save");

        shared_set_workspace(&[], &[]);
        let value = who_builtin(vec![Value::from("-file"), Value::from(path_str)]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn who_file_option_combines_literal_and_regex_selectors() {
        ensure_shared_resolver();
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("beta", Value::Tensor(tensor.clone())),
                ("gamma", Value::Tensor(tensor)),
            ],
            &[],
        );

        let dir = tempdir().expect("tempdir");
        let stem_path = dir.path().join("snapshot_combo");
        let stem_str = stem_path.to_string_lossy().to_string();
        call_builtin(
            "save",
            &[
                Value::from(stem_str.clone()),
                Value::from("alpha"),
                Value::from("beta"),
                Value::from("gamma"),
            ],
        )
        .expect("save");

        shared_set_workspace(&[], &[]);
        let value = who_builtin(vec![
            Value::from("-file"),
            Value::from(stem_str.clone()),
            Value::from("alpha"),
            Value::from("-regexp"),
            Value::from("^b"),
        ])
        .expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn who_file_option_adds_mat_extension() {
        ensure_shared_resolver();
        shared_set_workspace(&[("v", Value::Num(2.0))], &[]);

        let dir = tempdir().expect("tempdir");
        let stem_path = dir.path().join("snapshot_no_ext");
        let stem_str = stem_path.to_string_lossy().to_string();
        call_builtin("save", &[Value::from(stem_str.clone()), Value::from("v")]).expect("save");

        shared_set_workspace(&[], &[]);
        let value = who_builtin(vec![Value::from("-file"), Value::from(stem_str)]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["v".to_string()]);
    }

    #[test]
    fn who_handles_gpu_workspace_entries() {
        ensure_shared_resolver();
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let gpu_value = crate::call_builtin("gpuArray", &[Value::Tensor(tensor)]).expect("gpu");
            shared_set_workspace(
                &[("gpuVar", gpu_value.clone()), ("hostVar", Value::Num(5.0))],
                &[],
            );

            let value = who_builtin(Vec::new()).expect("who");
            let names = names_from_value(value);
            assert_eq!(names, vec!["gpuVar".to_string(), "hostVar".to_string()]);
            shared_set_workspace(&[], &[]);
        });
    }

    #[test]
    fn who_respects_global_filter_with_gpu_variables() {
        ensure_shared_resolver();
        test_support::with_test_provider(|_| {
            shared_set_workspace(&[], &[]);
            let gpu_scalar = crate::call_builtin("gpuArray", &[Value::Num(3.0)]).expect("gpu");
            shared_set_workspace(&[("shared_gpu", gpu_scalar)], &["shared_gpu"]);

            let value = who_builtin(vec![Value::from("global")]).expect("who");
            let names = names_from_value(value);
            assert_eq!(names, vec!["shared_gpu".to_string()]);
            shared_set_workspace(&[], &[]);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn who_handles_workspace_with_wgpu_provider() {
        ensure_shared_resolver();
        let _provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("wgpu provider");
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let gpu_value =
            crate::call_builtin("gpuArray", &[Value::Tensor(tensor)]).expect("gpuArray");
        shared_set_workspace(&[("wgpuVar", gpu_value)], &[]);

        let value = who_builtin(Vec::new()).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["wgpuVar".to_string()]);
        shared_set_workspace(&[], &[]);
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
