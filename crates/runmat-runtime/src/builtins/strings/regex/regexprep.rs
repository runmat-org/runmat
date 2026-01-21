//! MATLAB-compatible `regexprep` builtin for RunMat.

use std::sync::Arc;

use regex::{Captures, Regex, RegexBuilder};
use runmat_builtins::{CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "regexprep",
        builtin_path = "crate::builtins::strings::regex::regexprep"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "regexprep"
category: "strings/regex"
keywords: ["regexprep", "regular expression", "replace", "substitute", "regex"]
summary: "Perform MATLAB-compatible regular expression replacements on character vectors, string arrays, or cell arrays."
references:
  - https://www.mathworks.com/help/matlab/ref/regexprep.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs on the CPU. Inputs resident on the GPU are gathered before evaluation, and results remain on the host."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::regex::regexprep::tests"
  integration: "builtins::strings::regex::regexprep::tests::regexprep_elementwise_arrays"
---

# What does the `regexprep` function do in MATLAB / RunMat?
`regexprep(str, pattern, replacement)` searches `str` for regular expression matches and replaces
each match with the specified replacement text. Inputs can be character vectors, string scalars,
string arrays, or cell arrays of text, and the results honour MATLAB's container semantics.

## How does the `regexprep` function behave in MATLAB / RunMat?
- With scalar text inputs, the result is a scalar of the same kind (`char` inputs stay `char`,
  string scalars stay string scalars).
- String arrays and cell arrays are processed element-wise and produce containers with matching
  shapes. When patterns and replacements are provided as arrays of the same size, they are paired
  element-by-element; otherwise, scalar patterns/replacements broadcast to every element.
- Cell or string arrays of patterns apply sequentially: each `(pattern, replacement)` pair is
  applied in order to every element.
- The `'once'` flag limits each element to the first match. `'emptymatch','remove'` (default)
  skips zero-length matches, while `'emptymatch','allow'` lets them participate.
- `'ignorecase'` and `'matchcase'` control case sensitivity. `'lineanchors'`, `'dotall'`, and
  `'dotExceptNewline'` toggle multiline and dot behaviours in the same way as `regexp`.
- `'preservecase'` adjusts the replacement text so the case pattern of the first match (all upper,
  all lower, or title case) is preserved.

## `regexprep` Function GPU Execution Behaviour
`regexprep` executes entirely on the CPU. When the subject, pattern, or replacement values originate
from GPU-resident arrays, RunMat gathers them to host memory before performing the replacements.
Results remain on the host; callers that need GPU residency should explicitly move the values back
afterwards (e.g. with `gpuArray`).

## Examples of using the `regexprep` function in MATLAB / RunMat

### Replacing vowels in a character vector
```matlab
clean = regexprep('abracadabra', '[aeiou]', 'X');
```
Expected output:
```matlab
clean =
    'XbrXcXdXbrX'
```

### Applying multiple pattern/replacement pairs sequentially
```matlab
result = regexprep("Color: red", {'Color', 'red'}, {'Shade', 'blue'});
```
Expected output:
```matlab
result = "Shade: blue"
```

### Performing case-insensitive replacements
```matlab
names = regexprep(["Alpha"; "beta"; "GaMmA"], 'a', '_', 'ignorecase');
```
Expected output:
```matlab
names =
  3×1 string array
    "_lph_"
    "bet_"
    "G__mM_"
```

### Preserving case of the original match
```matlab
words = regexprep('MATLAB and matlab', 'matlab', 'runmat', 'preservecase');
```
Expected output:
```matlab
words =
    'RUNMAT and runmat'
```

### Limiting replacements to the first match with `'once'`
```matlab
out = regexprep("abababa", 'ba', 'XY', 'once');
```
Expected output:
```matlab
out = "aXYbaba"
```

### Using element-wise patterns for a string array
```matlab
expr = ["foo", "bar"];
pat = ["f", "ar"];
rep = ["F", "AR"];
exact = regexprep(expr, pat, rep);
```
Expected output:
```matlab
exact = 1×2 string
    "Foo"    "bAR"
```

## FAQ

### How are container outputs shaped?
Outputs mirror the input container. Character vectors return character vectors, string arrays
return string arrays with the same size, and cell arrays return cell arrays with matching shape.

### Can I supply multiple patterns at once?
Yes. Provide `pattern` and `replacement` as equally-sized cell or string arrays. Each pair is applied
sequentially to every element, mirroring MATLAB's behaviour.

### What if my replacement needs capture-group text?
Use `$1`, `$2`, … for numbered groups or `${name}` for named groups inside the replacement text.
These tokens expand to the corresponding captured substrings.

### Does `regexprep` support case-insensitive matching?
Yes. Pass the `'ignorecase'` option (or `'matchcase'` to revert). You can also request multiline
anchors or dot behaviour changes with `'lineanchors'`, `'dotall'`, or `'dotExceptNewline'`.

### What does `'preservecase'` do?
When enabled, `regexprep` inspects the alphabetic characters in the matched text. If they are all
uppercase, lowercase, or title-cased, the replacement text is adjusted to match that style.

### Does `regexprep` execute on the GPU?
No. All matching and replacement runs on the CPU. GPU inputs are downloaded automatically, but the
results remain in host memory.

## See Also
`regexp`, `regexpi`, `strrep`, `replace`, `contains`

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/regex/regexprep.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/regex/regexprep.rs)
- Found a behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::regex::regexprep")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "regexprep",
    op_kind: GpuOpKind::Custom("regex"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Runs on the CPU; GPU inputs are gathered before processing and results remain on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::regex::regexprep")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "regexprep",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Regex replacements are control-flow heavy and are excluded from fusion.",
};

const BUILTIN_NAME: &str = "regexprep";

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn with_builtin_context(mut err: RuntimeError) -> RuntimeError {
    if err.context.builtin.is_none() {
        err.context = err.context.with_builtin(BUILTIN_NAME);
    }
    err
}

#[runtime_builtin(
    name = "regexprep",
    category = "strings/regex",
    summary = "Regular expression replacement with MATLAB-compatible semantics.",
    keywords = "regexprep,regex,replace,substitute",
    accel = "sink",
    builtin_path = "crate::builtins::strings::regex::regexprep"
)]
async fn regexprep_builtin(
    subject: Value,
    pattern: Value,
    replacement: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let subject = gather_if_needed_async(&subject)
        .await
        .map_err(with_builtin_context)?;
    let pattern = gather_if_needed_async(&pattern)
        .await
        .map_err(with_builtin_context)?;
    let replacement = gather_if_needed_async(&replacement)
        .await
        .map_err(with_builtin_context)?;
    let options = RegexprepOptions::parse(BUILTIN_NAME, &rest)?;

    let subjects = SubjectCollection::collect(BUILTIN_NAME, subject).await?;
    let patterns = PatternCollection::collect(BUILTIN_NAME, pattern, &subjects).await?;
    let replacements = ReplacementCollection::collect(BUILTIN_NAME, replacement, &subjects).await?;
    let plan = build_plan(BUILTIN_NAME, &patterns, &replacements, &options, &subjects)?;

    let mut results = Vec::with_capacity(subjects.entries.len());
    for (idx, original) in subjects.entries.iter().enumerate() {
        let mut current = original.clone();
        if let Some(sequence) = plan.get(idx) {
            for compiled in sequence {
                current = compiled.apply(&current, &options);
            }
        }
        results.push(current);
    }

    reconstruct_output(BUILTIN_NAME, &subjects, results)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmptyMatchPolicy {
    Remove,
    Allow,
}

#[derive(Debug, Clone)]
struct RegexprepOptions {
    case_insensitive: bool,
    multi_line: bool,
    dot_all: bool,
    once: bool,
    emptymatch: EmptyMatchPolicy,
    preserve_case: bool,
}

impl RegexprepOptions {
    fn parse(builtin: &'static str, rest: &[Value]) -> BuiltinResult<Self> {
        let mut case_insensitive = false;
        let mut multi_line = false;
        let mut dot_all = false;
        let mut once = false;
        let mut emptymatch = EmptyMatchPolicy::Remove;
        let mut preserve_case = false;
        let mut idx = 0usize;
        while idx < rest.len() {
            let raw = value_to_lower_string(&rest[idx]).ok_or_else(|| {
                runtime_error_for(format!(
                    "{builtin}: expected option string, got {:?}",
                    rest[idx]
                ))
            })?;
            idx += 1;
            match raw.as_str() {
                "ignorecase" => case_insensitive = true,
                "matchcase" => case_insensitive = false,
                "lineanchors" => {
                    let flag = parse_toggle(rest.get(idx)).ok_or_else(|| {
                        runtime_error_for(format!(
                            "{builtin}: expected logical or 'on'/'off' after 'lineanchors'"
                        ))
                    })?;
                    multi_line = flag;
                    idx += 1;
                }
                "dotall" => {
                    let flag = parse_toggle(rest.get(idx)).ok_or_else(|| {
                        runtime_error_for(format!(
                            "{builtin}: expected logical or 'on'/'off' after 'dotall'"
                        ))
                    })?;
                    dot_all = flag;
                    idx += 1;
                }
                "dotexceptnewline" => {
                    let flag = parse_toggle(rest.get(idx)).ok_or_else(|| {
                        runtime_error_for(format!(
                            "{builtin}: expected logical or 'on'/'off' after 'dotExceptNewline'"
                        ))
                    })?;
                    dot_all = !flag;
                    idx += 1;
                }
                "once" => once = true,
                "emptymatch" => {
                    let value = rest.get(idx).ok_or_else(|| {
                        runtime_error_for(format!(
                            "{builtin}: expected 'allow' or 'remove' after 'emptymatch'"
                        ))
                    })?;
                    let policy = value_to_lower_string(value).ok_or_else(|| {
                        runtime_error_for(format!(
                            "{builtin}: expected 'allow' or 'remove' after 'emptymatch'"
                        ))
                    })?;
                    idx += 1;
                    match policy.as_str() {
                        "allow" => emptymatch = EmptyMatchPolicy::Allow,
                        "remove" => emptymatch = EmptyMatchPolicy::Remove,
                        other => {
                            return Err(runtime_error_for(format!(
                                "{builtin}: invalid emptymatch policy '{other}', expected 'allow' or 'remove'"
                            )))
                        }
                    }
                }
                "preservecase" => {
                    if idx < rest.len() {
                        if let Some(flag) = parse_toggle(rest.get(idx)) {
                            preserve_case = flag;
                            if flag {
                                case_insensitive = true;
                            }
                            idx += 1;
                        } else {
                            return Err(runtime_error_for(format!(
                                "{builtin}: expected logical or 'on'/'off' after 'preservecase'"
                            )));
                        }
                    } else {
                        preserve_case = true;
                        case_insensitive = true;
                    }
                }
                "warnings" | "useparallel" => {
                    if rest.get(idx).is_some() {
                        idx += 1;
                    }
                }
                other => {
                    return Err(runtime_error_for(format!(
                        "{builtin}: unrecognised option '{other}'"
                    )));
                }
            }
        }
        Ok(Self {
            case_insensitive,
            multi_line,
            dot_all,
            once,
            emptymatch,
            preserve_case,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum InputKind {
    CharScalar,
    StringScalar,
    CharMatrix { rows: usize },
    StringArray { rows: usize, cols: usize },
    CellArray { rows: usize, cols: usize },
}

#[derive(Debug, Clone)]
struct SubjectCollection {
    entries: Vec<String>,
    rows: usize,
    cols: usize,
    kind: InputKind,
}

impl SubjectCollection {
    async fn collect(builtin: &'static str, value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Ok(Self {
                entries: vec![s],
                rows: 1,
                cols: 1,
                kind: InputKind::StringScalar,
            }),
            Value::CharArray(array) => collect_char_array(array),
            Value::StringArray(array) => collect_string_array(array),
            Value::Cell(cell) => collect_cell_array(builtin, cell).await,
            other => Err(runtime_error_for(format!(
                "{builtin}: expected char vector, string, string array, or cell array of text, got {other:?}"
            ))),
        }
    }
}

fn collect_char_array(array: CharArray) -> BuiltinResult<SubjectCollection> {
    if array.rows == 0 {
        return Ok(SubjectCollection {
            entries: Vec::new(),
            rows: 0,
            cols: 0,
            kind: InputKind::CharMatrix { rows: 0 },
        });
    }
    if array.rows == 1 {
        let text: String = array.data.into_iter().collect();
        return Ok(SubjectCollection {
            entries: vec![text],
            rows: 1,
            cols: 1,
            kind: InputKind::CharScalar,
        });
    }
    let mut entries = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let mut line = String::with_capacity(array.cols);
        for col in 0..array.cols {
            line.push(array.data[row * array.cols + col]);
        }
        entries.push(line);
    }
    Ok(SubjectCollection {
        entries,
        rows: array.rows,
        cols: 1,
        kind: InputKind::CharMatrix { rows: array.rows },
    })
}

fn collect_string_array(array: StringArray) -> BuiltinResult<SubjectCollection> {
    let rows = array.rows();
    let cols = array.cols();
    let mut entries = Vec::with_capacity(array.data.len());
    for col in 0..cols {
        for row in 0..rows {
            let idx = row + col * rows;
            entries.push(array.data[idx].clone());
        }
    }
    Ok(SubjectCollection {
        entries,
        rows,
        cols,
        kind: InputKind::StringArray { rows, cols },
    })
}

async fn collect_cell_array(
    builtin: &'static str,
    cell: runmat_builtins::CellArray,
) -> BuiltinResult<SubjectCollection> {
    let mut entries = Vec::with_capacity(cell.data.len());
    for ptr in &cell.data {
        let value = gather_if_needed_async(ptr)
            .await
            .map_err(with_builtin_context)?;
        let text = extract_string(&value).ok_or_else(|| {
            runtime_error_for(format!(
                "{builtin}: cell array elements must be character vectors or string scalars, got {value:?}"
            ))
        })?;
        entries.push(text);
    }
    Ok(SubjectCollection {
        entries,
        rows: cell.rows,
        cols: cell.cols,
        kind: InputKind::CellArray {
            rows: cell.rows,
            cols: cell.cols,
        },
    })
}

#[derive(Debug, Clone)]
enum PatternCollection {
    Scalar(String),
    Sequence(Vec<String>),
    Elementwise(Vec<String>),
}

impl PatternCollection {
    async fn collect(
        builtin: &'static str,
        value: Value,
        subjects: &SubjectCollection,
    ) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Ok(Self::Scalar(s)),
            Value::CharArray(array) => collect_pattern_char_array(array),
            Value::StringArray(array) => collect_pattern_string_array(array, subjects),
            Value::Cell(cell) => collect_pattern_cell(builtin, cell, subjects).await,
            other => Err(runtime_error_for(format!(
                "{builtin}: expected char vector, string array, or cell array for pattern, got {other:?}"
            ))),
        }
    }
}

fn collect_pattern_char_array(array: CharArray) -> BuiltinResult<PatternCollection> {
    if array.rows <= 1 {
        let text: String = array.data.into_iter().collect();
        return Ok(PatternCollection::Scalar(text));
    }
    let mut entries = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let mut line = String::with_capacity(array.cols);
        for col in 0..array.cols {
            line.push(array.data[row * array.cols + col]);
        }
        entries.push(line);
    }
    Ok(PatternCollection::Sequence(entries))
}

fn collect_pattern_string_array(
    array: StringArray,
    subjects: &SubjectCollection,
) -> BuiltinResult<PatternCollection> {
    let rows = array.rows();
    let cols = array.cols();
    if rows == 1 && cols == 1 && !array.data.is_empty() {
        return Ok(PatternCollection::Scalar(array.data[0].clone()));
    }
    if rows == subjects.rows && cols == subjects.cols {
        return Ok(PatternCollection::Elementwise(array.data.clone()));
    }
    Ok(PatternCollection::Sequence(array.data.clone()))
}

async fn collect_pattern_cell(
    builtin: &'static str,
    cell: runmat_builtins::CellArray,
    subjects: &SubjectCollection,
) -> BuiltinResult<PatternCollection> {
    let mut entries = Vec::with_capacity(cell.data.len());
    for ptr in &cell.data {
        let value = gather_if_needed_async(ptr)
            .await
            .map_err(with_builtin_context)?;
        let text = extract_string(&value).ok_or_else(|| {
            runtime_error_for(format!(
                "{builtin}: pattern cell elements must be character vectors or string scalars, got {value:?}"
            ))
        })?;
        entries.push(text);
    }
    if cell.rows == subjects.rows && cell.cols == subjects.cols {
        return Ok(PatternCollection::Elementwise(entries));
    }
    Ok(PatternCollection::Sequence(entries))
}

#[derive(Debug, Clone)]
enum ReplacementCollection {
    Scalar(String),
    Sequence(Vec<String>),
    Elementwise(Vec<String>),
}

impl ReplacementCollection {
    async fn collect(
        builtin: &'static str,
        value: Value,
        subjects: &SubjectCollection,
    ) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Ok(Self::Scalar(s)),
            Value::CharArray(array) => collect_replacement_char_array(array),
            Value::StringArray(array) => collect_replacement_string_array(array, subjects),
            Value::Cell(cell) => collect_replacement_cell(builtin, cell, subjects).await,
            other => Err(runtime_error_for(format!(
                "{builtin}: expected char vector, string array, or cell array for replacement, got {other:?}"
            ))),
        }
    }
}

fn collect_replacement_char_array(array: CharArray) -> BuiltinResult<ReplacementCollection> {
    if array.rows <= 1 {
        let text: String = array.data.into_iter().collect();
        return Ok(ReplacementCollection::Scalar(text));
    }
    let mut entries = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let mut line = String::with_capacity(array.cols);
        for col in 0..array.cols {
            line.push(array.data[row * array.cols + col]);
        }
        entries.push(line);
    }
    Ok(ReplacementCollection::Sequence(entries))
}

fn collect_replacement_string_array(
    array: StringArray,
    subjects: &SubjectCollection,
) -> BuiltinResult<ReplacementCollection> {
    let rows = array.rows();
    let cols = array.cols();
    if rows == 1 && cols == 1 && !array.data.is_empty() {
        return Ok(ReplacementCollection::Scalar(array.data[0].clone()));
    }
    if rows == subjects.rows && cols == subjects.cols {
        return Ok(ReplacementCollection::Elementwise(array.data.clone()));
    }
    Ok(ReplacementCollection::Sequence(array.data.clone()))
}

async fn collect_replacement_cell(
    builtin: &'static str,
    cell: runmat_builtins::CellArray,
    subjects: &SubjectCollection,
) -> BuiltinResult<ReplacementCollection> {
    let mut entries = Vec::with_capacity(cell.data.len());
    for ptr in &cell.data {
        let value = gather_if_needed_async(ptr)
            .await
            .map_err(with_builtin_context)?;
        let text = extract_string(&value).ok_or_else(|| {
            runtime_error_for(format!(
                "{builtin}: replacement cell elements must be character vectors or string scalars, got {value:?}"
            ))
        })?;
        entries.push(text);
    }
    if cell.rows == subjects.rows && cell.cols == subjects.cols {
        return Ok(ReplacementCollection::Elementwise(entries));
    }
    Ok(ReplacementCollection::Sequence(entries))
}

#[derive(Clone)]
struct ReplacementTemplate {
    template: Arc<String>,
}

impl ReplacementTemplate {
    fn new(raw: &str) -> Self {
        Self {
            template: Arc::new(matlab_template_to_rust(raw)),
        }
    }

    fn expand(&self, caps: &Captures<'_>, preserve_case: bool) -> String {
        let mut out = String::new();
        caps.expand(self.template.as_str(), &mut out);
        if preserve_case {
            if let Some(mat) = caps.get(0) {
                return apply_preserve_case(mat.as_str(), &out);
            }
        }
        out
    }
}

#[derive(Clone)]
struct CompiledReplacement {
    regex: Arc<Regex>,
    template: ReplacementTemplate,
}

impl CompiledReplacement {
    fn new(
        builtin: &'static str,
        pattern: &str,
        replacement: &str,
        options: &RegexprepOptions,
    ) -> BuiltinResult<Self> {
        let regex = compile_regex(builtin, pattern, options)?;
        Ok(Self {
            regex,
            template: ReplacementTemplate::new(replacement),
        })
    }

    fn with_template(regex: Arc<Regex>, template: ReplacementTemplate) -> BuiltinResult<Self> {
        Ok(Self { regex, template })
    }

    fn apply(&self, text: &str, options: &RegexprepOptions) -> String {
        replace_with_policy(text, &self.regex, &self.template, options)
    }
}

fn compile_regex(
    builtin: &'static str,
    pattern: &str,
    options: &RegexprepOptions,
) -> BuiltinResult<Arc<Regex>> {
    let mut builder = RegexBuilder::new(pattern);
    if options.case_insensitive {
        builder.case_insensitive(true);
    }
    if options.multi_line {
        builder.multi_line(true);
    }
    if options.dot_all {
        builder.dot_matches_new_line(true);
    }
    builder
        .build()
        .map(Arc::new)
        .map_err(|e| runtime_error_for(format!("{builtin}: {e}")))
}

fn build_plan(
    builtin: &'static str,
    patterns: &PatternCollection,
    replacements: &ReplacementCollection,
    options: &RegexprepOptions,
    subjects: &SubjectCollection,
) -> BuiltinResult<Vec<Vec<CompiledReplacement>>> {
    let subject_len = subjects.entries.len();
    match patterns {
        PatternCollection::Scalar(pattern) => match replacements {
            ReplacementCollection::Scalar(text) => {
                let compiled = CompiledReplacement::new(builtin, pattern, text, options)?;
                Ok(vec![vec![compiled]; subject_len])
            }
            ReplacementCollection::Elementwise(values) => {
                if values.len() != subject_len {
                    return Err(runtime_error_for(format!(
                        "{builtin}: replacement array must match the subject size"
                    )));
                }
                let regex = compile_regex(builtin, pattern, options)?;
                let mut plan = Vec::with_capacity(subject_len);
                for value in values {
                    let template = ReplacementTemplate::new(value);
                    plan.push(vec![CompiledReplacement::with_template(
                        regex.clone(),
                        template,
                    )?]);
                }
                Ok(plan)
            }
            ReplacementCollection::Sequence(values) => {
                if values.len() != 1 {
                    return Err(runtime_error_for(format!(
                        "{builtin}: replacement sequence must match pattern sequence length"
                    )));
                }
                let compiled = CompiledReplacement::new(builtin, pattern, &values[0], options)?;
                Ok(vec![vec![compiled]; subject_len])
            }
        },
        PatternCollection::Sequence(patterns_vec) => {
            if patterns_vec.is_empty() {
                return Ok(vec![Vec::new(); subject_len]);
            }
            let sequence = match replacements {
                ReplacementCollection::Scalar(text) => {
                    let mut seq = Vec::with_capacity(patterns_vec.len());
                    for pattern in patterns_vec {
                        seq.push(CompiledReplacement::new(builtin, pattern, text, options)?);
                    }
                    seq
                }
                ReplacementCollection::Sequence(values) => {
                    if patterns_vec.len() != values.len() {
                        return Err(runtime_error_for(format!(
                            "{builtin}: pattern and replacement sequence lengths must match"
                        )));
                    }
                    let mut seq = Vec::with_capacity(patterns_vec.len());
                    for (pattern, value) in patterns_vec.iter().zip(values.iter()) {
                        seq.push(CompiledReplacement::new(builtin, pattern, value, options)?);
                    }
                    seq
                }
                ReplacementCollection::Elementwise(_) => {
                    return Err(runtime_error_for(format!(
                        "{builtin}: element-wise replacements require element-wise patterns"
                    )));
                }
            };
            Ok(vec![sequence.clone(); subject_len])
        }
        PatternCollection::Elementwise(values) => {
            if values.len() != subject_len {
                return Err(runtime_error_for(format!(
                    "{builtin}: pattern array must match the subject size"
                )));
            }
            match replacements {
                ReplacementCollection::Elementwise(reps) => {
                    if reps.len() != subject_len {
                        return Err(runtime_error_for(format!(
                            "{builtin}: replacement array must match the subject size"
                        )));
                    }
                    let mut plan = Vec::with_capacity(subject_len);
                    for (pattern, rep) in values.iter().zip(reps.iter()) {
                        let compiled = CompiledReplacement::new(builtin, pattern, rep, options)?;
                        plan.push(vec![compiled]);
                    }
                    Ok(plan)
                }
                ReplacementCollection::Scalar(rep) => {
                    let mut plan = Vec::with_capacity(subject_len);
                    for pattern in values {
                        let compiled = CompiledReplacement::new(builtin, pattern, rep, options)?;
                        plan.push(vec![compiled]);
                    }
                    Ok(plan)
                }
                ReplacementCollection::Sequence(_) => Err(runtime_error_for(format!(
                    "{builtin}: replacement sequence is incompatible with element-wise patterns"
                ))),
            }
        }
    }
}

fn replace_with_policy(
    text: &str,
    regex: &Regex,
    template: &ReplacementTemplate,
    options: &RegexprepOptions,
) -> String {
    let mut output = String::new();
    let mut last = 0usize;
    let mut replaced = 0usize;
    for caps in regex.captures_iter(text) {
        let mat = match caps.get(0) {
            Some(m) => m,
            None => continue,
        };
        if options.emptymatch == EmptyMatchPolicy::Remove && mat.start() == mat.end() {
            continue;
        }
        output.push_str(&text[last..mat.start()]);
        let replacement = template.expand(&caps, options.preserve_case);
        output.push_str(&replacement);
        last = mat.end();
        replaced += 1;
        if options.once {
            break;
        }
    }
    if replaced == 0 {
        return text.to_string();
    }
    output.push_str(&text[last..]);
    output
}

fn reconstruct_output(
    builtin: &'static str,
    subjects: &SubjectCollection,
    results: Vec<String>,
) -> BuiltinResult<Value> {
    match subjects.kind {
        InputKind::CharScalar => {
            let array = strings_to_char_array(builtin, 1, &results)?;
            Ok(Value::CharArray(array))
        }
        InputKind::StringScalar => Ok(Value::String(
            results.into_iter().next().unwrap_or_default(),
        )),
        InputKind::CharMatrix { rows } => {
            let array = strings_to_char_array(builtin, rows, &results)?;
            Ok(Value::CharArray(array))
        }
        InputKind::StringArray { rows, cols } => {
            let array = StringArray::new(results, vec![rows, cols])
                .map_err(|e| runtime_error_for(format!("{builtin}: {e}")))?;
            Ok(Value::StringArray(array))
        }
        InputKind::CellArray { rows, cols } => {
            let values = results.into_iter().map(Value::String).collect();
            make_cell(values, rows, cols)
                .map_err(|err| runtime_error_for(format!("{builtin}: {err}")))
        }
    }
}

fn strings_to_char_array(
    builtin: &'static str,
    rows: usize,
    strings: &[String],
) -> BuiltinResult<CharArray> {
    if rows == 0 {
        return CharArray::new(Vec::new(), 0, 0)
            .map_err(|e| runtime_error_for(format!("{builtin}: {e}")));
    }
    let max_cols = strings
        .iter()
        .take(rows)
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(rows * max_cols);
    for idx in 0..rows {
        let text = strings.get(idx).cloned().unwrap_or_default();
        let mut chars: Vec<char> = text.chars().collect();
        while chars.len() < max_cols {
            chars.push(' ');
        }
        data.extend(chars);
    }
    CharArray::new(data, rows, max_cols).map_err(|e| runtime_error_for(format!("{builtin}: {e}")))
}

fn matlab_template_to_rust(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let mut out = String::with_capacity(raw.len());
    let mut idx = 0usize;
    while idx < chars.len() {
        let ch = chars[idx];
        if ch == '\\' {
            if idx + 1 < chars.len() {
                let next = chars[idx + 1];
                if next.is_ascii_digit() {
                    out.push('$');
                    out.push(next);
                    idx += 2;
                    continue;
                } else if next == '\\' {
                    out.push('\\');
                    idx += 2;
                    continue;
                }
            }
            out.push('\\');
            idx += 1;
            continue;
        }
        if ch == '$' && idx + 1 < chars.len() && chars[idx + 1] == '{' {
            let mut end = idx + 2;
            while end < chars.len() && chars[end] != '}' {
                end += 1;
            }
            if end < chars.len() {
                let name: String = chars[idx + 2..end].iter().collect();
                out.push('$');
                out.push_str(&name);
                idx = end + 1;
                continue;
            }
        }
        out.push(ch);
        idx += 1;
    }
    out
}

fn apply_preserve_case(source: &str, replacement: &str) -> String {
    match classify_case(source) {
        CaseStyle::Upper => to_uppercase_full(replacement),
        CaseStyle::Lower => replacement.to_lowercase(),
        CaseStyle::Title => to_title_case(replacement),
        CaseStyle::Mixed => replacement.to_string(),
    }
}

#[derive(Debug, Clone, Copy)]
enum CaseStyle {
    Upper,
    Lower,
    Title,
    Mixed,
}

fn classify_case(source: &str) -> CaseStyle {
    let mut has_alpha = false;
    let mut all_upper = true;
    let mut all_lower = true;
    let mut title = true;
    let mut first_alpha_seen = false;
    for ch in source.chars() {
        if !ch.is_alphabetic() {
            continue;
        }
        has_alpha = true;
        if ch.is_uppercase() {
            all_lower = false;
        } else {
            all_upper = false;
        }
        if !first_alpha_seen {
            if !ch.is_uppercase() {
                title = false;
            }
            first_alpha_seen = true;
        } else if ch.is_uppercase() {
            title = false;
        }
    }
    if !has_alpha {
        CaseStyle::Mixed
    } else if all_upper {
        CaseStyle::Upper
    } else if all_lower {
        CaseStyle::Lower
    } else if title {
        CaseStyle::Title
    } else {
        CaseStyle::Mixed
    }
}

fn to_uppercase_full(text: &str) -> String {
    text.chars().flat_map(|c| c.to_uppercase()).collect()
}

fn to_title_case(text: &str) -> String {
    let lower = text.to_lowercase();
    let mut result = String::with_capacity(lower.len());
    let mut first_done = false;
    for ch in lower.chars() {
        if !first_done && ch.is_alphabetic() {
            for upper in ch.to_uppercase() {
                result.push(upper);
            }
            first_done = true;
        } else {
            result.push(ch);
        }
    }
    result
}

fn extract_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn value_to_lower_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            Some(ca.data.iter().collect::<String>().to_ascii_lowercase())
        }
        _ => None,
    }
}

fn parse_toggle(value: Option<&Value>) -> Option<bool> {
    let value = value?;
    match value {
        Value::Bool(b) => Some(*b),
        Value::Int(i) => Some(!i.is_zero()),
        Value::Num(n) => Some(*n != 0.0),
        _ => {
            let text = value_to_lower_string(value)?;
            match text.as_str() {
                "on" | "true" | "yes" | "1" => Some(true),
                "off" | "false" | "no" | "0" => Some(false),
                _ => None,
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    fn run_regexprep(
        subject: Value,
        pattern: Value,
        replacement: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        futures::executor::block_on(regexprep_builtin(subject, pattern, replacement, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_basic_replacement() {
        let result = run_regexprep(
            Value::String("abracadabra".into()),
            Value::String("[aeiou]".into()),
            Value::String("X".into()),
            Vec::new(),
        )
        .expect("regexprep");
        match result {
            Value::String(s) => assert_eq!(s, "XbrXcXdXbrX"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_sequence_patterns() {
        let subject =
            Value::StringArray(StringArray::new(vec!["Color: red".into()], vec![1, 1]).unwrap());
        let patterns = Value::Cell(
            runmat_builtins::CellArray::new(
                vec![Value::String("Color".into()), Value::String("red".into())],
                1,
                2,
            )
            .unwrap(),
        );
        let replacements = Value::Cell(
            runmat_builtins::CellArray::new(
                vec![Value::String("Shade".into()), Value::String("blue".into())],
                1,
                2,
            )
            .unwrap(),
        );
        let result =
            run_regexprep(subject, patterns, replacements, Vec::new()).expect("regexprep");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data.len(), 1);
                assert_eq!(sa.data[0], "Shade: blue");
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_ignore_case() {
        let result = run_regexprep(
            Value::StringArray(
                StringArray::new(vec!["Alpha".into(), "beta".into()], vec![2, 1]).unwrap(),
            ),
            Value::String("a".into()),
            Value::String("_".into()),
            vec![Value::String("ignorecase".into())],
        )
        .expect("regexprep");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data, vec![String::from("_lph_"), String::from("bet_")]);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_preserve_case() {
        let result = run_regexprep(
            Value::String("MATLAB and matlab".into()),
            Value::String("matlab".into()),
            Value::String("runmat".into()),
            vec![Value::String("preservecase".into())],
        )
        .expect("regexprep");
        match result {
            Value::String(s) => assert_eq!(s, "RUNMAT and runmat"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_once_option() {
        let result = run_regexprep(
            Value::StringArray(StringArray::new(vec!["abababa".into()], vec![1, 1]).unwrap()),
            Value::String("ba".into()),
            Value::String("XY".into()),
            vec![Value::String("once".into())],
        )
        .expect("regexprep");
        match result {
            Value::StringArray(sa) => assert_eq!(sa.data[0], "aXYbaba"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_elementwise_arrays() {
        let subject = Value::StringArray(
            StringArray::new(vec!["cat".into(), "dog".into()], vec![2, 1]).unwrap(),
        );
        let patterns =
            Value::StringArray(StringArray::new(vec!["a".into(), "o".into()], vec![2, 1]).unwrap());
        let replacements =
            Value::StringArray(StringArray::new(vec!["A".into(), "O".into()], vec![2, 1]).unwrap());
        let result =
            run_regexprep(subject, patterns, replacements, Vec::new()).expect("regexprep");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data, vec!["cAt".to_string(), "dOg".to_string()]);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_emptymatch_policy() {
        let subject = Value::String("abc".into());
        let pattern = Value::String("^".into());
        let replacement = Value::String("X".into());

        let unchanged = run_regexprep(
            subject.clone(),
            pattern.clone(),
            replacement.clone(),
            Vec::new(),
        )
        .expect("regexprep");
        assert_eq!(unchanged, Value::String("abc".into()));

        let allowed = run_regexprep(
            subject,
            pattern,
            replacement,
            vec![
                Value::String("emptymatch".into()),
                Value::String("allow".into()),
            ],
        )
        .expect("regexprep");
        assert_eq!(allowed, Value::String("Xabc".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_char_array_result() {
        let chars = CharArray::new(vec!['c', 'a', 't', 'd', 'o', 'g'], 2, 3).unwrap();
        let result = run_regexprep(
            Value::CharArray(chars),
            Value::String("[ao]".into()),
            Value::String("X".into()),
            Vec::new(),
        )
        .expect("regexprep");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 3);
                assert_eq!(out.data, vec!['c', 'X', 't', 'd', 'X', 'g']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_invalid_option_errors() {
        let err = run_regexprep(
            Value::String("test".into()),
            Value::String("t".into()),
            Value::String("x".into()),
            vec![Value::String("unknownOption".into())],
        )
        .expect_err("expected error");
        let message = err.message().to_string();
        assert!(
            message.contains("unrecognised option"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_boolean_name_value_pairs() {
        let subject = Value::String("foo\nbar".into());
        let pattern = Value::String("^bar".into());
        let replacement = Value::String("BAR".into());
        let anchored = run_regexprep(
            subject.clone(),
            pattern.clone(),
            replacement.clone(),
            vec![Value::String("lineanchors".into()), Value::Bool(true)],
        )
        .expect("regexprep");
        assert_eq!(anchored, Value::String("foo\nBAR".into()));

        let dotall = run_regexprep(
            Value::String("foo\nbar".into()),
            Value::String("foo.bar".into()),
            Value::String("HIT".into()),
            vec![Value::String("dotall".into()), Value::Bool(true)],
        )
        .expect("regexprep");
        assert_eq!(dotall, Value::String("HIT".into()));

        let preserve = run_regexprep(
            Value::String("Matlab".into()),
            Value::String("matlab".into()),
            Value::String("runmat".into()),
            vec![Value::String("preservecase".into()), Value::Bool(true)],
        )
        .expect("regexprep");
        assert_eq!(preserve, Value::String("Runmat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_cell_subject_outputs_cell() {
        let cell = runmat_builtins::CellArray::new(
            vec![Value::String("cat".into()), Value::String("dog".into())],
            2,
            1,
        )
        .unwrap();
        let result = run_regexprep(
            Value::Cell(cell),
            Value::String("[ao]".into()),
            Value::String("_".into()),
            Vec::new(),
        )
        .expect("regexprep");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 1);
                let first = out.get(0, 0).expect("cell value");
                let second = out.get(1, 0).expect("cell value");
                assert_eq!(first, Value::String("c_t".into()));
                assert_eq!(second, Value::String("d_g".into()));
            }
            other => panic!("expected cell, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_elementwise_mismatch_errors() {
        let subject = Value::StringArray(
            StringArray::new(vec!["cat".into(), "dog".into()], vec![2, 1]).unwrap(),
        );
        let patterns =
            Value::StringArray(StringArray::new(vec!["a".into(), "o".into()], vec![2, 1]).unwrap());
        let replacements = Value::StringArray(
            StringArray::new(vec!["A".into(), "O".into(), "U".into()], vec![3, 1]).unwrap(),
        );
        let err = run_regexprep(subject, patterns, replacements, Vec::new())
            .expect_err("expected error");
        let message = err.message().to_string();
        assert!(
            message.contains("replacement sequence is incompatible with element-wise patterns"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
