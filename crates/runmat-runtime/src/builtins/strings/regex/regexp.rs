//! MATLAB-compatible `regexp` builtin for RunMat.

use std::collections::HashMap;

use regex::RegexBuilder;
use runmat_builtins::{CharArray, StringArray, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{gather_if_needed, make_cell, register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "regexp"
category: "strings/regex"
keywords: ["regexp", "regular expression", "pattern", "match", "split", "tokens"]
summary: "Search text using regular expressions, returning match positions, substrings, tokens, or splits with MATLAB-compatible semantics."
references:
  - https://www.mathworks.com/help/matlab/ref/regexp.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the CPU. When inputs or outputs reside on the GPU, RunMat gathers data before matching and re-uploads cell contents after evaluation."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::regex::regexp::tests"
  integration: "builtins::strings::regex::regexp::tests::regexp_multi_output_default"
---

# What does the `regexp` function do in MATLAB / RunMat?
`regexp(text, pattern)` locates regular expression matches inside character vectors, string scalars,
string arrays, and cell arrays of character vectors. It can return starting indices, ending indices,
matched substrings, capture tokens, token extents, named-token structures, or the text split around
matches. The optional `'once'` flag restricts the search to the first match, while
`'emptymatch','allow'` keeps zero-length matches that otherwise get filtered out.

## How does the `regexp` function behave in MATLAB / RunMat?
- Single character vectors and string scalars return a numeric row vector of 1-based match start
  indices by default.
- String arrays and cell arrays always produce cell outputs that mirror the input shape, with each
  cell holding the result for the corresponding element.
- `'match'` returns matched substrings, `'tokens'` returns nested cells of capture-group substrings,
  `'tokenExtents'` returns `n × 2` double matrices with start/end indices for each token,
  `'names'` returns scalar struct values keyed by named tokens, and `'split'` yields the text segments
  between matches.
- `'once'` stops after the first match (per element), and every requested output honours that limit.
- `'emptymatch','remove'` (default) filters zero-length matches; `'emptymatch','allow'` keeps them
  so callers can observe optional patterns.
- `'forceCellOutput'` forces cell-array containers even for scalar inputs so downstream code can
  rely on uniform dimensions. MATLAB-compatible `'warnings','on'/'off'` flags are accepted but
  currently informational only.
- `'matchcase'` and `'ignorecase'` toggle case sensitivity, while `'lineanchors'` (`^`/`$`) and
  `'dotall'`/`'dotExceptNewline'` control how `.` interacts with newlines, mirroring MATLAB flags.

## `regexp` Function GPU Execution Behaviour
`regexp` executes entirely on the CPU and is registered as an acceleration sink. If any argument
resides on the GPU, the runtime gathers it before evaluation, computes all requested outputs on the
host, and returns host-side containers. Providers do not implement custom hooks for this builtin, so
no GPU kernels are required or invoked.

## Examples of using the `regexp` function in MATLAB / RunMat

### Find all 1-based match positions in a character vector
```matlab
idx = regexp('abracadabra', 'a');
```
Expected output:
```matlab
idx =
     1     4     6     8    11
```

### Return matched substrings using `'match'`
```matlab
matches = regexp('abc123xyz', '\d+', 'match');
```
Expected output:
```matlab
matches =
  1×1 cell array
    {'123'}
```

### Extract capture tokens
```matlab
tokens = regexp('2024-03-14', '(\d{4})-(\d{2})-(\d{2})', 'tokens');
year = tokens{1}{1};
month = tokens{1}{2};
day = tokens{1}{3};
```
Expected output:
```matlab
year =
    '2024'
month =
    '03'
day =
    '14'
```

### Split a string array around commas
```matlab
parts = regexp(["a,b,c"; "1,2,3"], ',', 'split');
```
Expected output:
```matlab
parts =
  2×1 cell array
    {1×3 cell}
    {1×3 cell}
```

### Return only the first match with `'once'`
```matlab
first_idx = regexp('abababa', 'ba', 'once');
```
Expected output:
```matlab
first_idx =
     2
```

### Work with named tokens
```matlab
matches = regexp('X=42; Y=7;', '(?<name>[A-Z])=(?<value>\d+)', 'names');
values = cellfun(@(s) str2double(s.value), matches);
```
Expected output:
```matlab
values =
     42     7
```

### Keep zero-length matches with `'emptymatch','allow'`
```matlab
idx = regexp('aba', 'b*', 'emptymatch', 'allow');
```
Expected output:
```matlab
idx =
     1     2     3     4
```

## FAQ

### What outputs does `regexp` return by default?
With a single output argument, `regexp` returns a numeric row vector of 1-based match starts.
When the call site asks for multiple outputs (e.g. `[startIdx, endIdx, matchStr] = regexp(...)`),
RunMat returns match starts, match ends, and matched substrings in that order, just like MATLAB.

### How can I request tokens or splits instead of indices?
Specify the desired output types as string flags, for example
`regexp(str, pat, 'match')`, `regexp(str, pat, 'tokens')`, or `regexp(str, pat, 'split')`.
Multiple flags combine, so `regexp(str, pat, 'match', 'tokens')` returns both outputs.

### Does `regexp` support case-insensitive matching?
Yes. Use `'ignorecase'` (or call `regexpi`) to enable case-insensitive matching, and `'matchcase'`
to revert to the default case-sensitive behaviour.

### How are string arrays and cell arrays handled?
For string arrays and cell arrays of char vectors, every output is a cell array whose shape matches
the input. Each cell contains the result for the corresponding element, which mirrors MATLAB's
container semantics.

### How do zero-length matches behave?
By default (`'emptymatch','remove'`), zero-length matches are filtered out so loops do not stall.
Specify `'emptymatch','allow'` to keep them, matching MATLAB's `'emptymatch'` flag.

### Can I force cell output even for character vectors?
Yes. Pass `'forceCellOutput'` to force the outputs into cell arrays, which is useful when writing
code that handles both scalar and array inputs uniformly.

### Does `regexp` run on the GPU?
No. RunMat executes `regexp` on the CPU. If inputs reside on the GPU, it gathers them first and then
re-uploads any numeric outputs when beneficial, preserving residency for downstream kernels.

### What happens when I ask for more outputs than I requested via flags?
RunMat follows MATLAB's rules: if you do not supply explicit output flags, the default multi-output
order is start indices, end indices, and matched substrings. Extra requested outputs beyond what you
specified become numeric zeros.

## See Also
`regexpi`, `regexprep`, `contains`, `split`, `strfind`

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/regex/regexp.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/regex/regexp.rs)
- Found a difference from MATLAB? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "regexp",
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
    notes: "Runs on the CPU; when inputs live on the GPU, the runtime gathers them before matching and re-uploads numeric tensors afterwards.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "regexp",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Regex evaluation is control-flow heavy and not eligible for fusion today.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("regexp", DOC_MD);

/// Evaluate a regular expression and return an evaluation handle that can produce MATLAB-compatible outputs.
pub fn evaluate(
    subject: Value,
    pattern: Value,
    rest: &[Value],
) -> Result<RegexpEvaluation, String> {
    let subject = gather_if_needed(&subject).map_err(|e| format!("regexp: {e}"))?;
    let pattern = gather_if_needed(&pattern).map_err(|e| format!("regexp: {e}"))?;
    let options = RegexpOptions::parse(rest)?;
    RegexpEvaluation::new(subject, pattern, options)
}

#[runtime_builtin(
    name = "regexp",
    category = "strings/regex",
    summary = "Regular expression matching with MATLAB-compatible outputs.",
    keywords = "regexp,regex,pattern,match,tokens,split",
    accel = "sink"
)]
fn regexp_builtin(subject: Value, pattern: Value, rest: Vec<Value>) -> Result<Value, String> {
    let evaluation = evaluate(subject, pattern, &rest)?;
    let mut outputs = evaluation.outputs_for_single()?;
    if outputs.is_empty() {
        return Ok(Value::Num(0.0));
    }
    if outputs.len() == 1 {
        Ok(outputs.remove(0))
    } else {
        let len = outputs.len();
        make_cell(outputs, 1, len)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputSpec {
    Start,
    End,
    Match,
    Tokens,
    TokenExtents,
    Names,
    Split,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmptyMatchPolicy {
    Remove,
    Allow,
}

#[derive(Debug, Clone)]
struct RegexpOptions {
    outputs: Vec<OutputSpec>,
    once: bool,
    emptymatch: EmptyMatchPolicy,
    force_cell_output: bool,
    case_insensitive: bool,
    multi_line: bool,
    dot_all: bool,
}

impl RegexpOptions {
    fn parse(rest: &[Value]) -> Result<Self, String> {
        let mut outputs = Vec::new();
        let mut once = false;
        let mut emptymatch = EmptyMatchPolicy::Remove;
        let mut force_cell_output = false;
        let mut case_insensitive = false;
        let mut multi_line = false;
        let mut dot_all = false;
        let mut idx = 0usize;
        while idx < rest.len() {
            let raw = value_to_lower_string(&rest[idx])
                .ok_or_else(|| format!("regexp: expected option string, got {:?}", rest[idx]))?;
            idx += 1;
            match raw.as_str() {
                "match" => outputs.push(OutputSpec::Match),
                "tokens" => outputs.push(OutputSpec::Tokens),
                "tokenextents" => outputs.push(OutputSpec::TokenExtents),
                "names" => outputs.push(OutputSpec::Names),
                "split" => outputs.push(OutputSpec::Split),
                "start" => outputs.push(OutputSpec::Start),
                "end" => outputs.push(OutputSpec::End),
                "once" => once = true,
                "forcecelloutput" => force_cell_output = true,
                "ignorecase" => case_insensitive = true,
                "matchcase" => case_insensitive = false,
                "lineanchors" => {
                    let flag = parse_on_off(rest.get(idx)).ok_or_else(|| {
                        "regexp: expected 'on' or 'off' after 'lineanchors'".to_string()
                    })?;
                    multi_line = flag;
                    idx += 1;
                }
                "dotall" => {
                    let flag = parse_on_off(rest.get(idx)).ok_or_else(|| {
                        "regexp: expected 'on' or 'off' after 'dotall'".to_string()
                    })?;
                    dot_all = flag;
                    idx += 1;
                }
                "dotexceptnewline" => {
                    let flag = parse_on_off(rest.get(idx)).ok_or_else(|| {
                        "regexp: expected 'on' or 'off' after 'dotExceptNewline'".to_string()
                    })?;
                    dot_all = !flag;
                    idx += 1;
                }
                "emptymatch" => {
                    let policy = rest.get(idx).ok_or_else(|| {
                        "regexp: expected 'allow' or 'remove' after 'emptymatch'".to_string()
                    })?;
                    let policy_str = value_to_lower_string(policy).ok_or_else(|| {
                        "regexp: expected 'allow' or 'remove' after 'emptymatch'".to_string()
                    })?;
                    idx += 1;
                    match policy_str.as_str() {
                        "allow" => emptymatch = EmptyMatchPolicy::Allow,
                        "remove" => emptymatch = EmptyMatchPolicy::Remove,
                        other => {
                            return Err(format!(
                                "regexp: invalid emptymatch policy '{other}', expected 'allow' or 'remove'"
                            ))
                        }
                    }
                }
                "warnings" => {
                    // MATLAB accepts 'on'/'off' here but RunMat currently ignores it.
                    if rest.get(idx).is_some() {
                        idx += 1;
                    }
                }
                other => {
                    return Err(format!("regexp: unrecognised option '{other}'"));
                }
            }
        }
        Ok(Self {
            outputs,
            once,
            emptymatch,
            force_cell_output,
            case_insensitive,
            multi_line,
            dot_all,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputKind {
    CharScalar,
    StringScalar,
    CharMatrix { rows: usize, cols: usize },
    StringArray { rows: usize, cols: usize },
    CellArray { rows: usize, cols: usize },
}

#[derive(Debug, Clone)]
struct SubjectCollection {
    entries: Vec<String>,
    rows: usize,
    cols: usize,
    kind: InputKind,
    force_cell_output: bool,
}

impl SubjectCollection {
    fn is_scalar_text(&self) -> bool {
        match self.kind {
            InputKind::CharScalar | InputKind::StringScalar => !self.force_cell_output,
            _ => false,
        }
    }
}

fn collect_subjects(value: Value, force_cell_output: bool) -> Result<SubjectCollection, String> {
    match value {
        Value::String(s) => Ok(SubjectCollection {
            entries: vec![s],
            rows: 1,
            cols: 1,
            kind: InputKind::CharScalar,
            force_cell_output,
        }),
        Value::CharArray(array) => collect_char_array(array, force_cell_output),
        Value::StringArray(array) => collect_string_array(array, force_cell_output),
        Value::Cell(cell) => collect_cell_array(cell, force_cell_output),
        other => Err(format!(
            "regexp: expected char vector, string, string array, or cell array of char vectors, got {other:?}"
        )),
    }
}

fn collect_char_array(
    array: CharArray,
    force_cell_output: bool,
) -> Result<SubjectCollection, String> {
    if array.rows == 0 {
        return Ok(SubjectCollection {
            entries: Vec::new(),
            rows: 0,
            cols: 0,
            kind: InputKind::CharMatrix {
                rows: 0,
                cols: array.cols,
            },
            force_cell_output: true,
        });
    }
    if array.rows == 1 {
        let text: String = array.data.into_iter().collect();
        return Ok(SubjectCollection {
            entries: vec![text],
            rows: 1,
            cols: 1,
            kind: InputKind::CharScalar,
            force_cell_output,
        });
    }
    let mut entries = Vec::with_capacity(array.rows);
    for r in 0..array.rows {
        let mut line = String::with_capacity(array.cols);
        for c in 0..array.cols {
            line.push(array.data[r * array.cols + c]);
        }
        entries.push(line);
    }
    Ok(SubjectCollection {
        entries,
        rows: array.rows,
        cols: 1,
        kind: InputKind::CharMatrix {
            rows: array.rows,
            cols: array.cols,
        },
        force_cell_output: true,
    })
}

fn collect_string_array(
    array: StringArray,
    force_cell_output: bool,
) -> Result<SubjectCollection, String> {
    let rows = array.rows();
    let cols = array.cols();
    let total = array.data.len();
    if total == 1 && rows == 1 && cols == 1 {
        return Ok(SubjectCollection {
            entries: vec![array.data[0].clone()],
            rows: 1,
            cols: 1,
            kind: InputKind::StringScalar,
            force_cell_output,
        });
    }
    let mut entries = Vec::with_capacity(total);
    for row in 0..rows {
        for col in 0..cols {
            let idx = row + col * rows;
            entries.push(array.data[idx].clone());
        }
    }
    Ok(SubjectCollection {
        entries,
        rows,
        cols,
        kind: InputKind::StringArray { rows, cols },
        force_cell_output: true,
    })
}

fn collect_cell_array(
    cell: runmat_builtins::CellArray,
    _force_cell_output: bool,
) -> Result<SubjectCollection, String> {
    let mut entries = Vec::with_capacity(cell.data.len());
    for ptr in &cell.data {
        let value = gather_if_needed(ptr).map_err(|e| format!("regexp: {e}"))?;
        let text = extract_string(&value).ok_or_else(|| {
            format!(
                "regexp: cell array elements must be character vectors or string scalars, got {value:?}"
            )
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
        force_cell_output: true,
    })
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

fn parse_on_off(value: Option<&Value>) -> Option<bool> {
    let text = value_to_lower_string(value?)?;
    match text.as_str() {
        "on" => Some(true),
        "off" => Some(false),
        _ => None,
    }
}

#[derive(Clone)]
struct TokenComponent {
    text: Option<String>,
    start: Option<f64>,
    end: Option<f64>,
}

#[derive(Clone)]
struct MatchComponents {
    start: f64,
    end: f64,
    matched: String,
    tokens: Vec<TokenComponent>,
    named: HashMap<String, String>,
    byte_start: usize,
    byte_end: usize,
}

#[derive(Clone)]
struct SubjectMatchData {
    matches: Vec<MatchComponents>,
    splits: Option<Vec<String>>,
}

pub struct RegexpEvaluation {
    subjects: SubjectCollection,
    options: RegexpOptions,
    named_groups: Vec<String>,
    data: Vec<SubjectMatchData>,
}

impl RegexpEvaluation {
    fn new(subject: Value, pattern: Value, options: RegexpOptions) -> Result<Self, String> {
        let subjects = collect_subjects(subject, options.force_cell_output)?;
        let pattern_str = extract_string(&pattern).ok_or_else(|| {
            format!("regexp: expected char vector or string pattern, got {pattern:?}")
        })?;
        let mut builder = RegexBuilder::new(&pattern_str);
        if options.case_insensitive {
            builder.case_insensitive(true);
        }
        if options.multi_line {
            builder.multi_line(true);
        }
        if options.dot_all {
            builder.dot_matches_new_line(true);
        }
        let regex = builder.build().map_err(|e| format!("regexp: {e}"))?;
        let mut named_groups = Vec::new();
        for name in regex.capture_names().flatten() {
            if !name.is_empty() {
                named_groups.push(name.to_string());
            }
        }
        let compute_splits = options
            .outputs
            .iter()
            .any(|spec| matches!(spec, OutputSpec::Split));
        let mut data = Vec::with_capacity(subjects.entries.len());
        for text in &subjects.entries {
            let subject_data =
                evaluate_subject(text, &regex, &named_groups, &options, compute_splits);
            data.push(subject_data);
        }
        Ok(Self {
            subjects,
            options,
            named_groups,
            data,
        })
    }

    pub fn outputs_for_single(&self) -> Result<Vec<Value>, String> {
        let specs = if self.options.outputs.is_empty() {
            vec![OutputSpec::Start]
        } else {
            self.options.outputs.clone()
        };
        self.values_for_specs(&specs)
    }

    #[allow(dead_code)] // Used by ignition's CallBuiltinMulti path
    pub fn outputs_for_multi(&self) -> Result<Vec<Value>, String> {
        let specs = if self.options.outputs.is_empty() {
            vec![OutputSpec::Start, OutputSpec::End, OutputSpec::Match]
        } else {
            self.options.outputs.clone()
        };
        self.values_for_specs(&specs)
    }

    fn values_for_specs(&self, specs: &[OutputSpec]) -> Result<Vec<Value>, String> {
        let mut results = Vec::with_capacity(specs.len());
        for spec in specs {
            let value = match spec {
                OutputSpec::Start => self.start_value()?,
                OutputSpec::End => self.end_value()?,
                OutputSpec::Match => self.match_value()?,
                OutputSpec::Tokens => self.tokens_value()?,
                OutputSpec::TokenExtents => self.token_extents_value()?,
                OutputSpec::Names => self.names_value()?,
                OutputSpec::Split => self.split_value()?,
            };
            results.push(value);
        }
        Ok(results)
    }

    fn start_value(&self) -> Result<Value, String> {
        if self.subjects.is_scalar_text() {
            let starts = &self.data[0].matches;
            let nums: Vec<f64> = starts.iter().map(|m| m.start).collect();
            vector_to_value(&nums)
        } else {
            self.collect_cells(|data| {
                let nums: Vec<f64> = data.matches.iter().map(|m| m.start).collect();
                vector_to_value(&nums)
            })
        }
    }

    fn end_value(&self) -> Result<Value, String> {
        if self.subjects.is_scalar_text() {
            let ends = &self.data[0].matches;
            let nums: Vec<f64> = ends.iter().map(|m| m.end).collect();
            vector_to_value(&nums)
        } else {
            self.collect_cells(|data| {
                let nums: Vec<f64> = data.matches.iter().map(|m| m.end).collect();
                vector_to_value(&nums)
            })
        }
    }

    fn match_value(&self) -> Result<Value, String> {
        if self.subjects.is_scalar_text() {
            let data = &self.data[0];
            if self.options.once {
                let value = data
                    .matches
                    .first()
                    .map(|m| Value::String(m.matched.clone()))
                    .unwrap_or_else(|| Value::String(String::new()));
                return Ok(value);
            }
            let mut elements = Vec::with_capacity(data.matches.len());
            for m in &data.matches {
                elements.push(Value::String(m.matched.clone()));
            }
            return make_cell(elements, 1, data.matches.len());
        }
        let once = self.options.once;
        self.collect_cells(|data| {
            if once {
                let value = data
                    .matches
                    .first()
                    .map(|m| Value::String(m.matched.clone()))
                    .unwrap_or_else(|| Value::String(String::new()));
                return Ok(value);
            }
            let mut elements = Vec::with_capacity(data.matches.len());
            for m in &data.matches {
                elements.push(Value::String(m.matched.clone()));
            }
            make_cell(elements, 1, data.matches.len())
        })
    }

    fn tokens_value(&self) -> Result<Value, String> {
        if self.subjects.is_scalar_text() {
            let data = &self.data[0];
            if self.options.once {
                if let Some(first) = data.matches.first() {
                    return tokens_to_cell(&first.tokens);
                }
                return make_cell(Vec::new(), 1, 0);
            }
            let mut per_match = Vec::with_capacity(data.matches.len());
            for m in &data.matches {
                let mut tokens = Vec::with_capacity(m.tokens.len());
                for token in &m.tokens {
                    tokens.push(Value::String(token.text.clone().unwrap_or_default()));
                }
                per_match.push(make_cell(tokens, 1, m.tokens.len())?);
            }
            return make_cell(per_match, 1, data.matches.len());
        }
        let once = self.options.once;
        self.collect_cells(|data| {
            if once {
                if let Some(first) = data.matches.first() {
                    return tokens_to_cell(&first.tokens);
                }
                return make_cell(Vec::new(), 1, 0);
            }
            let mut per_match = Vec::with_capacity(data.matches.len());
            for m in &data.matches {
                let mut tokens = Vec::with_capacity(m.tokens.len());
                for token in &m.tokens {
                    tokens.push(Value::String(token.text.clone().unwrap_or_default()));
                }
                per_match.push(make_cell(tokens, 1, m.tokens.len())?);
            }
            make_cell(per_match, 1, data.matches.len())
        })
    }

    fn token_extents_value(&self) -> Result<Value, String> {
        if self.subjects.is_scalar_text() {
            let data = &self.data[0];
            if self.options.once {
                if let Some(first) = data.matches.first() {
                    return token_extents_matrix(&first.tokens);
                }
                return empty_token_extents();
            }
            let mut per_match = Vec::with_capacity(data.matches.len());
            for m in &data.matches {
                per_match.push(token_extents_matrix(&m.tokens)?);
            }
            return make_cell(per_match, 1, data.matches.len());
        }
        let once = self.options.once;
        self.collect_cells(|data| {
            if once {
                if let Some(first) = data.matches.first() {
                    return token_extents_matrix(&first.tokens);
                }
                return empty_token_extents();
            }
            let mut per_match = Vec::with_capacity(data.matches.len());
            for m in &data.matches {
                per_match.push(token_extents_matrix(&m.tokens)?);
            }
            make_cell(per_match, 1, data.matches.len())
        })
    }

    fn names_value(&self) -> Result<Value, String> {
        let names = self.named_groups.clone();
        if self.subjects.is_scalar_text() {
            let data = &self.data[0];
            if self.options.once {
                let value = data
                    .matches
                    .first()
                    .map(|m| names_struct(&names, Some(m)))
                    .unwrap_or_else(|| names_struct(&names, None));
                return Ok(value);
            }
            let mut per_match = Vec::with_capacity(data.matches.len());
            for m in &data.matches {
                per_match.push(names_struct(&names, Some(m)));
            }
            return make_cell(per_match, 1, data.matches.len());
        }
        let once = self.options.once;
        self.collect_cells(|data| {
            if once {
                let value = data
                    .matches
                    .first()
                    .map(|m| names_struct(&names, Some(m)))
                    .unwrap_or_else(|| names_struct(&names, None));
                return Ok(value);
            }
            let mut per_match = Vec::with_capacity(data.matches.len());
            for m in &data.matches {
                per_match.push(names_struct(&names, Some(m)));
            }
            make_cell(per_match, 1, data.matches.len())
        })
    }

    fn split_value(&self) -> Result<Value, String> {
        if self.subjects.is_scalar_text() {
            let splits = self.data[0]
                .splits
                .as_ref()
                .cloned()
                .unwrap_or_else(Vec::new);
            let mut values = Vec::with_capacity(splits.len());
            for part in &splits {
                values.push(Value::String(part.clone()));
            }
            let len = splits.len();
            return make_cell(values, 1, len);
        }
        self.collect_cells(|data| {
            let splits = data.splits.as_ref().cloned().unwrap_or_else(Vec::new);
            let mut values = Vec::with_capacity(splits.len());
            for part in &splits {
                values.push(Value::String(part.clone()));
            }
            let len = splits.len();
            make_cell(values, 1, len)
        })
    }

    fn collect_cells<F>(&self, mut build: F) -> Result<Value, String>
    where
        F: FnMut(&SubjectMatchData) -> Result<Value, String>,
    {
        let rows = self.subjects.rows;
        let cols = self.subjects.cols;
        let mut values = Vec::with_capacity(self.data.len());
        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let value = if idx < self.data.len() {
                    build(&self.data[idx])?
                } else {
                    Value::String(String::new())
                };
                values.push(value);
            }
        }
        make_cell(values, rows, cols)
    }
}

fn evaluate_subject(
    text: &str,
    regex: &regex::Regex,
    named_groups: &[String],
    options: &RegexpOptions,
    compute_splits: bool,
) -> SubjectMatchData {
    let mut matches = Vec::new();
    let mut splits = if compute_splits {
        Some(Vec::new())
    } else {
        None
    };
    let mut last_split = 0usize;
    for caps in regex.captures_iter(text) {
        let mat = match caps.get(0) {
            Some(m) => m,
            None => continue,
        };
        if options.emptymatch == EmptyMatchPolicy::Remove && mat.start() == mat.end() {
            continue;
        }
        if let Some(split_segments) = splits.as_mut() {
            let segment = &text[last_split..mat.start()];
            split_segments.push(segment.to_string());
            last_split = mat.end();
        }
        let start = byte_to_char_index(text, mat.start()) as f64 + 1.0;
        let end = byte_to_char_index(text, mat.end()) as f64;
        let matched = mat.as_str().to_string();
        let mut tokens = Vec::new();
        for i in 1..caps.len() {
            let token = caps.get(i);
            let (text_opt, start_opt, end_opt) = if let Some(token_match) = token {
                let t_start = byte_to_char_index(text, token_match.start()) as f64 + 1.0;
                let t_end = byte_to_char_index(text, token_match.end()) as f64;
                (
                    Some(token_match.as_str().to_string()),
                    Some(t_start),
                    Some(t_end),
                )
            } else {
                (None, None, None)
            };
            tokens.push(TokenComponent {
                text: text_opt,
                start: start_opt,
                end: end_opt,
            });
        }
        let mut named = HashMap::new();
        for name in named_groups {
            if let Some(group) = caps.name(name) {
                named.insert(name.clone(), group.as_str().to_string());
            } else {
                named.insert(name.clone(), String::new());
            }
        }
        matches.push(MatchComponents {
            start,
            end,
            matched,
            tokens,
            named,
            byte_start: mat.start(),
            byte_end: mat.end(),
        });
        if options.emptymatch == EmptyMatchPolicy::Allow && mat.end() > mat.start() && !options.once
        {
            let zero_byte = mat.end();
            let already_present = matches
                .iter()
                .any(|m| m.byte_start == zero_byte && m.byte_end == zero_byte);
            if !already_present {
                let zero_char = byte_to_char_index(text, zero_byte);
                let mut zero_tokens = Vec::with_capacity(caps.len().saturating_sub(1));
                for _ in 1..caps.len() {
                    zero_tokens.push(TokenComponent {
                        text: None,
                        start: Some(zero_char as f64 + 1.0),
                        end: Some(zero_char as f64),
                    });
                }
                let mut zero_named = HashMap::new();
                for name in named_groups {
                    zero_named.insert(name.clone(), String::new());
                }
                matches.push(MatchComponents {
                    start: zero_char as f64 + 1.0,
                    end: zero_char as f64,
                    matched: String::new(),
                    tokens: zero_tokens,
                    named: zero_named,
                    byte_start: zero_byte,
                    byte_end: zero_byte,
                });
            }
        }
        if options.once {
            break;
        }
    }
    if let Some(split_segments) = splits.as_mut() {
        let remainder = &text[last_split..];
        split_segments.push(remainder.to_string());
    }
    SubjectMatchData { matches, splits }
}

fn byte_to_char_index(text: &str, byte_index: usize) -> usize {
    text[..byte_index].chars().count()
}

fn vector_to_value(values: &[f64]) -> Result<Value, String> {
    let cols = values.len();
    let shape = vec![1, cols];
    let tensor = Tensor::new(values.to_vec(), shape).map_err(|e| format!("regexp: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn tokens_to_cell(tokens: &[TokenComponent]) -> Result<Value, String> {
    let mut values = Vec::with_capacity(tokens.len());
    for token in tokens {
        values.push(Value::String(token.text.clone().unwrap_or_default()));
    }
    make_cell(values, 1, tokens.len())
}

fn token_extents_matrix(tokens: &[TokenComponent]) -> Result<Value, String> {
    let rows = tokens.len();
    let mut tensor_data = vec![0.0; rows * 2];
    for (row, token) in tokens.iter().enumerate() {
        tensor_data[row] = token.start.unwrap_or(f64::NAN);
        tensor_data[row + rows] = token.end.unwrap_or(f64::NAN);
    }
    let tensor = Tensor::new(tensor_data, vec![rows, 2]).map_err(|e| format!("regexp: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn empty_token_extents() -> Result<Value, String> {
    let tensor = Tensor::new(Vec::new(), vec![0, 2]).map_err(|e| format!("regexp: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn names_struct(names: &[String], match_data: Option<&MatchComponents>) -> Value {
    let mut st = StructValue::new();
    for name in names {
        let value = match match_data {
            Some(m) => m.named.get(name).cloned().unwrap_or_default(),
            None => String::new(),
        };
        st.fields.insert(name.clone(), Value::String(value));
    }
    Value::Struct(st)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "doc_export")]
    use crate::builtins::common::test_support;

    #[test]
    fn regexp_basic_positions() {
        let eval = evaluate(
            Value::String("abracadabra".into()),
            Value::String("a".into()),
            &[],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 4.0, 6.0, 8.0, 11.0]);
            }
            Value::Num(_) => panic!("expected tensor"),
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_match_output() {
        let eval = evaluate(
            Value::String("abc123xyz".into()),
            Value::String(r"\d+".into()),
            &[Value::String("match".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 1);
                assert_eq!(
                    unsafe { &*ca.data[0].as_raw() },
                    &Value::String("123".into())
                );
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_tokens_output() {
        let eval = evaluate(
            Value::String("2024-03-14".into()),
            Value::String(r"(\d{4})-(\d{2})-(\d{2})".into()),
            &[Value::String("tokens".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 1);
                let token_cell = unsafe { &*ca.data[0].as_raw() };
                match token_cell {
                    Value::Cell(inner) => {
                        assert_eq!(inner.data.len(), 3);
                        let year = unsafe { &*inner.data[0].as_raw() };
                        assert_eq!(year, &Value::String("2024".into()));
                        let month = unsafe { &*inner.data[1].as_raw() };
                        assert_eq!(month, &Value::String("03".into()));
                        let day = unsafe { &*inner.data[2].as_raw() };
                        assert_eq!(day, &Value::String("14".into()));
                    }
                    other => panic!("unexpected nested value {other:?}"),
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_match_once_returns_scalar() {
        let eval = evaluate(
            Value::String("abc123xyz".into()),
            Value::String(r"\d+".into()),
            &[Value::String("match".into()), Value::String("once".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::String(s) => assert_eq!(s, "123"),
            other => panic!("expected string output, got {other:?}"),
        }
    }

    #[test]
    fn regexp_tokens_once_flattens() {
        let eval = evaluate(
            Value::String("2024-03-14".into()),
            Value::String(r"(\d{4})-(\d{2})-(\d{2})".into()),
            &[Value::String("tokens".into()), Value::String("once".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 3);
                assert_eq!(ca.data.len(), 3);
                let year = unsafe { &*ca.data[0].as_raw() };
                assert_eq!(year, &Value::String("2024".into()));
                let month = unsafe { &*ca.data[1].as_raw() };
                assert_eq!(month, &Value::String("03".into()));
                let day = unsafe { &*ca.data[2].as_raw() };
                assert_eq!(day, &Value::String("14".into()));
            }
            other => panic!("expected cell row of tokens, got {other:?}"),
        }
    }

    #[test]
    fn regexp_token_extents_once_matrix() {
        let eval = evaluate(
            Value::String("2024-03-14".into()),
            Value::String(r"(\d{4})-(\d{2})-(\d{2})".into()),
            &[
                Value::String("tokenextents".into()),
                Value::String("once".into()),
            ],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                assert_eq!(t.data, vec![1.0, 6.0, 9.0, 4.0, 7.0, 10.0]);
            }
            other => panic!("expected token extent matrix, got {other:?}"),
        }
    }

    #[test]
    fn regexp_names_once_struct() {
        let eval = evaluate(
            Value::String("X=42; Y=7;".into()),
            Value::String("(?<name>[A-Z])=(?<value>\\d+)".into()),
            &[Value::String("names".into()), Value::String("once".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Struct(st) => {
                match st.fields.get("name") {
                    Some(Value::String(s)) => assert_eq!(s, "X"),
                    other => panic!("unexpected name field {other:?}"),
                }
                match st.fields.get("value") {
                    Some(Value::String(s)) => assert_eq!(s, "42"),
                    other => panic!("unexpected value field {other:?}"),
                }
            }
            other => panic!("expected struct output, got {other:?}"),
        }
    }

    #[test]
    fn regexp_split_string_array() {
        let array = StringArray::new(vec!["a,b,c".into(), "1,2,3".into()], vec![2, 1]).unwrap();
        let eval = evaluate(
            Value::StringArray(array),
            Value::String(",".into()),
            &[Value::String("split".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 2);
                for ptr in &ca.data {
                    match unsafe { &*ptr.as_raw() } {
                        Value::Cell(split) => assert_eq!(split.data.len(), 3),
                        other => panic!("unexpected nested value {other:?}"),
                    }
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_once_flag() {
        let eval = evaluate(
            Value::String("abababa".into()),
            Value::String("ba".into()),
            &[Value::String("once".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Num(n) => assert_eq!(*n, 2.0),
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0]),
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_multi_output_default() {
        let eval = evaluate(
            Value::String("abcabc".into()),
            Value::String("a.".into()),
            &[],
        )
        .unwrap();
        let outputs = eval.outputs_for_multi().unwrap();
        assert_eq!(outputs.len(), 3);
        match &outputs[0] {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 4.0]),
            other => panic!("unexpected output {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 5.0]),
            other => panic!("unexpected output {other:?}"),
        }
        match &outputs[2] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 2);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_emptymatch_allow() {
        let eval = evaluate(
            Value::String("aba".into()),
            Value::String("b*".into()),
            &[
                Value::String("emptymatch".into()),
                Value::String("allow".into()),
            ],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        match &outputs[0] {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]),
            Value::Num(n) => assert_eq!(*n, 1.0),
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_names_output() {
        let eval = evaluate(
            Value::String("X=42; Y=7;".into()),
            Value::String("(?<name>[A-Z])=(?<value>\\d+)".into()),
            &[Value::String("names".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 2);
                let first = unsafe { &*ca.data[0].as_raw() };
                match first {
                    Value::Struct(st) => {
                        match st.fields.get("name") {
                            Some(Value::String(s)) => assert_eq!(s, "X"),
                            other => panic!("unexpected name field {other:?}"),
                        }
                        match st.fields.get("value") {
                            Some(Value::String(s)) => assert_eq!(s, "42"),
                            other => panic!("unexpected value field {other:?}"),
                        }
                    }
                    other => panic!("unexpected struct {other:?}"),
                }
                let second = unsafe { &*ca.data[1].as_raw() };
                match second {
                    Value::Struct(st) => {
                        match st.fields.get("name") {
                            Some(Value::String(s)) => assert_eq!(s, "Y"),
                            other => panic!("unexpected name field {other:?}"),
                        }
                        match st.fields.get("value") {
                            Some(Value::String(s)) => assert_eq!(s, "7"),
                            other => panic!("unexpected value field {other:?}"),
                        }
                    }
                    other => panic!("unexpected struct {other:?}"),
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_token_extents_output() {
        let eval = evaluate(
            Value::String("2024-03-14".into()),
            Value::String(r"(\d{4})-(\d{2})-(\d{2})".into()),
            &[Value::String("tokenextents".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 1);
                let matrix = unsafe { &*ca.data[0].as_raw() };
                match matrix {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![3, 2]);
                        // Starts should be 1,6,9 (1-based); ends 4,7,10
                        assert_eq!(t.data[0], 1.0);
                        assert_eq!(t.data[1], 6.0);
                        assert_eq!(t.data[2], 9.0);
                        assert_eq!(t.data[3], 4.0);
                        assert_eq!(t.data[4], 7.0);
                        assert_eq!(t.data[5], 10.0);
                    }
                    other => panic!("unexpected token extent value {other:?}"),
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_force_cell_output_scalar() {
        let eval = evaluate(
            Value::String("abcabc".into()),
            Value::String("a.".into()),
            &[Value::String("forcecelloutput".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 1);
                assert_eq!(ca.data.len(), 1);
                let inner = unsafe { &*ca.data[0].as_raw() };
                match inner {
                    Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 4.0]),
                    other => panic!("unexpected inner value {other:?}"),
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_string_array_multi_dim_order() {
        let sa = StringArray::new(
            vec!["r1c1".into(), "r2c1".into(), "r1c2".into(), "r2c2".into()],
            vec![2, 2],
        )
        .unwrap();
        let eval = evaluate(
            Value::StringArray(sa),
            Value::String("r1".into()),
            &[Value::String("match".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 2);
                assert_eq!(ca.data.len(), 4);
                // Row 0, Col 0
                match unsafe { &*ca.data[0].as_raw() } {
                    Value::Cell(inner) => assert_eq!(inner.data.len(), 1),
                    other => panic!("unexpected inner cell {other:?}"),
                }
                // Row 0, Col 1
                match unsafe { &*ca.data[1].as_raw() } {
                    Value::Cell(inner) => assert_eq!(inner.data.len(), 1),
                    other => panic!("unexpected inner cell {other:?}"),
                }
                // Row 1, Col 0
                match unsafe { &*ca.data[2].as_raw() } {
                    Value::Cell(inner) => assert_eq!(inner.data.len(), 0),
                    other => panic!("unexpected inner cell {other:?}"),
                }
                // Row 1, Col 1
                match unsafe { &*ca.data[3].as_raw() } {
                    Value::Cell(inner) => assert_eq!(inner.data.len(), 0),
                    other => panic!("unexpected inner cell {other:?}"),
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_cell_array_multi_dim_order() {
        let cell = runmat_builtins::CellArray::new(
            vec![
                Value::String("r0c0".into()),
                Value::String("r0c1".into()),
                Value::String("r1c0".into()),
                Value::String("r1c1".into()),
            ],
            2,
            2,
        )
        .unwrap();
        let eval = evaluate(
            Value::Cell(cell),
            Value::String("r0".into()),
            &[Value::String("match".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 2);
                assert_eq!(ca.data.len(), 4);
                // Row 0, Col 0 should have one match
                match unsafe { &*ca.data[0].as_raw() } {
                    Value::Cell(inner) => assert_eq!(inner.data.len(), 1),
                    other => panic!("unexpected inner cell {other:?}"),
                }
                // Row 0, Col 1 should have one match
                match unsafe { &*ca.data[1].as_raw() } {
                    Value::Cell(inner) => assert_eq!(inner.data.len(), 1),
                    other => panic!("unexpected inner cell {other:?}"),
                }
                // Row 1 entries should be empty
                match unsafe { &*ca.data[2].as_raw() } {
                    Value::Cell(inner) => assert_eq!(inner.data.len(), 0),
                    other => panic!("unexpected inner cell {other:?}"),
                }
                match unsafe { &*ca.data[3].as_raw() } {
                    Value::Cell(inner) => assert_eq!(inner.data.len(), 0),
                    other => panic!("unexpected inner cell {other:?}"),
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_ignorecase_flag() {
        let eval = evaluate(
            Value::String("AbC".into()),
            Value::String("abc".into()),
            &[
                Value::String("match".into()),
                Value::String("ignorecase".into()),
            ],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 1);
                match unsafe { &*ca.data[0].as_raw() } {
                    Value::String(s) => assert_eq!(s, "AbC"),
                    Value::Cell(inner) => {
                        assert_eq!(inner.data.len(), 1);
                        match unsafe { &*inner.data[0].as_raw() } {
                            Value::String(s) => assert_eq!(s, "AbC"),
                            other => panic!("unexpected match value {other:?}"),
                        }
                    }
                    other => panic!("unexpected inner value {other:?}"),
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn regexp_lineanchors_and_dotall_flags() {
        let eval = evaluate(
            Value::String("first\nsecond".into()),
            Value::String("^second".into()),
            &[
                Value::String("lineanchors".into()),
                Value::String("on".into()),
            ],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Tensor(t) => assert_eq!(t.data, vec![7.0]),
            Value::Num(n) => assert_eq!(*n, 7.0),
            Value::Cell(_) => panic!("expected numeric output"),
            other => panic!("unexpected output {other:?}"),
        }

        let eval = evaluate(
            Value::String("first\nsecond".into()),
            Value::String("first.*second".into()),
            &[
                Value::String("match".into()),
                Value::String("dotall".into()),
                Value::String("on".into()),
            ],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 1);
                match unsafe { &*ca.data[0].as_raw() } {
                    Value::String(s) => assert_eq!(s, "first\nsecond"),
                    Value::Cell(inner) => {
                        // When forceCellOutput influences outer container, the inner should still be the match
                        assert_eq!(inner.data.len(), 1);
                    }
                    other => panic!("unexpected dotall output {other:?}"),
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(super::DOC_MD);
        assert!(!blocks.is_empty());
    }
}
