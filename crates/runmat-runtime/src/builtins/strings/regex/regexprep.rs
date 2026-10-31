//! MATLAB-compatible `regexprep` builtin for RunMat.

use runmat_value::NumericScalar;
fn required_regex_text(text: StringElement) -> BuiltinResult<Utf8Text> {
    text.into_text()
        .ok_or_else(|| runtime_error_for("regexprep: pattern or replacement must not be missing"))
}

fn into_required_regex_text(
    data: runmat_value::ExactArrayStorage<StringElement>,
) -> BuiltinResult<Vec<Utf8Text>> {
    into_text_elements(data)
        .into_iter()
        .map(required_regex_text)
        .collect()
}

#[cfg(test)]
use runmat_value::{IntValue, IntegerStorage, Tensor};
use std::sync::Arc;

use regex::{Captures, Regex, RegexBuilder};
#[cfg(test)]
use runmat_builtins::REGEXPREP_DESCRIPTOR;
use runmat_builtins::REGEXPREP_EXPLICIT_GPU_EXTENSION;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, StringArray, StringElement, Utf8Text, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinGpuSpec, ConstantStrategy, GpuOpKind, ReductionNaN, ResidencyPolicy,
};
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::regex::regexprep")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "regexprep",
    op_kind: GpuOpKind::Custom("regex"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runtime admission and compatibility gates inspect original operands before supported host fallback gathers.",
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
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::regex::regexprep"
)]
async fn regexprep_builtin(
    subject: Value,
    pattern: Value,
    replacement: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if [(&subject), (&pattern), (&replacement)]
        .into_iter()
        .any(crate::builtins::common::validation::value_contains_native_integer_class)
    {
        return Err(runtime_error_for(
            "regexprep: subject, pattern, and replacement must be text",
        ));
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&subject)
        || crate::builtins::common::validation::value_contains_explicit_gpu(&pattern)
        || crate::builtins::common::validation::value_contains_explicit_gpu(&replacement)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &REGEXPREP_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let subject = gather_if_needed_async(&subject)
        .await
        .map_err(with_builtin_context)?;
    let pattern = gather_if_needed_async(&pattern)
        .await
        .map_err(with_builtin_context)?;
    let replacement = gather_if_needed_async(&replacement)
        .await
        .map_err(with_builtin_context)?;
    let mut gathered_rest = Vec::with_capacity(rest.len());
    for value in rest {
        gathered_rest.push(
            gather_if_needed_async(&value)
                .await
                .map_err(with_builtin_context)?,
        );
    }
    let options = RegexprepOptions::parse(BUILTIN_NAME, &gathered_rest)?;

    let subjects = SubjectCollection::collect(BUILTIN_NAME, subject).await?;
    let patterns = PatternCollection::collect(BUILTIN_NAME, pattern, &subjects).await?;
    let replacements = ReplacementCollection::collect(BUILTIN_NAME, replacement, &subjects).await?;
    let plan = build_plan(BUILTIN_NAME, &patterns, &replacements, &options, &subjects)?;

    let mut results = Vec::with_capacity(subjects.entries.len());
    for (idx, original) in subjects.entries.iter().enumerate() {
        let Some(original) = original.as_str() else {
            results.push(StringElement::Missing);
            continue;
        };
        let mut current = original.to_owned();
        if let Some(sequence) = plan.get(idx) {
            for compiled in sequence {
                current = compiled.apply(&current, &options);
            }
        }
        results.push(StringElement::text(current));
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
    nth: Option<usize>,
    emptymatch: EmptyMatchPolicy,
    preserve_case: bool,
}

impl RegexprepOptions {
    fn parse(builtin: &'static str, rest: &[Value]) -> BuiltinResult<Self> {
        let mut case_insensitive = false;
        let mut multi_line = false;
        let mut dot_all = true;
        let mut once = false;
        let mut nth = None;
        let mut emptymatch = EmptyMatchPolicy::Remove;
        let mut preserve_case = false;
        let mut idx = 0usize;
        while idx < rest.len() {
            if let Some(occurrence) = parse_occurrence_option(&rest[idx], builtin)? {
                nth = Some(occurrence);
                idx += 1;
                continue;
            }
            let raw = value_to_lower_string(&rest[idx])?.ok_or_else(|| {
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
                    let flag = parse_toggle(rest.get(idx))?;
                    multi_line = flag.unwrap_or(true);
                    idx += usize::from(flag.is_some());
                }
                "dotall" => {
                    let flag = parse_toggle(rest.get(idx))?;
                    dot_all = flag.unwrap_or(true);
                    idx += usize::from(flag.is_some());
                }
                "dotexceptnewline" => {
                    let flag = parse_toggle(rest.get(idx))?;
                    dot_all = !flag.unwrap_or(true);
                    idx += usize::from(flag.is_some());
                }
                "once" => once = true,
                "emptymatch" => {
                    emptymatch = EmptyMatchPolicy::Allow;
                    let next_text = match rest.get(idx) {
                        Some(value) => value_to_lower_string(value)?,
                        None => None,
                    };
                    match next_text.as_deref() {
                        Some("allow") => idx += 1,
                        Some("remove") => {
                            emptymatch = EmptyMatchPolicy::Remove;
                            idx += 1;
                        }
                        _ => {}
                    }
                }
                "noemptymatch" => emptymatch = EmptyMatchPolicy::Remove,
                "stringanchors" => multi_line = false,
                "preservecase" => {
                    if idx < rest.len() {
                        if let Some(flag) = parse_toggle(rest.get(idx))? {
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
            nth,
            emptymatch,
            preserve_case,
        })
    }
}

fn parse_occurrence_option(value: &Value, builtin: &'static str) -> BuiltinResult<Option<usize>> {
    let numeric = match value {
        Value::Num(value) => Some(NumericScalar::F64(*value)),
        Value::Int(value) => Some(NumericScalar::from(value.clone())),
        Value::Tensor(tensor) => {
            if tensor.len() != 1 {
                return Err(runtime_error_for(format!(
                    "{builtin}: occurrence option must be a positive integer scalar"
                )));
            }
            tensor.numeric_value_at(0)
        }
        _ => None,
    };
    let Some(numeric) = numeric else {
        return Ok(None);
    };
    let occurrence = match numeric {
        NumericScalar::F64(value) => positive_occurrence_float(value),
        NumericScalar::F32(value) => positive_occurrence_float(f64::from(value)),
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_usize())
            .filter(|value| *value > 0),
    }
    .ok_or_else(|| {
        runtime_error_for(format!(
            "{builtin}: occurrence option must be a positive integer scalar"
        ))
    })?;
    Ok(Some(occurrence))
}

fn positive_occurrence_float(value: f64) -> Option<usize> {
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
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
    entries: Vec<StringElement>,
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
        let text = decode_char_array(array.data())?;
        return Ok(SubjectCollection {
            entries: vec![StringElement::text(text)],
            rows: 1,
            cols: 1,
            kind: InputKind::CharScalar,
        });
    }
    let mut entries = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        let line = decode_char_array(&array.data()[start..start + array.cols])?;
        entries.push(StringElement::text(line));
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
    let (data, _shape, _rows, _cols) = array.into_parts();
    let entries = into_text_elements(data);
    Ok(SubjectCollection {
        entries,
        rows,
        cols,
        kind: InputKind::StringArray { rows, cols },
    })
}

fn into_text_elements(data: runmat_value::ExactArrayStorage<StringElement>) -> Vec<StringElement> {
    match data.try_into_owned() {
        Ok(values) => values.into_iter().collect(),
        Err(shared) => shared.iter().cloned().collect(),
    }
}

async fn collect_cell_array(
    builtin: &'static str,
    cell: runmat_value::CellArray,
) -> BuiltinResult<SubjectCollection> {
    let mut entries = Vec::with_capacity(cell.data().len());
    for ptr in cell.data() {
        let value = gather_if_needed_async(ptr)
            .await
            .map_err(with_builtin_context)?;
        let text = extract_string(&value)?.ok_or_else(|| {
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
    Scalar(Utf8Text),
    Sequence(Vec<Utf8Text>),
    Elementwise(Vec<Utf8Text>),
}

impl PatternCollection {
    async fn collect(
        builtin: &'static str,
        value: Value,
        subjects: &SubjectCollection,
    ) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Ok(Self::Scalar(s.into_text().ok_or_else(|| runtime_error_for(format!("{builtin}: pattern or replacement must not be missing")))?)),
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
        let text = decode_char_array(array.data())?;
        return Ok(PatternCollection::Scalar(Utf8Text::new(text)));
    }
    let mut entries = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        let line = decode_char_array(&array.data()[start..start + array.cols])?;
        entries.push(Utf8Text::new(line));
    }
    Ok(PatternCollection::Sequence(entries))
}

fn collect_pattern_string_array(
    array: StringArray,
    subjects: &SubjectCollection,
) -> BuiltinResult<PatternCollection> {
    let rows = array.rows();
    let cols = array.cols();
    let (data, _shape, _rows, _cols) = array.into_parts();
    if rows == 1 && cols == 1 && !data.is_empty() {
        return Ok(PatternCollection::Scalar(required_regex_text(
            data[0].clone(),
        )?));
    }
    if rows == subjects.rows && cols == subjects.cols {
        return Ok(PatternCollection::Elementwise(into_required_regex_text(
            data,
        )?));
    }
    Ok(PatternCollection::Sequence(into_required_regex_text(data)?))
}

async fn collect_pattern_cell(
    builtin: &'static str,
    cell: runmat_value::CellArray,
    subjects: &SubjectCollection,
) -> BuiltinResult<PatternCollection> {
    let mut entries = Vec::with_capacity(cell.data().len());
    for ptr in cell.data() {
        let value = gather_if_needed_async(ptr)
            .await
            .map_err(with_builtin_context)?;
        let text = extract_string(&value)?.ok_or_else(|| {
            runtime_error_for(format!(
                "{builtin}: pattern cell elements must be character vectors or string scalars, got {value:?}"
            ))
        })?;
        entries.push(required_regex_text(text)?);
    }
    if cell.rows == subjects.rows && cell.cols == subjects.cols {
        return Ok(PatternCollection::Elementwise(entries));
    }
    Ok(PatternCollection::Sequence(entries))
}

#[derive(Debug, Clone)]
enum ReplacementCollection {
    Scalar(Utf8Text),
    Sequence(Vec<Utf8Text>),
    Elementwise(Vec<Utf8Text>),
}

impl ReplacementCollection {
    async fn collect(
        builtin: &'static str,
        value: Value,
        subjects: &SubjectCollection,
    ) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Ok(Self::Scalar(s.into_text().ok_or_else(|| runtime_error_for(format!("{builtin}: pattern or replacement must not be missing")))?)),
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
        let text = decode_char_array(array.data())?;
        return Ok(ReplacementCollection::Scalar(Utf8Text::new(text)));
    }
    let mut entries = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        let line = decode_char_array(&array.data()[start..start + array.cols])?;
        entries.push(Utf8Text::new(line));
    }
    Ok(ReplacementCollection::Sequence(entries))
}

fn collect_replacement_string_array(
    array: StringArray,
    subjects: &SubjectCollection,
) -> BuiltinResult<ReplacementCollection> {
    let rows = array.rows();
    let cols = array.cols();
    let (data, _shape, _rows, _cols) = array.into_parts();
    if rows == 1 && cols == 1 && !data.is_empty() {
        return Ok(ReplacementCollection::Scalar(required_regex_text(
            data[0].clone(),
        )?));
    }
    if rows == subjects.rows && cols == subjects.cols {
        return Ok(ReplacementCollection::Elementwise(
            into_required_regex_text(data)?,
        ));
    }
    Ok(ReplacementCollection::Sequence(into_required_regex_text(
        data,
    )?))
}

async fn collect_replacement_cell(
    builtin: &'static str,
    cell: runmat_value::CellArray,
    subjects: &SubjectCollection,
) -> BuiltinResult<ReplacementCollection> {
    let mut entries = Vec::with_capacity(cell.data().len());
    for ptr in cell.data() {
        let value = gather_if_needed_async(ptr)
            .await
            .map_err(with_builtin_context)?;
        let text = extract_string(&value)?.ok_or_else(|| {
            runtime_error_for(format!(
                "{builtin}: replacement cell elements must be character vectors or string scalars, got {value:?}"
            ))
        })?;
        entries.push(required_regex_text(text)?);
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
    builder.dot_matches_new_line(options.dot_all);
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
    let mut occurrence = 0usize;
    for caps in regex.captures_iter(text) {
        let mat = match caps.get(0) {
            Some(m) => m,
            None => continue,
        };
        if options.emptymatch == EmptyMatchPolicy::Remove && mat.start() == mat.end() {
            continue;
        }
        occurrence += 1;
        if options.nth.is_some_and(|target| occurrence != target) {
            continue;
        }
        output.push_str(&text[last..mat.start()]);
        let replacement = template.expand(&caps, options.preserve_case);
        output.push_str(&replacement);
        last = mat.end();
        replaced += 1;
        if options.once || options.nth.is_some() {
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
    results: Vec<StringElement>,
) -> BuiltinResult<Value> {
    match subjects.kind {
        InputKind::CharScalar => {
            let array = strings_to_char_array(builtin, 1, &results)?;
            Ok(Value::CharArray(array))
        }
        InputKind::StringScalar => Ok(Value::string(
            results.into_iter().next().unwrap_or_default(),
        )),
        InputKind::CharMatrix { rows } => {
            let array = strings_to_char_array(builtin, rows, &results)?;
            Ok(Value::CharArray(array))
        }
        InputKind::StringArray { rows, cols } => {
            let array = StringArray::from_elements_unbounded(results, vec![rows, cols])
                .map_err(|e| runtime_error_for(format!("{builtin}: {e}")))?;
            Ok(Value::StringArray(array))
        }
        InputKind::CellArray { rows, cols } => {
            let values = results.into_iter().map(Value::string).collect();
            make_cell(values, rows, cols)
                .map_err(|err| runtime_error_for(format!("{builtin}: {err}")))
        }
    }
}

fn strings_to_char_array(
    builtin: &'static str,
    rows: usize,
    strings: &[StringElement],
) -> BuiltinResult<CharArray> {
    if rows == 0 {
        return CharArray::from_code_units(Vec::new(), 0, 0)
            .map_err(|e| runtime_error_for(format!("{builtin}: {e}")));
    }
    let max_cols = strings
        .iter()
        .take(rows)
        .map(|s| {
            s.as_str()
                .expect("character output has textual input")
                .encode_utf16()
                .count()
        })
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(rows * max_cols);
    for idx in 0..rows {
        let text = strings.get(idx).cloned().unwrap_or_default();
        let mut units = text
            .as_str()
            .ok_or_else(|| {
                runtime_error_for(format!(
                    "{builtin}: missing cannot be converted to character output"
                ))
            })?
            .encode_utf16()
            .map(runmat_value::Utf16CodeUnit::from_raw)
            .collect::<Vec<_>>();
        while units.len() < max_cols {
            units.push(runmat_value::Utf16CodeUnit::from_raw(b' ' as u16));
        }
        data.extend(units);
    }
    CharArray::from_code_units(data, rows, max_cols)
        .map_err(|e| runtime_error_for(format!("{builtin}: {e}")))
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

fn decode_char_array(units: &[runmat_value::Utf16CodeUnit]) -> BuiltinResult<String> {
    runmat_value::decode_utf16_code_units(units)
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))
}

fn extract_string(value: &Value) -> BuiltinResult<Option<StringElement>> {
    Ok(match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data().len() == 1 => Some(sa.data()[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => {
            Some(StringElement::text(decode_char_array(ca.data())?))
        }
        _ => None,
    })
}

fn value_to_lower_string(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(s) => s.as_str().map(str::to_ascii_lowercase),
        Value::StringArray(sa) if sa.data().len() == 1 => {
            sa.data()[0].as_str().map(str::to_ascii_lowercase)
        }
        Value::CharArray(ca) if ca.rows == 1 => {
            Some(decode_char_array(ca.data())?.to_ascii_lowercase())
        }
        _ => None,
    })
}

fn parse_toggle(value: Option<&Value>) -> BuiltinResult<Option<bool>> {
    let Some(value) = value else {
        return Ok(None);
    };
    Ok(match value {
        Value::Bool(b) => Some(*b),
        Value::Int(i) => Some(!i.is_zero()),
        Value::Num(n) => Some(*n != 0.0),
        _ => {
            let Some(text) = value_to_lower_string(value)? else {
                return Ok(None);
            };
            match text.as_str() {
                "on" | "true" | "yes" | "1" => Some(true),
                "off" | "false" | "no" | "0" => Some(false),
                _ => None,
            }
        }
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = REGEXPREP_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"out = regexprep(subject, pattern, replacement)"));
        assert!(labels.contains(&"out = regexprep(subject, pattern, replacement, options...)"));
    }

    fn run_regexprep(
        subject: Value,
        pattern: Value,
        replacement: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        futures::executor::block_on(regexprep_builtin(
            subject,
            pattern,
            replacement,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
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
    fn regexprep_rejects_malformed_utf16_character_input() {
        let subject =
            CharArray::from_code_units(vec![runmat_value::Utf16CodeUnit::from_raw(0xD800)], 1, 1)
                .unwrap();
        let error = run_regexprep(
            Value::CharArray(subject),
            Value::String("x".into()),
            Value::String("y".into()),
            Vec::new(),
        )
        .expect_err("malformed UTF-16 must not be normalized or replaced");
        assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN_NAME));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_sequence_patterns() {
        let subject =
            Value::StringArray(StringArray::new(vec!["Color: red".into()], vec![1, 1]).unwrap());
        let patterns = Value::Cell(
            runmat_value::CellArray::new(
                vec![Value::String("Color".into()), Value::String("red".into())],
                1,
                2,
            )
            .unwrap(),
        );
        let replacements = Value::Cell(
            runmat_value::CellArray::new(
                vec![Value::String("Shade".into()), Value::String("blue".into())],
                1,
                2,
            )
            .unwrap(),
        );
        let result = run_regexprep(subject, patterns, replacements, Vec::new()).expect("regexprep");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data().len(), 1);
                assert_eq!(sa.data()[0], "Shade: blue");
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
                assert_eq!(sa.data(), vec![String::from("_lph_"), String::from("bet_")]);
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
            Value::StringArray(sa) => assert_eq!(sa.data()[0], "aXYbaba"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_single_occurrence_is_not_a_toggle_value() {
        let single = |value| Value::Tensor(Tensor::from_f32(vec![value], vec![1, 1]).unwrap());
        for flag in ["lineanchors", "dotall"] {
            assert_eq!(
                run_regexprep(
                    Value::String("a a".into()),
                    Value::String("a".into()),
                    Value::String("X".into()),
                    vec![Value::String(flag.into()), single(2.0)],
                )
                .expect("single N after a flag"),
                Value::String("a X".into()),
            );
        }
        let error = run_regexprep(
            Value::String("a a".into()),
            Value::String("a".into()),
            Value::String("X".into()),
            vec![Value::String("preservecase".into()), single(1.0)],
        )
        .expect_err("single is not a preservecase toggle");
        assert!(error.message().contains("expected logical or 'on'/'off'"));
        let error = run_regexprep(
            Value::String("a a".into()),
            Value::String("a".into()),
            Value::String("X".into()),
            vec![Value::String("dotall".into()), single(0.0)],
        )
        .expect_err("zero single N is not a toggle");
        assert!(error.message().contains("positive integer scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_integer_occurrence_option_is_exact_and_class_complete() {
        for occurrence in [
            IntValue::I8(2),
            IntValue::I16(2),
            IntValue::I32(2),
            IntValue::I64(2),
            IntValue::U8(2),
            IntValue::U16(2),
            IntValue::U32(2),
            IntValue::U64(2),
        ] {
            let result = run_regexprep(
                Value::String("a a a".into()),
                Value::String("a".into()),
                Value::String("X".into()),
                vec![Value::Int(occurrence)],
            )
            .expect("Nth occurrence");
            assert_eq!(result, Value::String("a X a".into()));
        }

        let occurrence =
            Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![3]), vec![1, 1]).unwrap());
        assert_eq!(
            run_regexprep(
                Value::String("a a a".into()),
                Value::String("a".into()),
                Value::String("X".into()),
                vec![occurrence],
            )
            .expect("typed occurrence tensor"),
            Value::String("a a X".into())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_rejects_invalid_integer_occurrence_options() {
        for occurrence in [
            Value::Int(IntValue::I8(0)),
            Value::Int(IntValue::I64(-1)),
            Value::Num(1.5),
        ] {
            let error = run_regexprep(
                Value::String("a a".into()),
                Value::String("a".into()),
                Value::String("X".into()),
                vec![occurrence],
            )
            .expect_err("invalid occurrence");
            assert!(error.message().contains("positive integer scalar"));
        }
        assert_eq!(
            run_regexprep(
                Value::String("a a".into()),
                Value::String("a".into()),
                Value::String("X".into()),
                vec![Value::Int(IntValue::U64(u64::from(u32::MAX)))],
            )
            .expect("valid occurrence beyond available matches"),
            Value::String("a a".into())
        );
        let widest_occurrence = run_regexprep(
            Value::String("a a".into()),
            Value::String("a".into()),
            Value::String("X".into()),
            vec![Value::Int(IntValue::U64(u64::MAX))],
        );
        if usize::BITS == 64 {
            assert_eq!(
                widest_occurrence.expect("valid native index beyond available matches"),
                Value::String("a a".into())
            );
        } else {
            let error = widest_occurrence.expect_err("occurrence exceeds the platform index range");
            assert!(error.message().contains("positive integer scalar"));
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
        let result = run_regexprep(subject, patterns, replacements, Vec::new()).expect("regexprep");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data(), vec!["cAt".to_string(), "dOg".to_string()]);
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

        let allowed_flag = run_regexprep(
            Value::String("abc".into()),
            Value::String("^".into()),
            Value::String("X".into()),
            vec![Value::String("emptymatch".into())],
        )
        .expect("standalone emptymatch");
        assert_eq!(allowed_flag, Value::String("Xabc".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_dot_matches_newline_by_default_and_option_overrides() {
        let subject = Value::String("a\nb".into());
        let pattern = Value::String("a.b".into());
        let replacement = Value::String("HIT".into());
        let default = run_regexprep(
            subject.clone(),
            pattern.clone(),
            replacement.clone(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(default, Value::String("HIT".into()));

        let excluding_newline = run_regexprep(
            subject.clone(),
            pattern.clone(),
            replacement.clone(),
            vec![Value::String("dotexceptnewline".into())],
        )
        .unwrap();
        assert_eq!(excluding_newline, subject);

        let explicit_off = run_regexprep(
            Value::String("a\nb".into()),
            pattern,
            replacement,
            vec![Value::String("dotall".into()), Value::Bool(false)],
        )
        .unwrap();
        assert_eq!(explicit_off, Value::String("a\nb".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexprep_char_array_result() {
        let chars =
            CharArray::from_unicode_scalars(vec!['c', 'a', 't', 'd', 'o', 'g'], 2, 3).unwrap();
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
                assert_eq!(
                    runmat_value::decode_utf16_code_units(out.data()).unwrap(),
                    "cXtdXg"
                );
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
        let cell = runmat_value::CellArray::new(
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
        let err =
            run_regexprep(subject, patterns, replacements, Vec::new()).expect_err("expected error");
        let message = err.message().to_string();
        assert!(
            message.contains("replacement sequence is incompatible with element-wise patterns"),
            "unexpected error message: {message}"
        );
    }
}
