//! MATLAB-compatible `whos` builtin for RunMat.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use glob::Pattern;
use regex::Regex;
use runmat_accelerate_api::{handle_is_logical, ProviderPrecision};
use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::shape::value_dimensions;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::class::class_name_for_value;
use crate::builtins::introspection::type_resolvers::whos_type;
use crate::builtins::io::mat::load::read_mat_file_for_builtin;
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::whos")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "whos",
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
    notes: "Host-only builtin. Arguments are gathered from the GPU if necessary; gpuArray metadata is derived from provider precision without launching kernels.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::whos")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "whos",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Introspection builtin; not eligible for fusion. Registration is for diagnostics only.",
};

#[runtime_builtin(
    name = "whos",
    category = "introspection",
    summary = "List variables in the workspace or MAT-files with MATLAB-compatible metadata.",
    keywords = "whos,workspace variables,memory usage,struct array",
    accel = "cpu",
    type_resolver(whos_type),
    builtin_path = "crate::builtins::introspection::whos"
)]
async fn whos_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let mut gathered = Vec::with_capacity(args.len());
    for arg in args {
        gathered.push(gather_if_needed_async(&arg).await.map_err(whos_flow)?);
    }
    let request = parse_request(&gathered).await?;

    let mut entries = match &request.source {
        WhosSource::Workspace => crate::workspace::snapshot().unwrap_or_default(),
        WhosSource::File(path) => read_mat_file_for_builtin(path, "whos")?,
    };

    if matches!(request.source, WhosSource::File(_)) {
        entries.sort_by(|a, b| a.0.cmp(&b.0));
    }

    let global_names: HashSet<String> = if matches!(request.source, WhosSource::Workspace) {
        crate::workspace::global_names().into_iter().collect()
    } else {
        HashSet::new()
    };

    let mut records = Vec::new();
    for (name, value) in entries {
        if !matches_filters(&name, &request.selectors, &request.regex_patterns) {
            continue;
        }
        let is_global = global_names.contains(&name);
        if request.only_global && !is_global {
            continue;
        }
        let record = WhosRecord::from_value(name, &value, is_global).await?;
        records.push(record);
    }

    records.sort_by(|a, b| a.name.cmp(&b.name));

    let mut values = Vec::with_capacity(records.len());
    for record in records {
        values.push(record.into_value()?);
    }
    let rows = values.len();
    make_cell(values, rows, 1).map_err(|err| build_runtime_error(err).with_builtin("whos").build())
}

#[derive(Debug)]
struct WhosRecord {
    name: String,
    size: runmat_builtins::Tensor,
    bytes: usize,
    class_name: String,
    is_global: bool,
    is_sparse: bool,
    is_complex: bool,
    nesting: String,
    persistent: bool,
}

impl WhosRecord {
    async fn from_value(name: String, value: &Value, is_global: bool) -> BuiltinResult<Self> {
        let dims = value_dimensions(value).await?;
        let size_tensor = dims_to_tensor(&dims)?;
        let mut seen = HashSet::new();
        let bytes = value_memory_bytes(value, &mut seen);
        let class_name = class_name_for_value(value);
        let is_complex = matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_));
        Ok(Self {
            name,
            size: size_tensor,
            bytes,
            class_name,
            is_global,
            is_sparse: false,
            is_complex,
            nesting: String::new(),
            persistent: false,
        })
    }

    fn into_value(self) -> BuiltinResult<Value> {
        let mut st = StructValue::new();
        st.fields
            .insert("name".to_string(), Value::String(self.name));
        st.fields
            .insert("size".to_string(), Value::Tensor(self.size));
        st.fields
            .insert("bytes".to_string(), Value::Num(self.bytes as f64));
        st.fields
            .insert("class".to_string(), Value::String(self.class_name));
        st.fields
            .insert("global".to_string(), Value::Bool(self.is_global));
        st.fields
            .insert("sparse".to_string(), Value::Bool(self.is_sparse));
        st.fields
            .insert("complex".to_string(), Value::Bool(self.is_complex));
        st.fields
            .insert("nesting".to_string(), Value::String(self.nesting));
        st.fields
            .insert("persistent".to_string(), Value::Bool(self.persistent));
        Ok(Value::Struct(st))
    }
}

async fn parse_request(values: &[Value]) -> BuiltinResult<WhosRequest> {
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
                        return Err(whos_error("whos: '-file' requires a filename"));
                    }
                    if path_value.is_some() {
                        return Err(whos_error("whos: '-file' may only be specified once"));
                    }
                    path_value = Some(values[idx].clone());
                    idx += 1;
                    continue;
                }
                "-regexp" => {
                    idx += 1;
                    if idx >= values.len() {
                        return Err(whos_error("whos: '-regexp' requires at least one pattern"));
                    }
                    while idx < values.len() {
                        if option_token(&values[idx])?.is_some() {
                            break;
                        }
                        let candidates = extract_name_list(&values[idx]).await?;
                        if candidates.is_empty() {
                            return Err(whos_error(
                                "whos: '-regexp' requires non-empty pattern strings",
                            ));
                        }
                        for pattern in candidates {
                            let regex = Regex::new(&pattern).map_err(|err| {
                                build_runtime_error(format!(
                                    "whos: invalid regular expression '{pattern}': {err}"
                                ))
                                .with_builtin("whos")
                                .with_source(err)
                                .build()
                            })?;
                            regex_patterns.push(regex);
                        }
                        idx += 1;
                    }
                    continue;
                }
                other => {
                    return Err(whos_error(format!("whos: unsupported option '{other}'")));
                }
            }
        }

        let extracted = extract_name_list(&values[idx]).await?;
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
        WhosSource::File(path)
    } else {
        WhosSource::Workspace
    };

    let selectors = build_selectors(&names)?;

    Ok(WhosRequest {
        source,
        selectors,
        regex_patterns,
        only_global,
    })
}

#[derive(Debug)]
struct WhosRequest {
    source: WhosSource,
    selectors: Vec<NameSelector>,
    regex_patterns: Vec<Regex>,
    only_global: bool,
}

#[derive(Debug)]
enum WhosSource {
    Workspace,
    File(PathBuf),
}

#[derive(Debug)]
enum NameSelector {
    Exact(String),
    Wildcard(Pattern),
}

fn whos_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("whos").build()
}

fn whos_flow(mut err: RuntimeError) -> RuntimeError {
    err.context = err.context.with_builtin("whos");
    err
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

fn build_selectors(names: &[String]) -> BuiltinResult<Vec<NameSelector>> {
    let mut selectors = Vec::with_capacity(names.len());
    for name in names {
        if contains_wildcards(name) {
            let pattern = Pattern::new(name).map_err(|err| {
                build_runtime_error(format!("whos: invalid pattern '{name}': {err}"))
                    .with_builtin("whos")
                    .with_source(err)
                    .build()
            })?;
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

fn parse_file_path(value: &Value) -> BuiltinResult<PathBuf> {
    let text = value_to_string_scalar(value)
        .ok_or_else(|| whos_error("whos: filename must be a character vector or string scalar"))?;
    let mut path = PathBuf::from(text);
    if path.extension().is_none() {
        path.set_extension("mat");
    }
    Ok(path)
}

fn option_token(value: &Value) -> BuiltinResult<Option<String>> {
    if let Some(token) = value_to_string_scalar(value) {
        if token.starts_with('-') {
            return Ok(Some(token.to_ascii_lowercase()));
        }
    }
    Ok(None)
}

#[async_recursion::async_recursion(?Send)]
async fn extract_name_list(value: &Value) -> BuiltinResult<Vec<String>> {
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
                let gathered = gather_if_needed_async(inner).await.map_err(whos_flow)?;
                if let Some(text) = value_to_string_scalar(&gathered) {
                    names.push(text);
                } else {
                    return Err(whos_error(
                        "whos: selection cells must contain string or character scalars",
                    ));
                }
            }
            Ok(names)
        }
        Value::GpuTensor(_) => {
            let gathered = gather_if_needed_async(value).await.map_err(whos_flow)?;
            extract_name_list(&gathered).await
        }
        _ => Err(whos_error(
            "whos: selections must be character vectors, string scalars, string arrays, or cell arrays of those types",
        )),
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

fn char_array_rows_as_strings(ca: &runmat_builtins::CharArray) -> Vec<String> {
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

fn dims_to_tensor(dims: &[usize]) -> BuiltinResult<runmat_builtins::Tensor> {
    let mut normalized = dims.to_vec();
    if normalized.is_empty() {
        normalized = vec![1, 1];
    }
    let data: Vec<f64> = normalized.iter().map(|dim| *dim as f64).collect();
    runmat_builtins::Tensor::new(data, vec![1, normalized.len()]).map_err(|err| {
        build_runtime_error(format!("whos: failed to materialize size vector: {err}"))
            .with_builtin("whos")
            .build()
    })
}

fn value_memory_bytes(value: &Value, seen: &mut HashSet<usize>) -> usize {
    match value {
        Value::Num(_) => 8,
        Value::Int(i) => match i {
            runmat_builtins::IntValue::I8(_) | runmat_builtins::IntValue::U8(_) => 1,
            runmat_builtins::IntValue::I16(_) | runmat_builtins::IntValue::U16(_) => 2,
            runmat_builtins::IntValue::I32(_) | runmat_builtins::IntValue::U32(_) => 4,
            runmat_builtins::IntValue::I64(_) | runmat_builtins::IntValue::U64(_) => 8,
        },
        Value::Bool(_) => 1,
        Value::LogicalArray(la) => la.data.len(),
        Value::CharArray(ca) => ca.data.len().saturating_mul(2),
        Value::String(s) => s.encode_utf16().count().saturating_mul(2),
        Value::StringArray(sa) => sa
            .data
            .iter()
            .map(|s| s.encode_utf16().count().saturating_mul(2))
            .sum(),
        Value::Tensor(t) => t.data.len().saturating_mul(8),
        Value::Complex(_, _) => 16,
        Value::ComplexTensor(t) => t.data.len().saturating_mul(16),
        Value::Cell(ca) => {
            let mut total = 0usize;
            for handle in &ca.data {
                let ptr = unsafe { handle.as_raw() } as usize;
                if seen.insert(ptr) {
                    let value = unsafe { &*handle.as_raw() };
                    total = total.saturating_add(value_memory_bytes(value, seen));
                }
            }
            total
        }
        Value::Struct(st) => st.fields.values().fold(0usize, |acc, v| {
            acc.saturating_add(value_memory_bytes(v, seen))
        }),
        Value::GpuTensor(handle) => {
            #[cfg(all(test, feature = "wgpu"))]
            {
                if handle.device_id != 0 {
                    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                    );
                }
            }
            let elem_size = if handle_is_logical(handle) {
                1
            } else {
                gpu_element_size_bytes()
            };
            let count = handle
                .shape
                .iter()
                .fold(1usize, |acc, dim| acc.saturating_mul(*dim));
            count.saturating_mul(elem_size)
        }
        Value::Object(obj) => obj.properties.values().fold(0usize, |acc, v| {
            acc.saturating_add(value_memory_bytes(v, seen))
        }),
        Value::HandleObject(handle) => {
            let ptr = unsafe { handle.target.as_raw() } as usize;
            if seen.insert(ptr) {
                let inner = unsafe { &*handle.target.as_raw() };
                value_memory_bytes(inner, seen)
            } else {
                0
            }
        }
        Value::Listener(listener) => {
            let mut total = 0usize;
            let target_ptr = unsafe { listener.target.as_raw() } as usize;
            if seen.insert(target_ptr) {
                let value = unsafe { &*listener.target.as_raw() };
                total = total.saturating_add(value_memory_bytes(value, seen));
            }
            let callback_ptr = unsafe { listener.callback.as_raw() } as usize;
            if seen.insert(callback_ptr) {
                let value = unsafe { &*listener.callback.as_raw() };
                total = total.saturating_add(value_memory_bytes(value, seen));
            }
            total
        }
        Value::Closure(closure) => closure.captures.iter().fold(0usize, |acc, v| {
            acc.saturating_add(value_memory_bytes(v, seen))
        }),
        Value::FunctionHandle(_) => 0,
        Value::ClassRef(name) => name.len().saturating_mul(2),
        Value::MException(exc) => {
            let base = exc
                .message
                .len()
                .saturating_mul(2)
                .saturating_add(exc.identifier.len().saturating_mul(2));
            exc.stack.iter().fold(base, |acc, frame| {
                acc.saturating_add(frame.len().saturating_mul(2))
            })
        }
        Value::OutputList(values) => values
            .iter()
            .fold(0usize, |acc, v| acc.saturating_add(value_memory_bytes(v, seen))),
    }
}

fn gpu_element_size_bytes() -> usize {
    runmat_accelerate_api::provider()
        .map(|provider| match provider.precision() {
            ProviderPrecision::F32 => 4,
            ProviderPrecision::F64 => 8,
        })
        .unwrap_or(8)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::call_builtin_async;
    use futures::executor::block_on;
    use runmat_builtins::{CellArray, CharArray, StructValue as TestStruct, Tensor};
    use runmat_thread_local::runmat_thread_local;
    use std::cell::RefCell;
    use std::collections::{HashMap, HashSet};
    use tempfile::tempdir;

    fn whos_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::whos_builtin(args))
    }

    runmat_thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
        static TEST_GLOBALS: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
    }

    pub(crate) fn ensure_test_resolver() {
        crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
            lookup: |name| TEST_WORKSPACE.with(|slot| slot.borrow().get(name).cloned()),
            snapshot: || {
                let mut entries: Vec<(String, Value)> =
                    TEST_WORKSPACE.with(|slot| slot.borrow().clone().into_iter().collect());
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                entries
            },
            globals: || TEST_GLOBALS.with(|slot| slot.borrow().iter().cloned().collect()),
            assign: None,
        });
    }

    pub(crate) fn set_workspace(entries: &[(&str, Value)], globals: &[&str]) {
        TEST_WORKSPACE.with(|slot| {
            let mut map = slot.borrow_mut();
            map.clear();
            for (name, value) in entries {
                map.insert((*name).to_string(), value.clone());
            }
        });
        TEST_GLOBALS.with(|slot| {
            let mut set = slot.borrow_mut();
            set.clear();
            for name in globals {
                set.insert((*name).to_string());
            }
        });
    }

    fn workspace_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::workspace::test_guard()
    }

    fn structs_from_value(value: Value) -> Vec<TestStruct> {
        match value {
            Value::Cell(cell) => cell
                .data
                .iter()
                .map(|ptr| unsafe { &*ptr.as_raw() }.clone())
                .map(|value| match value {
                    Value::Struct(st) => st,
                    other => panic!("expected struct entry, got {other:?}"),
                })
                .collect(),
            Value::Struct(st) => vec![st],
            other => panic!("expected struct array, got {other:?}"),
        }
    }

    fn field_string(struct_value: &TestStruct, field: &str) -> Option<String> {
        struct_value
            .fields
            .get(field)
            .and_then(|value| match value {
                Value::String(s) => Some(s.clone()),
                Value::CharArray(ca) if ca.rows == 1 => {
                    Some(ca.data.iter().collect::<String>().trim().to_string())
                }
                _ => None,
            })
    }

    fn field_bool(struct_value: &TestStruct, field: &str) -> Option<bool> {
        struct_value
            .fields
            .get(field)
            .and_then(|value| match value {
                Value::Bool(b) => Some(*b),
                _ => None,
            })
    }

    fn field_bytes(struct_value: &TestStruct) -> Option<f64> {
        struct_value
            .fields
            .get("bytes")
            .and_then(|value| match value {
                Value::Num(n) => Some(*n),
                _ => None,
            })
    }

    fn field_size(struct_value: &TestStruct) -> Option<Vec<f64>> {
        struct_value
            .fields
            .get("size")
            .and_then(|value| match value {
                Value::Tensor(t) => Some(t.data.clone()),
                _ => None,
            })
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    pub(crate) fn char_array_from_rows(rows: &[&str]) -> CharArray {
        let cols = rows.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut data = Vec::with_capacity(rows.len() * cols.max(1));
        for row in rows {
            let mut chars: Vec<char> = row.chars().collect();
            while chars.len() < cols {
                chars.push(' ');
            }
            if cols == 0 {
                continue;
            }
            data.extend(chars);
        }
        if cols == 0 {
            CharArray::new(Vec::new(), rows.len(), 0).expect("char array")
        } else {
            CharArray::new(data, rows.len(), cols).expect("char array")
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_lists_workspace_variables() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        set_workspace(
            &[("a", Value::Num(42.0)), ("b", Value::Tensor(tensor))],
            &[],
        );

        let value = whos_builtin(Vec::new()).expect("whos");
        let entries = structs_from_value(value);
        assert_eq!(entries.len(), 2);

        let first = entries
            .iter()
            .find(|st| field_string(st, "name").as_deref() == Some("a"))
            .expect("entry 'a'");
        assert_eq!(field_size(first).unwrap(), vec![1.0, 1.0]);
        assert_eq!(field_bytes(first).unwrap(), 8.0);
        assert_eq!(field_bool(first, "global"), Some(false));

        let second = entries
            .iter()
            .find(|st| field_string(st, "name").as_deref() == Some("b"))
            .expect("entry 'b'");
        assert_eq!(field_size(second).unwrap(), vec![2.0, 3.0]);
        assert_eq!(field_bytes(second).unwrap(), 48.0);
        assert_eq!(field_bool(second, "complex"), Some(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_filters_with_wildcard() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(
            &[("alpha", Value::Num(1.0)), ("beta", Value::Num(2.0))],
            &[],
        );

        let value = whos_builtin(vec![Value::from("a*")]).expect("whos");
        let entries = structs_from_value(value);
        assert_eq!(entries.len(), 1);
        assert_eq!(field_string(&entries[0], "name").unwrap(), "alpha");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_filters_with_regex() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(
            &[
                ("foo", Value::Num(1.0)),
                ("bar", Value::Num(2.0)),
                ("baz", Value::Num(3.0)),
            ],
            &[],
        );

        let value = whos_builtin(vec![Value::from("-regexp"), Value::from("^ba")]).expect("whos");
        let entries = structs_from_value(value);
        let names: Vec<String> = entries
            .iter()
            .map(|st| field_string(st, "name").unwrap())
            .collect();
        assert_eq!(names, vec!["bar", "baz"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_filters_global_only() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(
            &[("shared", Value::Num(1.0)), ("local", Value::Num(2.0))],
            &["shared"],
        );

        let value = whos_builtin(vec![Value::from("global")]).expect("whos");
        let entries = structs_from_value(value);
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(field_string(entry, "name").unwrap(), "shared");
        assert_eq!(field_bool(entry, "global"), Some(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_accepts_char_array_arguments() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("gamma", Value::Num(3.0)),
                ("theta", Value::Num(4.0)),
            ],
            &[],
        );

        let arg = Value::CharArray(char_array_from_rows(&["alpha", "gamma"]));
        let value = whos_builtin(vec![arg]).expect("whos");
        let entries = structs_from_value(value);
        let names: Vec<String> = entries
            .iter()
            .map(|st| field_string(st, "name").unwrap())
            .collect();
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_accepts_cell_array_arguments() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("gamma", Value::Num(3.0)),
                ("omega", Value::Num(4.0)),
            ],
            &[],
        );

        let cell = CellArray::new(vec![Value::from("gamma"), Value::from("alpha")], 2, 1).unwrap();
        let value = whos_builtin(vec![Value::Cell(cell)]).expect("whos");
        let entries = structs_from_value(value);
        let names: Vec<String> = entries
            .iter()
            .map(|st| field_string(st, "name").unwrap())
            .collect();
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_rejects_numeric_selection() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[], &[]);
        let err = whos_builtin(vec![Value::Num(7.0)]).expect_err("whos should error");
        let message = error_message(err);
        assert!(
            message.contains("whos: selections must"),
            "unexpected error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_rejects_unknown_option() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[], &[]);
        let err = whos_builtin(vec![Value::from("-bogus")]).expect_err("whos should error");
        let message = error_message(err);
        assert!(
            message.contains("unsupported option"),
            "unexpected error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_requires_filename_for_file_option() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[], &[]);
        let err = whos_builtin(vec![Value::from("-file")]).expect_err("whos should error");
        let message = error_message(err);
        assert!(
            message.contains("'-file' requires a filename"),
            "error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_requires_pattern_for_regexp() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[], &[]);
        let err = whos_builtin(vec![Value::from("-regexp")]).expect_err("whos should error");
        let message = error_message(err);
        assert!(
            message.contains("'-regexp' requires at least one pattern"),
            "error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_rejects_invalid_regex() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[], &[]);
        let err = whos_builtin(vec![Value::from("-regexp"), Value::from("[")])
            .expect_err("whos should error");
        let message = error_message(err);
        assert!(
            message.contains("invalid regular expression"),
            "error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_file_option_reads_mat_file() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("beta", Value::Tensor(tensor.clone())),
            ],
            &[],
        );

        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("snapshot.mat");
        let path_str = file_path.to_string_lossy().to_string();
        block_on(call_builtin_async(
            "save",
            &[
                Value::from(path_str.clone()),
                Value::from("alpha"),
                Value::from("beta"),
            ],
        ))
        .expect("save");

        set_workspace(&[], &[]);
        let value = whos_builtin(vec![Value::from("-file"), Value::from(path_str)]).expect("whos");
        let entries = structs_from_value(value);
        let names: Vec<String> = entries
            .iter()
            .map(|st| field_string(st, "name").unwrap())
            .collect();
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_reports_gpu_bytes() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..16).map(|v| v as f64).collect(), vec![4, 4]).unwrap();
            let gpu_value = block_on(crate::call_builtin_async(
                "gpuArray",
                &[Value::Tensor(tensor.clone())],
            ))
            .expect("gpuArray");
            set_workspace(&[("G", gpu_value.clone())], &[]);

            let value = whos_builtin(vec![Value::from("G")]).expect("whos");
            let entries = structs_from_value(value);
            assert_eq!(entries.len(), 1);
            let bytes = field_bytes(&entries[0]).unwrap();
            let elem = match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F32 => 4.0,
                runmat_accelerate_api::ProviderPrecision::F64 => 8.0,
            };
            assert_eq!(bytes, elem * tensor.data.len() as f64);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn whos_reports_gpu_logical_bytes() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![0.0, 1.0, 0.0, 1.0], vec![4, 1]).unwrap();
            let gpu_value = block_on(crate::call_builtin_async(
                "gpuArray",
                &[Value::Tensor(tensor.clone())],
            ))
            .expect("gpuArray");
            let handle = match gpu_value {
                Value::GpuTensor(ref h) => {
                    runmat_accelerate_api::set_handle_logical(h, true);
                    h.clone()
                }
                _ => panic!("expected gpu tensor"),
            };
            set_workspace(&[("logical_gpu", Value::GpuTensor(handle))], &[]);

            let value = whos_builtin(vec![Value::from("logical_gpu")]).expect("whos");
            let entries = structs_from_value(value);
            assert_eq!(entries.len(), 1);
            let bytes = field_bytes(&entries[0]).unwrap();
            assert_eq!(bytes, tensor.data.len() as f64);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn whos_reports_gpu_bytes_with_wgpu_provider() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        use runmat_accelerate_api::AccelProvider;
        let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("wgpu provider");
        let tensor = Tensor::new(vec![0.0; 12], vec![3, 4]).unwrap();
        let gpu_value = block_on(crate::call_builtin_async(
            "gpuArray",
            &[Value::Tensor(tensor.clone())],
        ))
        .expect("gpuArray");
        set_workspace(&[("WG", gpu_value)], &[]);

        let value = whos_builtin(vec![Value::from("WG")]).expect("whos");
        let entries = structs_from_value(value);
        assert_eq!(entries.len(), 1);
        let bytes = field_bytes(&entries[0]).unwrap();
        let elem = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F32 => 4.0,
            runmat_accelerate_api::ProviderPrecision::F64 => 8.0,
        };
        let expected = elem * tensor.data.len() as f64;
        assert!((bytes - expected).abs() < 1e-6);
    }
}
