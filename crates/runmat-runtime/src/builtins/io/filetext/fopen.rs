//! MATLAB-compatible `fopen` builtin exposing host file streams.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use runmat_builtins::{CharArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::registry::{self, FileInfo, RegisteredFile};
use crate::{gather_if_needed, make_cell};
use runmat_filesystem::OpenOptions;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fopen")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fopen",
    op_kind: GpuOpKind::Custom("file-io"),
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
        "Host-only file I/O. Inputs gathered from GPU when necessary; outputs remain on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fopen")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fopen",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O is not eligible for fusion; metadata registered for completeness only.",
};

#[runtime_builtin(
    name = "fopen",
    category = "io/filetext",
    summary = "Open a file and obtain a MATLAB-compatible file identifier.",
    keywords = "fopen,file,io,permission,encoding",
    accel = "cpu",
    builtin_path = "crate::builtins::io::filetext::fopen"
)]
fn fopen_builtin(args: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(&args)?;
    Ok(eval.first_output())
}

#[derive(Clone)]
pub struct FopenEval {
    kind: FopenEvalKind,
}

#[derive(Clone)]
enum FopenEvalKind {
    Open(OpenOutputs),
    Query(QueryOutputs),
    List(Box<ListOutputs>),
}

#[derive(Clone)]
pub(crate) struct OpenOutputs {
    pub fid: f64,
    pub message: String,
    pub machinefmt: String,
    pub encoding: String,
}

#[derive(Clone)]
pub(crate) struct QueryOutputs {
    pub filename: String,
    pub permission: String,
    pub machinefmt: String,
    pub encoding: String,
}

#[derive(Clone)]
pub(crate) struct ListOutputs {
    pub handles: Tensor,
    pub names: Value,
    pub machinefmts: Value,
    pub encodings: Value,
}

impl FopenEval {
    fn open(outputs: OpenOutputs) -> Self {
        Self {
            kind: FopenEvalKind::Open(outputs),
        }
    }

    fn query(outputs: QueryOutputs) -> Self {
        Self {
            kind: FopenEvalKind::Query(outputs),
        }
    }

    fn list(outputs: ListOutputs) -> Self {
        Self {
            kind: FopenEvalKind::List(Box::new(outputs)),
        }
    }

    pub fn first_output(&self) -> Value {
        match &self.kind {
            FopenEvalKind::Open(outputs) => Value::Num(outputs.fid),
            FopenEvalKind::Query(outputs) => char_array_value(&outputs.filename),
            FopenEvalKind::List(outputs) => Value::Tensor(outputs.handles.clone()),
        }
    }

    pub fn outputs(&self) -> Vec<Value> {
        match &self.kind {
            FopenEvalKind::Open(outputs) => outputs.outputs(),
            FopenEvalKind::Query(outputs) => outputs.outputs(),
            FopenEvalKind::List(outputs) => outputs.outputs(),
        }
    }

    #[cfg(test)]
    pub(crate) fn as_open(&self) -> Option<&OpenOutputs> {
        match &self.kind {
            FopenEvalKind::Open(outputs) => Some(outputs),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn as_query(&self) -> Option<&QueryOutputs> {
        match &self.kind {
            FopenEvalKind::Query(outputs) => Some(outputs),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn as_list(&self) -> Option<&ListOutputs> {
        match &self.kind {
            FopenEvalKind::List(outputs) => Some(outputs),
            _ => None,
        }
    }
}

impl OpenOutputs {
    fn success(fid: i32, machinefmt: String, encoding: String) -> Self {
        Self {
            fid: fid as f64,
            message: String::new(),
            machinefmt,
            encoding,
        }
    }

    fn failure(message: String) -> Self {
        Self {
            fid: -1.0,
            message,
            machinefmt: String::new(),
            encoding: String::new(),
        }
    }

    fn outputs(&self) -> Vec<Value> {
        vec![
            Value::Num(self.fid),
            char_array_value(&self.message),
            char_array_value(&self.machinefmt),
            char_array_value(&self.encoding),
        ]
    }
}

impl QueryOutputs {
    fn from_info(info: FileInfo) -> Self {
        let filename = match info.path {
            Some(path) => path.to_string_lossy().to_string(),
            None => info.name,
        };
        Self {
            filename,
            permission: info.permission,
            machinefmt: info.machinefmt,
            encoding: info.encoding,
        }
    }

    fn not_found() -> Self {
        Self {
            filename: String::new(),
            permission: String::new(),
            machinefmt: String::new(),
            encoding: String::new(),
        }
    }

    fn outputs(&self) -> Vec<Value> {
        vec![
            char_array_value(&self.filename),
            char_array_value(&self.permission),
            char_array_value(&self.machinefmt),
            char_array_value(&self.encoding),
        ]
    }
}

impl ListOutputs {
    fn from_infos(infos: Vec<FileInfo>) -> Result<Self, String> {
        let mut handles: Vec<f64> = infos.iter().map(|info| info.id as f64).collect();
        let rows = handles.len();
        if rows == 0 {
            handles = Vec::new();
        }
        let shape = if rows == 0 { vec![0, 1] } else { vec![rows, 1] };
        let tensor = Tensor::new(handles, shape).map_err(|e| format!("fopen: {e}"))?;

        let mut name_values = Vec::with_capacity(infos.len());
        let mut machine_values = Vec::with_capacity(infos.len());
        let mut encoding_values = Vec::with_capacity(infos.len());
        for info in &infos {
            let display = match &info.path {
                Some(path) => path.to_string_lossy().to_string(),
                None => info.name.clone(),
            };
            name_values.push(char_array_value(&display));
            machine_values.push(char_array_value(&info.machinefmt));
            encoding_values.push(char_array_value(&info.encoding));
        }

        let names = make_cell_column(name_values)?;
        let machinefmts = make_cell_column(machine_values)?;
        let encodings = make_cell_column(encoding_values)?;

        Ok(Self {
            handles: tensor,
            names,
            machinefmts,
            encodings,
        })
    }

    fn outputs(&self) -> Vec<Value> {
        vec![
            Value::Tensor(self.handles.clone()),
            self.names.clone(),
            self.machinefmts.clone(),
            self.encodings.clone(),
        ]
    }
}

#[derive(Clone)]
struct Permission {
    canonical: String,
    read: bool,
    write: bool,
    append: bool,
    create: bool,
    truncate: bool,
    binary: bool,
}

impl Permission {
    fn parse(value: Option<&Value>) -> Result<Self, String> {
        let raw = match value {
            Some(v) => {
                let text = scalar_string(
                    v,
                    "fopen: expected permission as a string scalar or character vector",
                )?;
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    return Err("fopen: permission string must not be empty".to_string());
                }
                trimmed.to_string()
            }
            None => "r".to_string(),
        };

        let mut chars = raw.chars();
        let base = chars
            .next()
            .ok_or_else(|| "fopen: permission string must not be empty".to_string())?
            .to_ascii_lowercase();

        let mut read = false;
        let mut write = false;
        let mut append = false;
        let mut create = false;
        let mut truncate = false;

        match base {
            'r' => {
                read = true;
            }
            'w' => {
                write = true;
                create = true;
                truncate = true;
            }
            'a' => {
                write = true;
                create = true;
                append = true;
            }
            _ => {
                return Err(format!("fopen: unsupported permission prefix '{base}'"));
            }
        }

        let mut plus = false;
        let mut binary = false;
        let mut explicit_text = false;

        for c in chars {
            match c {
                '+' => {
                    if plus {
                        return Err(
                            "fopen: duplicate '+' modifier in permission string".to_string()
                        );
                    }
                    plus = true;
                    read = true;
                    write = true;
                }
                'b' | 'B' => {
                    if binary {
                        return Err(
                            "fopen: duplicate 'b' modifier in permission string".to_string()
                        );
                    }
                    binary = true;
                }
                't' | 'T' => {
                    if explicit_text {
                        return Err(
                            "fopen: duplicate 't' modifier in permission string".to_string()
                        );
                    }
                    explicit_text = true;
                }
                other => {
                    return Err(format!("fopen: unrecognised permission modifier '{other}'"));
                }
            }
        }

        if binary && explicit_text {
            return Err(
                "fopen: permission modifiers 'b' and 't' are mutually exclusive".to_string(),
            );
        }

        let mut canonical = String::new();
        canonical.push(base);
        if binary {
            canonical.push('b');
        } else if explicit_text {
            canonical.push('t');
        }
        if plus {
            canonical.push('+');
        }

        Ok(Self {
            canonical,
            read,
            write,
            append,
            create,
            truncate,
            binary,
        })
    }
}

pub fn evaluate(args: &[Value]) -> Result<FopenEval, String> {
    let gathered = gather_args(args)?;
    if gathered.is_empty() {
        return handle_all(&[]);
    }
    let first = &gathered[0];
    if matches_keyword(first, "all") {
        return handle_all(&gathered[1..]);
    }
    if is_numeric_value(first) {
        return handle_query(first, &gathered[1..]);
    }
    handle_open(first, &gathered[1..])
}

fn handle_open(path_value: &Value, rest: &[Value]) -> Result<FopenEval, String> {
    if rest.len() > 3 {
        return Err("fopen: too many input arguments".to_string());
    }

    let path = value_to_path(path_value)?;
    let mut args = rest.iter();
    let permission = Permission::parse(args.next())?;
    let machinefmt = parse_machinefmt(args.next())?;
    let encoding = parse_encoding(args.next(), &permission)?;

    let mut options = OpenOptions::new();
    options.read(permission.read);
    options.write(permission.write);
    options.create(permission.create);
    options.append(permission.append);
    options.truncate(permission.truncate);

    match options.open(&path) {
        Ok(file) => {
            let handle = Arc::new(StdMutex::new(file));
            let registered = RegisteredFile {
                path: path.clone(),
                permission: permission.canonical.clone(),
                machinefmt: machinefmt.clone(),
                encoding: encoding.clone(),
                handle,
            };
            let fid = registry::register_file(registered);
            Ok(FopenEval::open(OpenOutputs::success(
                fid, machinefmt, encoding,
            )))
        }
        Err(err) => Ok(FopenEval::open(OpenOutputs::failure(err.to_string()))),
    }
}

fn handle_query(fid_value: &Value, rest: &[Value]) -> Result<FopenEval, String> {
    if !rest.is_empty() {
        return Err("fopen: too many input arguments".to_string());
    }
    let fid = parse_fid(fid_value)?;
    let outputs = match registry::info_for(fid) {
        Some(info) => QueryOutputs::from_info(info),
        None => QueryOutputs::not_found(),
    };
    Ok(FopenEval::query(outputs))
}

fn handle_all(rest: &[Value]) -> Result<FopenEval, String> {
    if rest.len() > 1 {
        return Err("fopen: too many input arguments".to_string());
    }
    let machinefmt_filter = if let Some(value) = rest.first() {
        Some(parse_machinefmt(Some(value))?)
    } else {
        None
    };
    let mut infos = registry::list_infos();
    if let Some(filter) = &machinefmt_filter {
        infos.retain(|info| info.machinefmt.eq_ignore_ascii_case(filter));
    }
    let outputs = ListOutputs::from_infos(infos)?;
    Ok(FopenEval::list(outputs))
}

fn gather_args(args: &[Value]) -> Result<Vec<Value>, String> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(gather_if_needed(value).map_err(|e| format!("fopen: {e}"))?);
    }
    Ok(gathered)
}

fn matches_keyword(value: &Value, keyword: &str) -> bool {
    extract_scalar_string(value)
        .map(|text| text.eq_ignore_ascii_case(keyword))
        .unwrap_or(false)
}

fn is_numeric_value(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_))
}

fn parse_fid(value: &Value) -> Result<i32, String> {
    let num: f64 = value
        .try_into()
        .map_err(|_| "fopen: file identifier must be numeric".to_string())?;
    if !num.is_finite() {
        return Err("fopen: file identifier must be finite".to_string());
    }
    let rounded = num.round();
    if (rounded - num).abs() > f64::EPSILON {
        return Err("fopen: file identifier must be an integer".to_string());
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err("fopen: file identifier is out of range".to_string());
    }
    Ok(rounded as i32)
}

fn value_to_path(value: &Value) -> Result<PathBuf, String> {
    let raw = scalar_string(
        value,
        "fopen: expected filename as a string scalar or character vector",
    )?;
    normalize_path(&raw)
}

fn normalize_path(raw: &str) -> Result<PathBuf, String> {
    if raw.trim().is_empty() {
        return Err("fopen: filename must not be empty".to_string());
    }
    Ok(Path::new(raw).to_path_buf())
}

fn parse_machinefmt(value: Option<&Value>) -> Result<String, String> {
    match value {
        None => Ok("native".to_string()),
        Some(v) => {
            let text = scalar_string(
                v,
                "fopen: expected machine format as a string scalar or character vector",
            )?;
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Err("fopen: machine format must not be empty".to_string());
            }
            let lower = trimmed.to_ascii_lowercase();
            let collapsed: String = lower
                .chars()
                .filter(|c| !matches!(c, '-' | '_' | ' '))
                .collect();
            if matches!(collapsed.as_str(), "native" | "n" | "system" | "default") {
                return Ok("native".to_string());
            }
            if matches!(collapsed.as_str(), "l" | "le" | "littleendian" | "pc") {
                return Ok("ieee-le".to_string());
            }
            if matches!(collapsed.as_str(), "b" | "be" | "bigendian" | "mac") {
                return Ok("ieee-be".to_string());
            }
            if matches!(collapsed.as_str(), "vaxd" | "vaxg" | "cray") {
                return Ok(collapsed);
            }
            if let Some(suffix) = lower.strip_prefix("ieee-le") {
                return Ok(format!("ieee-le{suffix}"));
            }
            if let Some(suffix) = lower.strip_prefix("ieee-be") {
                return Ok(format!("ieee-be{suffix}"));
            }
            Err(format!("fopen: unsupported machine format '{trimmed}'"))
        }
    }
}

fn parse_encoding(value: Option<&Value>, permission: &Permission) -> Result<String, String> {
    match value {
        None => {
            if permission.binary {
                Ok("binary".to_string())
            } else {
                Ok("UTF-8".to_string())
            }
        }
        Some(v) => {
            let text = scalar_string(
                v,
                "fopen: expected encoding as a string scalar or character vector",
            )?;
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Err("fopen: encoding name must not be empty".to_string());
            }
            Ok(normalize_encoding_label(trimmed))
        }
    }
}

fn normalize_encoding_label(label: &str) -> String {
    let lower = label.to_ascii_lowercase();
    match lower.as_str() {
        "utf-8" | "utf8" | "unicode" => "UTF-8".to_string(),
        "ascii" | "us-ascii" | "us_ascii" | "usascii" => "US-ASCII".to_string(),
        "latin1" | "latin-1" | "iso-8859-1" | "iso8859-1" => "latin1".to_string(),
        "windows-1252" | "cp1252" => "windows-1252".to_string(),
        "shift-jis" | "shift_jis" | "sjis" => "Shift_JIS".to_string(),
        "binary" => "binary".to_string(),
        "system" | "default" | "native" => "system".to_string(),
        _ => label.to_string(),
    }
}

fn scalar_string(value: &Value, err: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(err.to_string()),
    }
}

fn extract_scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

fn make_cell_column(values: Vec<Value>) -> Result<Value, String> {
    let len = values.len();
    if len == 0 {
        make_cell(values, 0, 0)
    } else {
        make_cell(values, len, 1)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::filetext::registry;
    use runmat_filesystem as fs;
    use runmat_time::system_time_now;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    fn unique_path(prefix: &str) -> PathBuf {
        let now = system_time_now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards");
        let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
        std::env::temp_dir().join(filename)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_read_existing_file_returns_fid() {
        registry::reset_for_tests();
        let path = unique_path("fopen_read");
        fs::write(&path, "hello world").unwrap();

        let args = vec![Value::from(path.to_string_lossy().to_string())];
        let eval = evaluate(&args).expect("fopen");
        let open = eval.as_open().expect("expected open result");
        assert!(open.fid >= 3.0);
        assert!(open.message.is_empty());
        assert_eq!(open.machinefmt, "native");
        assert_eq!(open.encoding, "UTF-8");

        let _ = registry::close(open.fid as i32);
        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_missing_file_returns_error() {
        registry::reset_for_tests();
        let path = unique_path("fopen_missing");
        let args = vec![Value::from(path.to_string_lossy().to_string())];
        let eval = evaluate(&args).expect("fopen");
        let open = eval.as_open().expect("open output");
        assert_eq!(open.fid, -1.0);
        assert!(!open.message.is_empty());
        assert!(open.machinefmt.is_empty());
        assert!(open.encoding.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_query_returns_metadata() {
        registry::reset_for_tests();
        let path = unique_path("fopen_query");
        {
            let mut file = fs::File::create(&path).unwrap();
            writeln!(&mut file, "data").unwrap();
        }
        let args = vec![Value::from(path.to_string_lossy().to_string())];
        let eval = evaluate(&args).expect("fopen");
        let open = eval.as_open().expect("open result");
        let fid = open.fid;
        assert!(fid >= 3.0);

        let query_eval = evaluate(&[Value::from(fid)]).expect("fopen query");
        let query = query_eval.as_query().expect("query result");
        assert!(query.filename.contains("fopen_query"));
        assert_eq!(query.permission, "r");
        assert_eq!(query.machinefmt, "native");

        let _ = registry::close(fid as i32);
        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_all_lists_handles() {
        registry::reset_for_tests();
        let path = unique_path("fopen_all");
        fs::write(&path, "abc").unwrap();
        let eval_open =
            evaluate(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
        let fid = eval_open.as_open().unwrap().fid;

        let list_eval = evaluate(&[Value::from("all")]).expect("fopen all");
        let list = list_eval.as_list().expect("list result");
        assert!(!list.handles.data.is_empty());
        assert!(list
            .handles
            .data
            .iter()
            .any(|v| (*v - fid).abs() < f64::EPSILON));

        if let Value::Cell(names) = &list.names {
            assert_eq!(names.data.len(), list.handles.data.len());
            assert_eq!(names.rows, list.handles.data.len());
            assert_eq!(names.cols, 1);
        } else {
            panic!("expected cell array for names");
        }
        if let Value::Cell(machinefmts) = &list.machinefmts {
            assert_eq!(machinefmts.rows, list.handles.data.len());
            assert_eq!(machinefmts.cols, 1);
        } else {
            panic!("expected cell array for machine formats");
        }
        if let Value::Cell(encodings) = &list.encodings {
            assert_eq!(encodings.rows, list.handles.data.len());
            assert_eq!(encodings.cols, 1);
        } else {
            panic!("expected cell array for encodings");
        }

        let _ = registry::close(fid as i32);
        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_all_machinefmt_filters_entries() {
        registry::reset_for_tests();
        let native_path = unique_path("fopen_native");
        let be_path = unique_path("fopen_ieee_be");
        fs::write(&native_path, "native").unwrap();
        fs::write(&be_path, "be").unwrap();

        let native_eval =
            evaluate(&[Value::from(native_path.to_string_lossy().to_string())]).expect("native");
        let native_fid = native_eval.as_open().unwrap().fid;

        let be_eval = evaluate(&[
            Value::from(be_path.to_string_lossy().to_string()),
            Value::from("r"),
            Value::from("ieee-be"),
        ])
        .expect("ieee-be");
        let be_fid = be_eval.as_open().unwrap().fid;

        let list_eval =
            evaluate(&[Value::from("all"), Value::from("ieee-be")]).expect("fopen all filter");
        let list = list_eval.as_list().expect("list result");
        assert_eq!(list.handles.data.len(), 1);
        assert!((list.handles.data[0] - be_fid).abs() < f64::EPSILON);

        let _ = registry::close(native_fid as i32);
        let _ = registry::close(be_fid as i32);
        fs::remove_file(&native_path).unwrap();
        fs::remove_file(&be_path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_binary_default_encoding_binary() {
        registry::reset_for_tests();
        let path = unique_path("fopen_binary");
        {
            let _ = fs::File::create(&path).unwrap();
        }
        let eval = evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("wb"),
        ])
        .expect("fopen");
        let open = eval.as_open().unwrap();
        assert_eq!(open.encoding, "binary");
        let _ = registry::close(open.fid as i32);
        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_encoding_argument_is_preserved() {
        registry::reset_for_tests();
        let path = unique_path("fopen_encoding");
        let eval = evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
            Value::from("ieee-be"),
            Value::from("latin1"),
        ])
        .expect("fopen");
        let open = eval.as_open().unwrap();
        assert_eq!(open.machinefmt, "ieee-be");
        assert_eq!(open.encoding, "latin1");
        let _ = registry::close(open.fid as i32);
        if path.exists() {
            fs::remove_file(&path).unwrap();
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_permission_canonicalizes_plus_binary_order() {
        registry::reset_for_tests();
        let path = unique_path("fopen_perm_order");
        fs::write(&path, "seed").unwrap();
        let eval = evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("r+b"),
        ])
        .expect("fopen");
        let open = eval.as_open().unwrap();
        assert!(open.fid >= 3.0);
        assert_eq!(open.encoding, "binary");
        let query = evaluate(&[Value::Num(open.fid)]).expect("query");
        let info = query.as_query().unwrap();
        assert_eq!(info.permission, "rb+");
        let _ = registry::close(open.fid as i32);
        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_machinefmt_preserves_suffix() {
        registry::reset_for_tests();
        let path = unique_path("fopen_machinefmt_suffix");
        let eval = evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
            Value::from("ieee-be.l64"),
        ])
        .expect("fopen");
        let open = eval.as_open().unwrap();
        assert_eq!(open.machinefmt, "ieee-be.l64");
        let query = evaluate(&[Value::Num(open.fid)]).expect("query");
        let info = query.as_query().unwrap();
        assert_eq!(info.machinefmt, "ieee-be.l64");
        let _ = registry::close(open.fid as i32);
        if path.exists() {
            fs::remove_file(&path).unwrap();
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_machinefmt_pc_alias_maps_to_ieee_le() {
        registry::reset_for_tests();
        let path = unique_path("fopen_machinefmt_pc");
        let eval = evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
            Value::from("pc"),
        ])
        .expect("fopen");
        let open = eval.as_open().unwrap();
        assert_eq!(open.machinefmt, "ieee-le");
        let query = evaluate(&[Value::Num(open.fid)]).expect("query");
        let info = query.as_query().unwrap();
        assert_eq!(info.machinefmt, "ieee-le");
        let _ = registry::close(open.fid as i32);
        if path.exists() {
            fs::remove_file(&path).unwrap();
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_outputs_vector_padding() {
        registry::reset_for_tests();
        let path = unique_path("fopen_outputs");
        fs::write(&path, "check").unwrap();
        let eval = evaluate(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 4);
        assert!(matches!(outputs[0], Value::Num(_)));
        assert!(matches!(outputs[1], Value::CharArray(_)));
        let _ = registry::close(eval.as_open().unwrap().fid as i32);
        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fopen_invalid_fid_returns_empty() {
        registry::reset_for_tests();
        let eval = evaluate(&[Value::from(9999.0)]).expect("fopen");
        let query = eval.as_query().expect("query result");
        assert!(query.filename.is_empty());
        assert!(query.permission.is_empty());
    }
}
