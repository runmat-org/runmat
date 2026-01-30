//! MATLAB-compatible `fclose` builtin for RunMat.
//!
//! Mirrors MATLAB semantics for closing individual files, vectors of file
//! identifiers, or all open files. The implementation integrates with the
//! shared file registry managed by `fopen` and always executes on the host.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::{
    helpers::{char_array_value, extract_scalar_string},
    registry,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const INVALID_IDENTIFIER_MESSAGE: &str =
    "Invalid file identifier. Use fopen to generate a valid file ID.";
const BUILTIN_NAME: &str = "fclose";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fclose")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fclose",
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
        "Host-only operation: closes identifiers stored in the shared file registry; GPU inputs are gathered automatically.",
};

fn fclose_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let message = err.message().to_string();
    let identifier = err.identifier().map(|value| value.to_string());
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {message}"))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fclose")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fclose",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O is not eligible for fusion; metadata is registered for completeness.",
};

#[runtime_builtin(
    name = "fclose",
    category = "io/filetext",
    summary = "Close one file, multiple files, or all files opened with fopen.",
    keywords = "fclose,file,close,io,identifier",
    accel = "cpu",
    builtin_path = "crate::builtins::io::filetext::fclose"
)]
async fn fclose_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&args).await?;
    Ok(eval.first_output())
}

#[derive(Debug, Clone)]
pub struct FcloseEval {
    status: f64,
    message: String,
}

impl FcloseEval {
    fn success() -> Self {
        Self {
            status: 0.0,
            message: String::new(),
        }
    }

    fn failure(message: String) -> Self {
        Self {
            status: -1.0,
            message,
        }
    }

    pub fn first_output(&self) -> Value {
        Value::Num(self.status)
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![Value::Num(self.status), char_array_value(&self.message)]
    }

    #[cfg(test)]
    pub(crate) fn status(&self) -> f64 {
        self.status
    }

    #[cfg(test)]
    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

pub async fn evaluate(args: &[Value]) -> BuiltinResult<FcloseEval> {
    let gathered = gather_args(args).await?;
    match gathered.len() {
        0 => Ok(close_all()),
        1 => handle_single_argument(&gathered[0]),
        _ => Err(fclose_error("fclose: too many input arguments")),
    }
}

fn handle_single_argument(value: &Value) -> BuiltinResult<FcloseEval> {
    if matches_keyword(value, "all") {
        return Ok(close_all());
    }
    let fids = collect_file_ids(value).map_err(|err| fclose_error(format!("fclose: {err}")))?;
    Ok(close_fids(&fids))
}

fn close_all() -> FcloseEval {
    let infos = registry::list_infos();
    for info in infos {
        if info.id >= 3 {
            let _ = registry::close(info.id);
        }
    }
    FcloseEval::success()
}

fn close_fids(fids: &[i32]) -> FcloseEval {
    if fids.is_empty() {
        return FcloseEval::success();
    }
    let mut status_ok = true;
    let mut message = String::new();
    for &fid in fids {
        if fid < 0 {
            status_ok = false;
            if message.is_empty() {
                message = INVALID_IDENTIFIER_MESSAGE.to_string();
            }
            continue;
        }
        if fid < 3 {
            continue;
        }
        if registry::close(fid).is_none() {
            status_ok = false;
            if message.is_empty() {
                message = INVALID_IDENTIFIER_MESSAGE.to_string();
            }
        }
    }
    if status_ok {
        FcloseEval::success()
    } else {
        FcloseEval::failure(message)
    }
}

fn collect_file_ids(value: &Value) -> BuiltinResult<Vec<i32>> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(vec![parse_scalar_fid(value)?]),
        Value::Tensor(t) => {
            let mut ids = Vec::with_capacity(t.data.len());
            for &n in &t.data {
                ids.push(parse_fid_from_f64(n)?);
            }
            Ok(ids)
        }
        Value::LogicalArray(la) => {
            let mut ids = Vec::with_capacity(la.data.len());
            for &b in &la.data {
                let v = if b != 0 { 1 } else { 0 };
                ids.push(v);
            }
            Ok(ids)
        }
        Value::Cell(ca) => {
            let mut ids = Vec::with_capacity(ca.data.len());
            for ptr in &ca.data {
                let nested = collect_file_ids(ptr)?;
                ids.extend(nested);
            }
            Ok(ids)
        }
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err(fclose_error("file identifier must be numeric or 'all'"))
        }
        _ => Err(fclose_error("file identifier must be numeric or 'all'")),
    }
}

fn parse_scalar_fid(value: &Value) -> BuiltinResult<i32> {
    match value {
        Value::Int(i) => {
            let v = i.to_i64();
            if v < i32::MIN as i64 || v > i32::MAX as i64 {
                return Err(fclose_error("file identifier is out of range"));
            }
            Ok(v as i32)
        }
        Value::Num(n) => parse_fid_from_f64(*n),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        _ => Err(fclose_error("file identifier must be numeric or 'all'")),
    }
}

fn parse_fid_from_f64(value: f64) -> BuiltinResult<i32> {
    if !value.is_finite() {
        return Err(fclose_error("file identifier must be finite"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(fclose_error("file identifier must be an integer"));
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err(fclose_error("file identifier is out of range"));
    }
    Ok(rounded as i32)
}

async fn gather_args(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(gathered)
}

fn matches_keyword(value: &Value, keyword: &str) -> bool {
    extract_scalar_string(value)
        .map(|text| text.eq_ignore_ascii_case(keyword))
        .unwrap_or(false)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::filetext::{fopen, registry};
    use runmat_builtins::{CellArray, LogicalArray, StringArray, Tensor};
    use runmat_filesystem as fs;
    use runmat_time::system_time_now;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::MutexGuard;
    use std::time::UNIX_EPOCH;

    fn unwrap_error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_evaluate(args: &[Value]) -> BuiltinResult<FcloseEval> {
        futures::executor::block_on(evaluate(args))
    }

    fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
        futures::executor::block_on(fopen::evaluate(args))
    }

    fn registry_guard() -> MutexGuard<'static, ()> {
        registry::test_guard()
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let now = system_time_now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards");
        let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
        std::env::temp_dir().join(filename)
    }

    fn open_temp_file(prefix: &str) -> (f64, PathBuf) {
        let path = unique_path(prefix);
        {
            let mut file = fs::File::create(&path).unwrap();
            writeln!(&mut file, "data").unwrap();
        }
        let eval = run_fopen(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
        let fid = eval.as_open().unwrap().fid;
        assert!(fid >= 3.0, "expected valid file identifier");
        (fid, path)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_closes_single_file() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_single");
        let eval = run_evaluate(&[Value::Num(fid)]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
        assert!(registry::info_for(fid as i32).is_none());
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_invalid_identifier_returns_error() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let eval = run_evaluate(&[Value::Num(9999.0)]).expect("fclose");
        assert_eq!(eval.status(), -1.0);
        assert_eq!(eval.message(), INVALID_IDENTIFIER_MESSAGE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_all_closes_everything() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_all");
        let eval = run_evaluate(&[Value::from("all")]).expect("fclose all");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid as i32).is_none());
        let infos = registry::list_infos();
        assert!(infos.iter().all(|info| info.id < 3));
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_no_args_closes_all() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_no_args");
        let eval = run_evaluate(&[]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid as i32).is_none());
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_vector_of_fids_closes_each() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path1 = unique_path("fclose_vec1");
        fs::write(&path1, "a").unwrap();
        let fid1 = run_fopen(&[Value::from(path1.to_string_lossy().to_string())])
            .expect("open 1")
            .as_open()
            .unwrap()
            .fid;
        let path2 = unique_path("fclose_vec2");
        fs::write(&path2, "b").unwrap();
        let fid2 = run_fopen(&[Value::from(path2.to_string_lossy().to_string())])
            .expect("open 2")
            .as_open()
            .unwrap()
            .fid;
        let tensor = Tensor::new(vec![fid1, fid2], vec![2, 1]).expect("tensor construction");
        let eval = run_evaluate(&[Value::Tensor(tensor)]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid1 as i32).is_none());
        assert!(registry::info_for(fid2 as i32).is_none());
        fs::remove_file(path1).unwrap();
        fs::remove_file(path2).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_repeat_returns_error_message() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_repeat");
        let first = run_evaluate(&[Value::Num(fid)]).expect("fclose");
        assert_eq!(first.status(), 0.0);
        let second = run_evaluate(&[Value::Num(fid)]).expect("fclose second");
        assert_eq!(second.status(), -1.0);
        assert_eq!(second.message(), INVALID_IDENTIFIER_MESSAGE);
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_standard_stream_bool_argument() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let eval = run_evaluate(&[Value::Bool(true)]).expect("fclose stdout");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 2);
        assert!(matches!(outputs[1], Value::CharArray(ref ca) if ca.rows == 1 && ca.cols == 0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_logical_array_converts_to_numeric_ids() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let logical = LogicalArray::new(vec![1u8, 0u8, 1u8], vec![3]).expect("logical array");
        let eval = run_evaluate(&[Value::LogicalArray(logical)]).expect("fclose logical");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_cell_array_closes_each_entry() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid1, path1) = open_temp_file("fclose_cell1");
        let (fid2, path2) = open_temp_file("fclose_cell2");
        let cell = CellArray::new(vec![Value::Num(fid1), Value::Num(fid2)], 1, 2).expect("cell");
        let eval = run_evaluate(&[Value::Cell(cell)]).expect("fclose cell");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid1 as i32).is_none());
        assert!(registry::info_for(fid2 as i32).is_none());
        fs::remove_file(path1).unwrap();
        fs::remove_file(path2).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_tensor_with_non_integer_entries_errors() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let tensor = Tensor::new(vec![1.5], vec![1, 1]).expect("tensor");
        let err = unwrap_error_message(run_evaluate(&[Value::Tensor(tensor)]).unwrap_err());
        assert_eq!(err, "fclose: file identifier must be an integer");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_string_array_all_equivalent() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let strings = StringArray::new(vec!["all".to_string()], vec![1]).expect("string array");
        let eval = run_evaluate(&[Value::StringArray(strings)]).expect("fclose all");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_accepts_empty_tensor() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).expect("tensor");
        let eval = run_evaluate(&[Value::Tensor(tensor)]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fclose_errors_on_non_numeric_input() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let err = unwrap_error_message(run_evaluate(&[Value::from("not-a-fid")]).unwrap_err());
        assert_eq!(err, "fclose: file identifier must be numeric or 'all'");
    }
}
