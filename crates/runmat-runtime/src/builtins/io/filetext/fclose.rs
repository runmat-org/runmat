//! MATLAB-compatible `fclose` builtin for RunMat.
//!
//! MATLAB-compatible mode closes one scalar double file identifier or all open
//! files through the character-vector `all` form. Broader RunMat forms are
//! extension-gated. The implementation integrates with the shared file registry
//! managed by `fopen` and always executes on the host.

#[cfg(test)]
use runmat_builtins::FCLOSE_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, FCLOSE_CELL_ID_EXTENSION, FCLOSE_ERROR_INTERNAL,
    FCLOSE_ERROR_INVALID_IDENTIFIER, FCLOSE_ERROR_INVALID_INPUT, FCLOSE_INTEGER_ID_EXTENSION,
    FCLOSE_LOGICAL_ID_EXTENSION, FCLOSE_MESSAGE_OUTPUT_EXTENSION, FCLOSE_NO_INPUT_EXTENSION,
    FCLOSE_RESIDENT_ID_EXTENSION, FCLOSE_SINGLE_ID_EXTENSION, FCLOSE_STRING_ALL_EXTENSION,
    FCLOSE_VECTOR_ID_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, Value};

use crate::builtins::common::tensor;
use crate::builtins::io::filetext::{
    helpers::{char_array_value, extract_scalar_string},
    registry,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fclose";

fn fclose_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fclose_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fclose_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FCLOSE_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "fclose",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::filetext::fclose"
)]
async fn fclose_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    preflight_fclose_outputs()?;
    let eval = evaluate_retained(runtime.as_ref(), args).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        return Ok(Value::output_list(
            eval.outputs()
                .into_iter()
                .take(out_count)
                .collect::<Vec<_>>(),
        ));
    }
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
    evaluate_in(None, args).await
}

pub async fn evaluate_in(
    runtime: Option<&crate::context::RuntimeContext>,
    args: &[Value],
) -> BuiltinResult<FcloseEval> {
    let args = crate::call::BuiltinArguments::from_borrowed(args, runtime)?;
    evaluate_retained(runtime, args).await
}

async fn evaluate_retained(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<FcloseEval> {
    preflight_fclose_inputs(args.view())?;
    let gathered =
        super::cooperation::await_io(runtime, BUILTIN_NAME, gather_args(runtime, &args)).await??;
    match gathered.len() {
        0 => close_all(runtime).await,
        1 => handle_single_argument(runtime, &gathered[0]).await,
        _ => Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "too many input arguments",
        )),
    }
}

fn preflight_fclose_outputs() -> BuiltinResult<()> {
    let Some(count) = crate::output_count::current_output_count() else {
        return Ok(());
    };
    if count > 2 {
        return Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "too many output arguments",
        ));
    }
    if count == 2 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FCLOSE_MESSAGE_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn preflight_fclose_inputs(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    match args.len() {
        0 => crate::compatibility::ensure_builtin_extension_enabled(
            &FCLOSE_NO_INPUT_EXTENSION,
            BUILTIN_NAME,
        ),
        1 => preflight_fclose_value(&args[0]),
        _ => Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "too many input arguments",
        )),
    }
}

fn preflight_fclose_value(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) | Value::CharArray(_) => Ok(()),
        Value::Int(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FCLOSE_INTEGER_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Bool(_) => crate::compatibility::ensure_builtin_extension_enabled(
            &FCLOSE_LOGICAL_ID_EXTENSION,
            BUILTIN_NAME,
        ),
        Value::Tensor(tensor) => {
            if tensor.len() != 1 {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_VECTOR_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if tensor.integer_storage().is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if tensor.numeric_dtype() == NumericDType::F32 {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_SINGLE_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        Value::LogicalArray(array) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FCLOSE_LOGICAL_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if array.len() != 1 {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_VECTOR_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        Value::Cell(cell) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FCLOSE_CELL_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            for nested in cell.data() {
                preflight_fclose_value(nested)?;
            }
            Ok(())
        }
        Value::String(_) | Value::StringArray(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FCLOSE_STRING_ALL_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FCLOSE_RESIDENT_ID_EXTENSION,
                BUILTIN_NAME,
            )?;
            if tensor::element_count(handle.shape()) != 1 {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_VECTOR_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_INTEGER_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_LOGICAL_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FCLOSE_SINGLE_ID_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

async fn handle_single_argument(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
) -> BuiltinResult<FcloseEval> {
    if matches_keyword(value, "all") {
        return close_all(runtime).await;
    }
    let fids = collect_file_ids(value)?;
    close_fids(runtime, &fids).await
}

async fn close_all(runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<FcloseEval> {
    let infos = registry::list_infos_in(runtime);
    let mut status_ok = true;
    let mut message = String::new();
    for info in infos {
        if info.id >= 3 {
            if let Err(err) = close_one(runtime, info.id).await? {
                status_ok = false;
                if message.is_empty() {
                    message = err.to_string();
                }
            }
        }
    }
    if status_ok {
        Ok(FcloseEval::success())
    } else {
        Ok(FcloseEval::failure(message))
    }
}

async fn close_fids(
    runtime: Option<&crate::context::RuntimeContext>,
    fids: &[i32],
) -> BuiltinResult<FcloseEval> {
    if fids.is_empty() {
        return Ok(FcloseEval::success());
    }
    let mut status_ok = true;
    let mut message = String::new();
    for &fid in fids {
        if fid < 0 {
            status_ok = false;
            if message.is_empty() {
                message = FCLOSE_ERROR_INVALID_IDENTIFIER.message.to_string();
            }
            continue;
        }
        if fid < 3 {
            continue;
        }
        match close_one(runtime, fid).await? {
            Ok(Some(_)) => {}
            Ok(None) => {
                status_ok = false;
                if message.is_empty() {
                    message = FCLOSE_ERROR_INVALID_IDENTIFIER.message.to_string();
                }
            }
            Err(err) => {
                status_ok = false;
                if message.is_empty() {
                    message = err.to_string();
                }
            }
        }
    }
    if status_ok {
        Ok(FcloseEval::success())
    } else {
        Ok(FcloseEval::failure(message))
    }
}

async fn close_one(
    runtime: Option<&crate::context::RuntimeContext>,
    fid: i32,
) -> BuiltinResult<std::io::Result<Option<registry::FileInfo>>> {
    let prepared = match super::cooperation::await_io(
        runtime,
        BUILTIN_NAME,
        registry::prepare_close_async_in(fid, runtime),
    )
    .await?
    {
        Ok(prepared) => prepared,
        Err(error) => return Ok(Err(error)),
    };
    // The checked-out entry remains owned here. An inline cancellation from a
    // Ready flush, or a pending cooperative turn, restores its exact handle.
    super::cooperation::checkpoint(runtime, BUILTIN_NAME).await?;
    Ok(prepared.map(|entry| entry.commit(runtime)).transpose())
}

fn collect_file_ids(value: &Value) -> BuiltinResult<Vec<i32>> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(vec![parse_scalar_fid(value)?]),
        Value::Tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                return storage
                    .exact_values()
                    .iter()
                    .map(parse_integer_fid)
                    .collect();
            }
            let mut ids = Vec::with_capacity(t.len());
            for n in tensor::tensor_values_f64(t) {
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
            let mut ids = Vec::with_capacity(ca.data().len());
            for ptr in ca.data() {
                let nested = collect_file_ids(ptr)?;
                ids.extend(nested);
            }
            Ok(ids)
        }
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err(fclose_error_with_detail(
                &FCLOSE_ERROR_INVALID_INPUT,
                "file identifier must be numeric or 'all'",
            ))
        }
        _ => Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier must be numeric or 'all'",
        )),
    }
}

fn parse_scalar_fid(value: &Value) -> BuiltinResult<i32> {
    match value {
        Value::Int(i) => parse_integer_fid(i),
        Value::Num(n) => parse_fid_from_f64(*n),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        _ => Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier must be numeric or 'all'",
        )),
    }
}

fn parse_integer_fid(value: &IntValue) -> BuiltinResult<i32> {
    value.try_to_i32().ok_or_else(|| {
        fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier is out of range",
        )
    })
}

fn parse_fid_from_f64(value: f64) -> BuiltinResult<i32> {
    if !value.is_finite() {
        return Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier must be finite",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier must be an integer",
        ));
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err(fclose_error_with_detail(
            &FCLOSE_ERROR_INVALID_INPUT,
            "file identifier is out of range",
        ));
    }
    Ok(rounded as i32)
}

async fn gather_args(
    runtime: Option<&crate::context::RuntimeContext>,
    args: &crate::call::BuiltinArguments,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    args.try_map_preserving_roots_async(args.len(), runtime, async |index| {
        gather_if_needed_async(&args[index])
            .await
            .map_err(map_control_flow)
    })
    .await
}

fn matches_keyword(value: &Value, keyword: &str) -> bool {
    extract_scalar_string(value)
        .map(|text| text.eq_ignore_ascii_case(keyword))
        .unwrap_or(false)
}

#[cfg(test)]
pub(crate) mod tests;
