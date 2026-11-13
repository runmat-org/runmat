//! MATLAB-compatible HDF5 structured file I/O builtins and native backend.

use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use runmat_builtins::HDF5_ERROR_PROVIDER as ERR_PROVIDER;
#[cfg(target_arch = "wasm32")]
use runmat_builtins::HDF5_ERROR_WASM as ERR_WASM;
use runmat_builtins::{
    BuiltinDescriptor, BuiltinErrorDescriptor, H5DISP_CAPTURE_EXTENSION, H5DISP_DESCRIPTOR,
    H5INFO_DESCRIPTOR, H5READ_DESCRIPTOR, H5WRITEATT_DESCRIPTOR, H5WRITE_DESCRIPTOR,
    HDF5INFO_DESCRIPTOR, HDF5READ_DESCRIPTOR, HDF5WRITE_DESCRIPTOR,
    HDF5_ERROR_ARGUMENT as ERR_ARGUMENT, HDF5_ERROR_IO as ERR_IO,
    HDF5_ERROR_UNSUPPORTED as ERR_UNSUPPORTED,
};
use runmat_filesystem::NativePathIntent;
use runmat_macros::runtime_builtin;
#[cfg(not(target_arch = "wasm32"))]
use runmat_value::NumericStorage;
use runmat_value::{
    IntValue, IntegerStorage, LogicalArray, NumericScalar, StringArray, Tensor, Value,
};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::validation::value_text_scalar_string;
use crate::console::{record_console_line, ConsoleStream};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

mod path;
use path::Hdf5PathLease;
mod h5disp_controls;
use h5disp_controls::H5DispControls;

const H5READ_NAME: &str = "h5read";
const H5WRITE_NAME: &str = "h5write";
const H5WRITEATT_NAME: &str = "h5writeatt";
const H5INFO_NAME: &str = "h5info";
const H5DISP_NAME: &str = "h5disp";
const HDF5READ_NAME: &str = "hdf5read";
const HDF5WRITE_NAME: &str = "hdf5write";
const HDF5INFO_NAME: &str = "hdf5info";

fn hdf5_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn admit_output_count(
    builtin: &'static str,
    descriptor: &'static BuiltinDescriptor,
) -> BuiltinResult<()> {
    crate::output_count::validate_fixed_outputs(descriptor, &ERR_ARGUMENT, builtin).map_err(
        |source| {
            build_runtime_error(format!("{builtin}: too many output arguments"))
                .with_builtin(builtin)
                .with_identifier(
                    ERR_ARGUMENT
                        .identifier
                        .expect("declared HDF5 argument identifier"),
                )
                .with_source(source)
                .build()
        },
    )
}

fn map_control_flow(builtin: &'static str, err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(|value| value.to_string());
    let message = err.message().to_string();
    let mut builder = build_runtime_error(message)
        .with_builtin(builtin)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn string_scalar(value: &Value, builtin: &'static str, label: &str) -> BuiltinResult<String> {
    value_text_scalar_string(value).ok_or_else(|| {
        hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: expected {label} as a string scalar or character vector"),
        )
    })
}

fn resolve_path(value: &Value, builtin: &'static str) -> BuiltinResult<PathBuf> {
    let raw = string_scalar(value, builtin, "filename")?;
    if raw.trim().is_empty() {
        return Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: filename must not be empty"),
        ));
    }
    let expanded = expand_user_path(raw.trim(), builtin)
        .map_err(|msg| hdf5_error(builtin, &ERR_ARGUMENT, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn hdf5_path(value: &Value, builtin: &'static str, label: &str) -> BuiltinResult<String> {
    let raw = string_scalar(value, builtin, label)?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: {label} must not be empty"),
        ));
    }
    Ok(if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{trimmed}")
    })
}

async fn finish_selection_args(
    builtin: &'static str,
    runtime: Option<&crate::context::RuntimeContext>,
    rest: &crate::call::BuiltinArguments,
    host_controls: Option<SelectionControls>,
) -> BuiltinResult<SelectionControls> {
    if let Some(controls) = host_controls {
        return Ok(controls);
    }
    let gather = rest.try_map_preserving_roots_async(rest.len(), runtime, async |index| {
        gather_if_needed_async(&rest[index])
            .await
            .map_err(|err| map_control_flow(builtin, err))
    });
    let gathered = match runtime {
        Some(runtime) => runtime.scope(gather).await?,
        None => gather.await?,
    };
    parse_selection_controls(gathered.view(), builtin)
}

fn preflight_writable_data(value: &Value, builtin: &'static str) -> BuiltinResult<()> {
    if matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Tensor(_)
            | Value::Bool(_)
            | Value::LogicalArray(_)
            | Value::String(_)
            | Value::StringArray(_)
            | Value::CharArray(_)
            | Value::GpuTensor(_)
    ) {
        Ok(())
    } else {
        Err(hdf5_error(
            builtin,
            &ERR_UNSUPPORTED,
            format!("{builtin}: only dense numeric, logical, string, and char data can be written to HDF5"),
        ))
    }
}

#[runtime_builtin(
    name = "h5read",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn h5read_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    dataset: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    admit_output_count(H5READ_NAME, &H5READ_DESCRIPTOR)?;
    let path = resolve_path(&filename, H5READ_NAME)?;
    let dataset_path = hdf5_path(&dataset, H5READ_NAME, "dataset")?;
    let host_controls = prepare_host_selection_controls(rest.view(), H5READ_NAME)?;
    let path = path::capture(path, NativePathIntent::Read, H5READ_NAME)?;
    let selection =
        finish_selection_args(H5READ_NAME, runtime.as_ref(), &rest, host_controls).await?;
    path::check(&path, H5READ_NAME)?;
    let result = imp::read_dataset(H5READ_NAME, &path, dataset_path, selection)?;
    path::check(&path, H5READ_NAME)?;
    Ok(result)
}

#[runtime_builtin(
    name = "hdf5read",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn hdf5read_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    dataset: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    admit_output_count(HDF5READ_NAME, &HDF5READ_DESCRIPTOR)?;
    let path = resolve_path(&filename, HDF5READ_NAME)?;
    let dataset_path = hdf5_path(&dataset, HDF5READ_NAME, "dataset")?;
    let host_controls = prepare_host_selection_controls(rest.view(), HDF5READ_NAME)?;
    let path = path::capture(path, NativePathIntent::Read, HDF5READ_NAME)?;
    let selection =
        finish_selection_args(HDF5READ_NAME, runtime.as_ref(), &rest, host_controls).await?;
    path::check(&path, HDF5READ_NAME)?;
    let result = imp::read_dataset(HDF5READ_NAME, &path, dataset_path, selection)?;
    path::check(&path, HDF5READ_NAME)?;
    Ok(result)
}

#[runtime_builtin(
    name = "h5write",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn h5write_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    dataset: Value,
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    admit_output_count(H5WRITE_NAME, &H5WRITE_DESCRIPTOR)?;
    let path = resolve_path(&filename, H5WRITE_NAME)?;
    let dataset_path = hdf5_path(&dataset, H5WRITE_NAME, "dataset")?;
    preflight_writable_data(&data, H5WRITE_NAME)?;
    let host_controls = prepare_host_selection_controls(rest.view(), H5WRITE_NAME)?;
    let path = path::capture(path, NativePathIntent::ReadWriteCreate, H5WRITE_NAME)?;
    let selection =
        finish_selection_args(H5WRITE_NAME, runtime.as_ref(), &rest, host_controls).await?;
    let gather = gather_if_needed_async(&data);
    let data = match runtime.as_ref() {
        Some(runtime) => runtime.scope(gather).await,
        None => gather.await,
    }
    .map_err(|err| map_control_flow(H5WRITE_NAME, err))?;
    path::check(&path, H5WRITE_NAME)?;
    imp::write_dataset(H5WRITE_NAME, &path, dataset_path, data, selection)?;
    path::check(&path, H5WRITE_NAME)?;
    Ok(Value::output_list(Vec::new()))
}

#[runtime_builtin(
    name = "hdf5write",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn hdf5write_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    dataset: Value,
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    admit_output_count(HDF5WRITE_NAME, &HDF5WRITE_DESCRIPTOR)?;
    let path = resolve_path(&filename, HDF5WRITE_NAME)?;
    let dataset_path = hdf5_path(&dataset, HDF5WRITE_NAME, "dataset")?;
    preflight_writable_data(&data, HDF5WRITE_NAME)?;
    let host_controls = prepare_host_selection_controls(rest.view(), HDF5WRITE_NAME)?;
    let path = path::capture(path, NativePathIntent::ReadWriteCreate, HDF5WRITE_NAME)?;
    let selection =
        finish_selection_args(HDF5WRITE_NAME, runtime.as_ref(), &rest, host_controls).await?;
    let gather = gather_if_needed_async(&data);
    let data = match runtime.as_ref() {
        Some(runtime) => runtime.scope(gather).await,
        None => gather.await,
    }
    .map_err(|err| map_control_flow(HDF5WRITE_NAME, err))?;
    path::check(&path, HDF5WRITE_NAME)?;
    imp::write_dataset(HDF5WRITE_NAME, &path, dataset_path, data, selection)?;
    path::check(&path, HDF5WRITE_NAME)?;
    Ok(Value::output_list(Vec::new()))
}

#[runtime_builtin(
    name = "h5writeatt",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn h5writeatt_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    location: Value,
    attribute: Value,
    data: Value,
) -> BuiltinResult<Value> {
    admit_output_count(H5WRITEATT_NAME, &H5WRITEATT_DESCRIPTOR)?;
    let path = resolve_path(&filename, H5WRITEATT_NAME)?;
    let location_path = hdf5_path(&location, H5WRITEATT_NAME, "location")?;
    let attr_name = string_scalar(&attribute, H5WRITEATT_NAME, "attribute")?;
    if attr_name.trim().is_empty() {
        return Err(hdf5_error(
            H5WRITEATT_NAME,
            &ERR_ARGUMENT,
            "h5writeatt: attribute must not be empty",
        ));
    }
    preflight_writable_data(&data, H5WRITEATT_NAME)?;
    let path = path::capture(path, NativePathIntent::ReadWriteCreate, H5WRITEATT_NAME)?;
    let gather = gather_if_needed_async(&data);
    let data = match runtime.as_ref() {
        Some(runtime) => runtime.scope(gather).await,
        None => gather.await,
    }
    .map_err(|err| map_control_flow(H5WRITEATT_NAME, err))?;
    path::check(&path, H5WRITEATT_NAME)?;
    imp::write_attribute(H5WRITEATT_NAME, &path, location_path, attr_name, data)?;
    path::check(&path, H5WRITEATT_NAME)?;
    Ok(Value::output_list(Vec::new()))
}

#[runtime_builtin(
    name = "h5info",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn h5info_builtin(
    filename: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    admit_output_count(H5INFO_NAME, &H5INFO_DESCRIPTOR)?;
    let path = resolve_path(&filename, H5INFO_NAME)?;
    let location = optional_location(rest.view(), H5INFO_NAME)?;
    let path = path::capture(path, NativePathIntent::Read, H5INFO_NAME)?;
    path::check(&path, H5INFO_NAME)?;
    let result = imp::info(H5INFO_NAME, &path, location)?;
    path::check(&path, H5INFO_NAME)?;
    Ok(result)
}

#[runtime_builtin(
    name = "hdf5info",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn hdf5info_builtin(
    filename: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    admit_output_count(HDF5INFO_NAME, &HDF5INFO_DESCRIPTOR)?;
    let path = resolve_path(&filename, HDF5INFO_NAME)?;
    let location = optional_location(rest.view(), HDF5INFO_NAME)?;
    let path = path::capture(path, NativePathIntent::Read, HDF5INFO_NAME)?;
    path::check(&path, HDF5INFO_NAME)?;
    let result = imp::info(HDF5INFO_NAME, &path, location)?;
    path::check(&path, HDF5INFO_NAME)?;
    Ok(result)
}

#[runtime_builtin(
    name = "h5disp",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::hdf5"
)]
async fn h5disp_builtin(
    filename: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    admit_output_count(H5DISP_NAME, &H5DISP_DESCRIPTOR)?;
    if crate::output_count::current_output_count().is_some_and(|requested| requested > 0) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &H5DISP_CAPTURE_EXTENSION,
            H5DISP_NAME,
        )?;
    }
    let path = resolve_path(&filename, H5DISP_NAME)?;
    let controls = h5disp_controls::parse(rest.view())?;
    let path = path::capture(path, NativePathIntent::Read, H5DISP_NAME)?;
    path::check(&path, H5DISP_NAME)?;
    let result = imp::disp(H5DISP_NAME, &path, controls)?;
    path::check(&path, H5DISP_NAME)?;
    if crate::output_count::current_output_count() == Some(0) {
        record_console_line(ConsoleStream::Stdout, result)?;
        Ok(Value::output_list(Vec::new()))
    } else {
        Ok(Value::string(result))
    }
}

fn optional_location(
    rest: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<String> {
    match rest.len() {
        0 => Ok("/".to_string()),
        1 => hdf5_path(&rest[0], builtin, "location"),
        _ => Err(hdf5_error(
            builtin,
            &ERR_ARGUMENT,
            format!("{builtin}: expected at most one location argument"),
        )),
    }
}

mod geometry;
use geometry::*;

#[cfg(target_arch = "wasm32")]
mod imp {
    use super::*;

    pub fn read_dataset(
        builtin: &'static str,
        _path: &Hdf5PathLease,
        _dataset_path: String,
        _controls: SelectionControls,
    ) -> BuiltinResult<Value> {
        Err(hdf5_error(builtin, &ERR_WASM, ERR_WASM.message))
    }

    pub fn write_dataset(
        builtin: &'static str,
        _path: &Hdf5PathLease,
        _dataset_path: String,
        _data: Value,
        _controls: SelectionControls,
    ) -> BuiltinResult<()> {
        Err(hdf5_error(builtin, &ERR_WASM, ERR_WASM.message))
    }

    pub fn write_attribute(
        builtin: &'static str,
        _path: &Hdf5PathLease,
        _location: String,
        _attribute: String,
        _data: Value,
    ) -> BuiltinResult<()> {
        Err(hdf5_error(builtin, &ERR_WASM, ERR_WASM.message))
    }

    pub fn info(
        builtin: &'static str,
        _path: &Hdf5PathLease,
        _location: String,
    ) -> BuiltinResult<Value> {
        Err(hdf5_error(builtin, &ERR_WASM, ERR_WASM.message))
    }

    pub fn disp(
        builtin: &'static str,
        _path: &Hdf5PathLease,
        _controls: H5DispControls,
    ) -> BuiltinResult<String> {
        Err(hdf5_error(builtin, &ERR_WASM, ERR_WASM.message))
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
use native as imp;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod h5disp_tests;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod path_tests;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod alias_tests;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod failure_tests;
#[cfg(test)]
mod admission_tests;
#[cfg(test)]
mod test_arguments;
