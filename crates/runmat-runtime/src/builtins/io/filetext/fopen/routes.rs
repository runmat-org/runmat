//! Open, query, and enumeration routes through one scoped FID registry.

use super::output::{ListOutputs, QueryOutputs};
use super::*;
use crate::builtins::io::filetext::registry::RegisteredFile;
use runmat_filesystem::OpenOptions;
use std::sync::{Arc, Mutex as StdMutex};

pub(super) async fn handle_open(
    runtime: Option<&crate::context::RuntimeContext>,
    path_value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    filesystem: Option<std::sync::Arc<registry::FileFilesystem>>,
) -> BuiltinResult<FopenEval> {
    if rest.len() > 3 {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
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

    let filesystem = filesystem.expect("an admitted filename captured its filesystem");
    if let Err(error) = filesystem.check() {
        return Ok(FopenEval::open(OpenOutputs::failure(error.to_string())));
    }
    match super::super::cooperation::await_io(
        runtime,
        BUILTIN_NAME,
        options.open_async_on(filesystem.provider.as_ref(), &filesystem.path),
    )
    .await?
    {
        Ok(file) => {
            if let Err(error) = filesystem.check() {
                return Ok(FopenEval::open(OpenOutputs::failure(error.to_string())));
            }
            let handle = Arc::new(StdMutex::new(Some(file)));
            let registered = RegisteredFile {
                path: path.clone(),
                permission: permission.canonical.clone(),
                access: permission.access(),
                machinefmt: machinefmt.clone(),
                encoding: encoding.clone(),
                handle,
            };
            let fid = registry::register_file_in(registered, runtime, Some(filesystem));
            Ok(FopenEval::open(OpenOutputs::success(
                fid, machinefmt, encoding,
            )))
        }
        Err(err) => Ok(FopenEval::open(OpenOutputs::failure(err.to_string()))),
    }
}

pub(super) fn handle_query(
    runtime: Option<&crate::context::RuntimeContext>,
    fid_value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<FopenEval> {
    if !rest.is_empty() {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
    }
    let fid = parse_fid(fid_value)?;
    let outputs = match registry::info_for_in(fid, runtime) {
        Some(info) => QueryOutputs::from_info(info),
        None => QueryOutputs::not_found(),
    };
    Ok(FopenEval::query(outputs))
}

pub(super) fn handle_all(
    runtime: Option<&crate::context::RuntimeContext>,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<FopenEval> {
    if rest.len() > 1 {
        return Err(fopen_error_with_detail(
            &FOPEN_ERROR_INVALID_INPUT,
            "too many input arguments",
        ));
    }
    let machinefmt_filter = if let Some(value) = rest.first() {
        Some(parse_machinefmt(Some(value))?)
    } else {
        None
    };
    let mut infos = registry::list_infos_in(runtime);
    if let Some(filter) = &machinefmt_filter {
        infos.retain(|info| info.machinefmt.eq_ignore_ascii_case(filter));
    }
    let outputs = ListOutputs::from_infos(infos)?;
    Ok(FopenEval::list(outputs))
}
