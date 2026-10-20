//! Public userpath controls and immediate ordered search-path publication.

use runmat_builtins::{
    UserPathControl, UserPathForm, RESTOREDEFAULTPATH_RETURN_EXTENSION,
    USERPATH_PARTIAL_PATH_EXTENSION,
};
use runmat_value::Value;

use crate::builtins::common::path_state::CapturedUserPathAuthority;
use crate::context::{
    PreferenceServiceError, RuntimeContext, UserPathMutation, UserPathServiceError, UserPathSetting,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

enum Request {
    Read,
    Change(UserPathMutation),
}

pub(super) async fn run(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let request = decode(args)?;
    let runtime = runtime.ok_or_else(unavailable)?;
    let service = runtime
        .service_ports()
        .user_path()
        .cloned()
        .ok_or_else(unavailable)?;
    let search = runtime.search_path().cloned().ok_or_else(unavailable)?;
    let authority = CapturedUserPathAuthority::capture().map_err(authority_error)?;
    // A known invalid path must fail before persistent storage is touched.
    if let Request::Change(UserPathMutation::Set(ref text)) = request {
        authority.resolve_custom(text).map_err(path_error)?;
    }
    let resolution_runtime = runtime.clone();
    let changed = matches!(&request, Request::Change(_));
    let explicit_reset = matches!(&request, Request::Change(UserPathMutation::Reset));
    let snapshot = match request {
        Request::Read => service.read(runtime).await.map_err(service_error)?,
        Request::Change(mutation) => {
            // This is the last local authority check before the application
            // store authorizes its independent durable transaction.
            authority.check_current().map_err(authority_error)?;
            match service.commit(runtime, mutation).await {
                Ok(snapshot) => snapshot,
                Err(source) => {
                    // A definitive rejection leaves the old path valid; only
                    // an indeterminate authorized commit can invalidate it.
                    if matches!(
                        &source,
                        UserPathServiceError::Storage(source)
                            if matches!(source.as_ref(), PreferenceServiceError::CommitIndeterminate(_))
                    ) {
                        search.mark_user_authority_stale();
                    }
                    return Err(service_error(source));
                }
            }
        }
    };
    if let Err(source) = authority.check_current() {
        if changed {
            search.mark_user_authority_stale();
            return Err(committed_but_local_path_unavailable(
                snapshot.revision,
                source,
            ));
        }
        return Err(authority_error(source));
    }
    if explicit_reset && !matches!(&snapshot.setting, UserPathSetting::Default) {
        search.mark_user_authority_stale();
        return Err(build_runtime_error(
            "userpath: application settings service returned a non-default value for reset",
        )
        .with_builtin("userpath")
        .with_identifier("RunMat:userpath:Storage")
        .build());
    }
    let mut cooperation = resolution_runtime.cpu_cooperation();
    if let Err(source) = cooperation.checkpoint(0).await {
        if changed {
            search.mark_user_authority_stale();
            return Err(committed_but_local_cancellation(snapshot.revision, source));
        }
        return Err(source.into_runtime_error());
    }
    let resolution = async {
        if explicit_reset {
            authority.resolve_after_reset().await
        } else {
            authority.resolve(&snapshot.setting).await
        }
    };
    let resolved = match cooperation.await_cancellable(resolution).await {
        Ok(Ok(resolved)) => resolved,
        Ok(Err(source)) if changed => {
            search.mark_user_authority_stale();
            return Err(committed_but_local_path_unavailable(
                snapshot.revision,
                source,
            ));
        }
        Ok(Err(source)) => return Err(path_error(source)),
        Err(source) if changed => {
            search.mark_user_authority_stale();
            return Err(committed_but_local_cancellation(snapshot.revision, source));
        }
        Err(source) => return Err(source.into_runtime_error()),
    };
    if let Err(source) = authority.check_current() {
        if changed {
            search.mark_user_authority_stale();
            return Err(committed_but_local_path_unavailable(
                snapshot.revision,
                source,
            ));
        }
        return Err(authority_error(source));
    }
    if search.user_authority_current() {
        search.replace_user_if_newer(snapshot.revision, resolved.search_segment);
    } else if !search.rehydrate_user(snapshot.revision, resolved.search_segment) {
        return Err(build_runtime_error(
            "userpath: application setting read returned an older revision",
        )
        .with_builtin("userpath")
        .with_identifier("RunMat:userpath:StaleRevision")
        .build());
    }
    Ok(super::char_value(&resolved.presentation))
}

pub(super) fn restore(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(
            build_runtime_error("restoredefaultpath: too many output arguments")
                .with_builtin("restoredefaultpath")
                .with_identifier("RunMat:restoredefaultpath:TooManyOutputs")
                .build(),
        );
    }
    if !args.is_empty() {
        return Err(
            build_runtime_error("restoredefaultpath: too many input arguments")
                .with_builtin("restoredefaultpath")
                .with_identifier("RunMat:restoredefaultpath:InvalidCall")
                .build(),
        );
    }
    if crate::output_count::current_output_count() == Some(1) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RESTOREDEFAULTPATH_RETURN_EXTENSION,
            "restoredefaultpath",
        )?;
    }
    let runtime = runtime.ok_or_else(restore_unavailable)?;
    let search = runtime.search_path().ok_or_else(restore_unavailable)?;
    if !search.user_authority_current() {
        return Err(build_runtime_error(
            "restoredefaultpath: user-path setting must be rehydrated before source resolution",
        )
        .with_builtin("restoredefaultpath")
        .with_identifier("RunMat:restoredefaultpath:StaleUserPath")
        .build());
    }
    search.restore_factory();
    Ok(super::char_value(&search.current_string()))
}

fn decode(args: crate::call::BuiltinArguments) -> BuiltinResult<Request> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(build_runtime_error("userpath: too many output arguments")
            .with_builtin("userpath")
            .with_identifier("RunMat:userpath:TooManyOutputs")
            .build());
    }
    let mut args = args.into_iter();
    let Some(value) = args.next() else {
        return Ok(Request::Read);
    };
    if args.next().is_some() {
        return Err(build_runtime_error("userpath: too many input arguments")
            .with_builtin("userpath")
            .with_identifier("RunMat:userpath:InvalidCall")
            .build());
    }
    let text =
        crate::builtins::common::validation::value_text_scalar_string(&value).ok_or_else(|| {
            build_runtime_error("userpath: path or command must be host text")
                .with_builtin("userpath")
                .with_identifier("RunMat:userpath:InvalidPath")
                .build()
        })?;
    let control = UserPathControl::parse(&text).map_err(|_| {
        build_runtime_error("userpath: invalid path or command")
            .with_builtin("userpath")
            .with_identifier("RunMat:userpath:InvalidPath")
            .build()
    })?;
    let mutation = match control {
        UserPathControl::Reset => UserPathMutation::Reset,
        UserPathControl::Clear => UserPathMutation::Clear,
        UserPathControl::Custom {
            form: UserPathForm::Absolute,
            ..
        } => UserPathMutation::Set(text),
        UserPathControl::Custom {
            form: UserPathForm::Partial,
            ..
        } => {
            crate::compatibility::ensure_builtin_feature_enabled(
                &USERPATH_PARTIAL_PATH_EXTENSION,
                "userpath",
            )?;
            UserPathMutation::Set(text)
        }
    };
    Ok(Request::Change(mutation))
}

fn unavailable() -> RuntimeError {
    build_runtime_error("userpath: application settings service is unavailable")
        .with_builtin("userpath")
        .with_identifier("RunMat:RuntimeContext:CapabilityUnavailable")
        .build()
}

fn restore_unavailable() -> RuntimeError {
    build_runtime_error("restoredefaultpath: ordered search-path service is unavailable")
        .with_builtin("restoredefaultpath")
        .with_identifier("RunMat:RuntimeContext:CapabilityUnavailable")
        .build()
}

fn authority_error(source: std::io::Error) -> RuntimeError {
    build_runtime_error(format!("userpath: filesystem authority failed: {source}"))
        .with_builtin("userpath")
        .with_identifier("RunMat:userpath:FilesystemAuthority")
        .with_source(source)
        .build()
}

fn path_error(source: std::io::Error) -> RuntimeError {
    if source.kind() != std::io::ErrorKind::InvalidInput {
        return authority_error(source);
    }
    build_runtime_error(format!("userpath: invalid path: {source}"))
        .with_builtin("userpath")
        .with_identifier("RunMat:userpath:InvalidPath")
        .with_source(source)
        .build()
}

fn committed_but_local_path_unavailable(revision: u64, source: std::io::Error) -> RuntimeError {
    build_runtime_error(format!(
        "userpath: application setting committed at revision {revision}, but local path authority is unavailable: {source}"
    ))
    .with_builtin("userpath")
    .with_identifier("RunMat:userpath:CommittedButLocalPathUnavailable")
    .with_source(source)
    .build()
}

fn committed_but_local_cancellation(
    revision: u64,
    source: crate::execution::ExecutionServiceError,
) -> RuntimeError {
    build_runtime_error(format!(
        "userpath: application setting committed at revision {revision}, but local publication was cancelled"
    ))
    .with_builtin("userpath")
    .with_identifier("RunMat:userpath:CommittedButLocalPathUnavailable")
    .with_source(source)
    .build()
}

fn service_error(source: UserPathServiceError) -> RuntimeError {
    match source {
        UserPathServiceError::Execution(source) => source.into_runtime_error(),
        source => build_runtime_error(format!("userpath: application setting failed: {source}"))
            .with_builtin("userpath")
            .with_identifier("RunMat:userpath:Storage")
            .with_source(source)
            .build(),
    }
}

#[cfg(test)]
#[path = "user_path/tests.rs"]
mod tests;
