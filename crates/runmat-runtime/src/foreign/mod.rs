mod callback;
mod conversion;
mod error;
mod handle;
mod manifest;
#[cfg(not(target_arch = "wasm32"))]
mod mex;
#[cfg(not(target_arch = "wasm32"))]
mod native_ffi;
mod policy;
mod runtime;
mod telemetry;

pub use callback::*;
pub use conversion::*;
pub use error::*;
pub use handle::*;
pub use manifest::*;
#[cfg(not(target_arch = "wasm32"))]
pub use mex::*;
#[cfg(not(target_arch = "wasm32"))]
pub use native_ffi::*;
pub use policy::*;
pub use runtime::*;
pub use telemetry::*;

#[cfg(not(target_arch = "wasm32"))]
pub fn run_extension_host() -> Result<(), String> {
    match std::env::var(native_ffi::NATIVE_FFI_HOST_KIND_ENV)
        .ok()
        .as_deref()
    {
        Some(native_ffi::NATIVE_FFI_HOST_KIND) => native_ffi::run_native_ffi_extension_host(),
        Some(kind) => Err(format!("unknown extension host kind `{kind}`")),
        None => mex::run_mex_extension_host(),
    }
}

/// Read a member from a foreign resource through the adapter that owns its
/// type family.
pub async fn load_foreign_member(
    reference: runmat_value::ForeignRef,
    member: String,
) -> Result<runmat_value::Value, crate::RuntimeError> {
    invoke_foreign_member(
        reference.clone(),
        "get_member",
        vec![
            runmat_value::Value::Foreign(reference),
            runmat_value::Value::String(member),
        ],
    )
    .await
}

/// Write a member on a foreign resource through the adapter that owns its
/// type family. The adapter returns the resource identity so assignment keeps
/// handle semantics.
pub async fn store_foreign_member(
    reference: runmat_value::ForeignRef,
    member: String,
    value: runmat_value::Value,
) -> Result<runmat_value::Value, crate::RuntimeError> {
    invoke_foreign_member(
        reference.clone(),
        "set_member",
        vec![
            runmat_value::Value::Foreign(reference),
            runmat_value::Value::String(member),
            value,
        ],
    )
    .await
}

/// Invoke a method on a foreign resource through its owning adapter. Object
/// dispatch remains backend-neutral; native libraries, JVMs, and later
/// adapters define their own method sets behind this operation.
pub async fn invoke_foreign_method(
    reference: runmat_value::ForeignRef,
    method: String,
    arguments: Vec<runmat_value::Value>,
    requested_outputs: usize,
) -> Result<runmat_value::Value, crate::RuntimeError> {
    let mut call_arguments = Vec::with_capacity(arguments.len() + 2);
    call_arguments.push(runmat_value::Value::Foreign(reference.clone()));
    call_arguments.push(runmat_value::Value::String(method));
    call_arguments.extend(arguments);
    invoke_foreign(
        reference,
        "invoke_member",
        call_arguments,
        requested_outputs,
    )
    .await
}

async fn invoke_foreign_member(
    reference: runmat_value::ForeignRef,
    operation: &str,
    arguments: Vec<runmat_value::Value>,
) -> Result<runmat_value::Value, crate::RuntimeError> {
    invoke_foreign(reference, operation, arguments, 1).await
}

async fn invoke_foreign(
    reference: runmat_value::ForeignRef,
    operation: &str,
    arguments: Vec<runmat_value::Value>,
    requested_outputs: usize,
) -> Result<runmat_value::Value, crate::RuntimeError> {
    let context = crate::context::legacy::active().ok_or_else(|| {
        foreign_error(
            ForeignErrorKind::HostUnavailable,
            "foreign member access has no active runtime context",
        )
    })?;
    let service = context
        .service_ports()
        .require_foreign(operation)
        .map_err(|error| error.into_runtime_error())?
        .clone();
    context
        .scope(service.invoke(
            context.clone(),
            crate::context::ForeignCall {
                adapter: reference.type_identity.family,
                symbol: operation.into(),
                arguments,
                requested_outputs,
            },
        ))
        .await
}

/// Resolve the modern `clib.<interface>.<function>` namespace through the same
/// session adapter used by the legacy shared-library builtins.
pub async fn try_invoke_clib(
    context: crate::context::RuntimeContext,
    name: &str,
    arguments: Vec<runmat_value::Value>,
    requested_outputs: usize,
) -> Option<Result<runmat_value::Value, crate::RuntimeError>> {
    let segments = name.split('.').collect::<Vec<_>>();
    if segments.len() < 3 || segments[0] != "clib" {
        return None;
    }
    if segments.iter().any(|segment| segment.trim().is_empty()) {
        return Some(Err(foreign_error(
            ForeignErrorKind::InvalidCall,
            "clib namespace contains an empty segment",
        )));
    }
    let Some(service) = context.service_ports().foreign().cloned() else {
        return Some(Err(foreign_error(
            ForeignErrorKind::UnsupportedOnWasm,
            format!(
                "native interface `{}` is unavailable on this host",
                segments[1]
            ),
        )));
    };
    let mut call_arguments = Vec::with_capacity(arguments.len() + 2);
    call_arguments.push(runmat_value::Value::String(segments[1].into()));
    call_arguments.push(runmat_value::Value::String(segments[2..].join(".")));
    call_arguments.extend(arguments);
    Some(
        context
            .scope(service.invoke(
                context.clone(),
                crate::context::ForeignCall {
                    adapter: "native-ffi".into(),
                    symbol: "call".into(),
                    arguments: call_arguments,
                    requested_outputs,
                },
            ))
            .await,
    )
}

/// Route supported interface-generation entrypoints into the native adapter.
pub async fn try_invoke_clibgen(
    context: crate::context::RuntimeContext,
    name: &str,
    arguments: Vec<runmat_value::Value>,
    requested_outputs: usize,
) -> Option<Result<runmat_value::Value, crate::RuntimeError>> {
    if name != "clibgen.buildInterface" {
        return None;
    }
    let Some(service) = context.service_ports().foreign().cloned() else {
        return Some(Err(foreign_error(
            ForeignErrorKind::UnsupportedOnWasm,
            "native interface generation is unavailable on this host",
        )));
    };
    Some(
        context
            .scope(service.invoke(
                context.clone(),
                crate::context::ForeignCall {
                    adapter: "native-ffi".into(),
                    symbol: "build_interface".into(),
                    arguments,
                    requested_outputs,
                },
            ))
            .await,
    )
}
