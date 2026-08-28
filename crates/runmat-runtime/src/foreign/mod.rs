mod callback;
mod conversion;
mod error;
mod extension;
mod handle;
#[cfg(not(target_arch = "wasm32"))]
mod java;
mod manifest;
#[cfg(not(target_arch = "wasm32"))]
mod mex;
#[cfg(not(target_arch = "wasm32"))]
mod native_ffi;
mod policy;
#[cfg(not(target_arch = "wasm32"))]
mod python;
mod runtime;
mod telemetry;

pub use callback::*;
pub use conversion::*;
pub use error::*;
pub use extension::*;
pub use handle::*;
#[cfg(not(target_arch = "wasm32"))]
pub use java::*;
pub use manifest::*;
#[cfg(not(target_arch = "wasm32"))]
pub use mex::*;
#[cfg(not(target_arch = "wasm32"))]
pub use native_ffi::*;
pub use policy::*;
#[cfg(not(target_arch = "wasm32"))]
pub use python::*;
pub use runtime::*;
pub use telemetry::*;

#[cfg(not(target_arch = "wasm32"))]
pub async fn run_extension_host() -> Result<(), String> {
    match std::env::var(native_ffi::NATIVE_FFI_HOST_KIND_ENV)
        .ok()
        .as_deref()
    {
        Some(native_ffi::NATIVE_FFI_HOST_KIND) => native_ffi::run_native_ffi_extension_host(),
        Some(python::PYTHON_HOST_KIND) => python::run_python_extension_host(),
        Some(kind) => Err(format!("unknown extension host kind `{kind}`")),
        None => mex::run_mex_extension_host().await,
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

/// Read one Python-style item through the resource's owning adapter. Source
/// indices remain one-based until the adapter performs its language-specific
/// conversion, so the VM does not acquire Python indexing policy.
pub async fn index_foreign_resource(
    reference: runmat_value::ForeignRef,
    indices: Vec<runmat_value::Value>,
) -> Result<runmat_value::Value, crate::RuntimeError> {
    invoke_foreign_member(
        reference.clone(),
        "get_item",
        vec![
            runmat_value::Value::Foreign(reference),
            runmat_value::Value::OutputList(indices),
        ],
    )
    .await
}

/// Assign one Python-style item and return the same foreign resource identity,
/// matching handle-object assignment semantics.
pub async fn assign_foreign_resource_index(
    reference: runmat_value::ForeignRef,
    indices: Vec<runmat_value::Value>,
    value: runmat_value::Value,
) -> Result<runmat_value::Value, crate::RuntimeError> {
    invoke_foreign_member(
        reference.clone(),
        "set_item",
        vec![
            runmat_value::Value::Foreign(reference),
            runmat_value::Value::OutputList(indices),
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

/// Resolve a qualified constructor or static call through the session Java
/// adapter after source and package resolution has classified it as an external
/// name. Native-library namespaces retain their own adapter routing.
pub async fn try_invoke_java(
    context: crate::context::RuntimeContext,
    name: &str,
    arguments: Vec<runmat_value::Value>,
    requested_outputs: usize,
) -> Option<Result<runmat_value::Value, crate::RuntimeError>> {
    if !is_java_qualified_candidate(name) {
        return None;
    }
    let Some(service) = context.service_ports().foreign().cloned() else {
        return (name.starts_with("java.") || name.starts_with("javax.")).then(|| {
            Err(foreign_error(
                ForeignErrorKind::UnsupportedOnWasm,
                format!("Java call `{name}` is unavailable on this host"),
            ))
        });
    };
    if !name.starts_with("java.") && !name.starts_with("javax.") {
        let configured = context
            .scope(service.invoke(
                context.clone(),
                crate::context::ForeignCall {
                    adapter: "java".into(),
                    symbol: "has_classpath".into(),
                    arguments: Vec::new(),
                    requested_outputs: 1,
                },
            ))
            .await;
        if !matches!(configured, Ok(runmat_value::Value::Bool(true))) {
            return None;
        }
    }
    let mut call_arguments = Vec::with_capacity(arguments.len() + 1);
    call_arguments.push(runmat_value::Value::String(name.into()));
    call_arguments.extend(arguments);
    let result = context
        .scope(service.invoke(
            context.clone(),
            crate::context::ForeignCall {
                adapter: "java".into(),
                symbol: "invoke_qualified".into(),
                arguments: call_arguments,
                requested_outputs,
            },
        ))
        .await;
    if result
        .as_ref()
        .is_err_and(|error| error.identifier() == Some("RunMat:Java:ClassNotFound"))
    {
        None
    } else {
        Some(result)
    }
}

/// Resolve the explicit `py.<module>.<callable>` namespace through the
/// session's Python adapter. No other dotted namespace is claimed here.
pub async fn try_invoke_python(
    context: crate::context::RuntimeContext,
    name: &str,
    arguments: Vec<runmat_value::Value>,
    requested_outputs: usize,
) -> Option<Result<runmat_value::Value, crate::RuntimeError>> {
    if !name.starts_with("py.") || name.len() <= 3 {
        return None;
    }
    let Some(service) = context.service_ports().foreign().cloned() else {
        return Some(Err(foreign_error(
            ForeignErrorKind::UnsupportedOnWasm,
            format!("Python call `{name}` is unavailable on this host"),
        )));
    };
    let mut call_arguments = Vec::with_capacity(arguments.len() + 1);
    call_arguments.push(runmat_value::Value::String(name.into()));
    call_arguments.extend(arguments);
    Some(
        context
            .scope(service.invoke(
                context.clone(),
                crate::context::ForeignCall {
                    adapter: "python".into(),
                    symbol: "invoke_qualified".into(),
                    arguments: call_arguments,
                    requested_outputs,
                },
            ))
            .await,
    )
}

fn is_java_qualified_candidate(name: &str) -> bool {
    if !name.contains('.') || name.starts_with("clib.") || name.starts_with("clibgen.") {
        return false;
    }
    if name
        .rsplit_once('.')
        .is_some_and(|(class_name, _)| crate::class_registry::get_class(class_name).is_some())
    {
        return false;
    }
    true
}

#[cfg(test)]
mod java_qualified_tests {
    use super::is_java_qualified_candidate;

    #[test]
    fn accepts_custom_java_packages_without_claiming_native_namespaces() {
        assert!(is_java_qualified_candidate("fixture.dynamic.Value.create"));
        assert!(is_java_qualified_candidate("fixture.dynamic.value.create"));
        assert!(is_java_qualified_candidate("java.util.ArrayList"));
        assert!(is_java_qualified_candidate("unresolvedObject.verifyEqual"));
        assert!(!is_java_qualified_candidate("plain_name"));
        assert!(!is_java_qualified_candidate("clib.fixture.call"));
        assert!(!is_java_qualified_candidate("clibgen.buildInterface"));
    }
}
