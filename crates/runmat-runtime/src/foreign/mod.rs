mod callback;
mod conversion;
mod error;
mod extension;
mod handle;
#[cfg(not(target_arch = "wasm32"))]
mod isolation_input;
#[cfg(not(target_arch = "wasm32"))]
mod java;
mod manifest;
#[cfg(not(target_arch = "wasm32"))]
mod native_lane_profiling;
#[cfg(not(target_arch = "wasm32"))]
mod mex;
#[cfg(not(target_arch = "wasm32"))]
mod native_ffi;
mod policy;
#[cfg(not(target_arch = "wasm32"))]
mod project_retirement;
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
pub(crate) use mex::MexCallbackAncestry;
#[cfg(not(target_arch = "wasm32"))]
pub use mex::*;
#[cfg(not(target_arch = "wasm32"))]
pub use native_ffi::*;
pub use policy::*;
#[cfg(not(target_arch = "wasm32"))]
pub use python::*;
pub use runtime::*;
pub use telemetry::*;

/// Foreign routing either returns the untouched argument owner for semantic
/// fallback, or has selected a terminal adapter operation. An empty selected
/// result is not a routing decline.
/// `Continue` transfers the original arguments to the next resolver;
/// `Break` stops resolution with the selected operation's result.
pub type ForeignAttempt = std::ops::ControlFlow<
    Result<crate::execution::RootedValueSequence, crate::RuntimeError>,
    crate::call::RootedFunctionArguments,
>;

pub(crate) fn single_output_sequence(
    value: runmat_value::Value,
) -> Result<runmat_value::ValueSequence, crate::RuntimeError> {
    runmat_value::ValueSequence::single(value).map_err(crate::sequence::sequence_error_to_runtime)
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn run_extension_host() -> Result<(), String> {
    match std::env::var(native_ffi::NATIVE_FFI_HOST_KIND_ENV)
        .ok()
        .as_deref()
    {
        Some(native_ffi::NATIVE_FFI_HOST_KIND) => native_ffi::run_native_ffi_extension_host().await,
        Some(python::PYTHON_HOST_KIND) => python::run_python_extension_host().await,
        Some(kind) => Err(format!("unknown extension host kind `{kind}`")),
        None => mex::run_mex_extension_host().await,
    }
}

/// Read a member from a foreign resource through the adapter that owns its
/// type family.
pub async fn load_foreign_member(
    reference: runmat_value::ForeignRef,
    member: String,
) -> Result<crate::execution::RootedValue, crate::RuntimeError> {
    invoke_foreign_member(
        reference.clone(),
        "get_member",
        vec![
            runmat_value::Value::Foreign(reference),
            runmat_value::Value::String(member.into()),
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
) -> Result<crate::execution::RootedValue, crate::RuntimeError> {
    invoke_foreign_member(
        reference.clone(),
        "set_member",
        vec![
            runmat_value::Value::Foreign(reference),
            runmat_value::Value::String(member.into()),
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
) -> Result<crate::execution::RootedValue, crate::RuntimeError> {
    let mut arguments = Vec::with_capacity(indices.len() + 1);
    arguments.push(runmat_value::Value::Foreign(reference.clone()));
    arguments.extend(indices);
    invoke_foreign_member(reference.clone(), "get_item", arguments).await
}

/// Assign one Python-style item and return the same foreign resource identity,
/// matching handle-object assignment semantics.
pub async fn assign_foreign_resource_index(
    reference: runmat_value::ForeignRef,
    indices: Vec<runmat_value::Value>,
    value: runmat_value::Value,
) -> Result<crate::execution::RootedValue, crate::RuntimeError> {
    let mut arguments = Vec::with_capacity(indices.len() + 2);
    arguments.push(runmat_value::Value::Foreign(reference.clone()));
    arguments.extend(indices);
    arguments.push(value);
    invoke_foreign_member(reference.clone(), "set_item", arguments).await
}

/// Invoke a method on a foreign resource through its owning adapter. Object
/// dispatch remains backend-neutral; native libraries, JVMs, and later
/// adapters define their own method sets behind this operation.
pub async fn invoke_foreign_method(
    reference: runmat_value::ForeignRef,
    method: String,
    arguments: crate::execution::RootedValues,
    requested_outputs: usize,
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
    let context = crate::context::legacy::active().ok_or_else(|| {
        foreign_error(
            ForeignErrorKind::HostUnavailable,
            "foreign member access has no active runtime context",
        )
    })?;
    let arguments = crate::call::RootedFunctionArguments::ordinary(arguments)
        .prepend_routing_in_context(runmat_value::Value::String(method.into()), &context)?
        .prepend_routing_in_context(runmat_value::Value::Foreign(reference.clone()), &context)?;
    invoke_foreign_rooted(reference, "invoke_member", arguments, requested_outputs).await
}

async fn invoke_foreign_member(
    reference: runmat_value::ForeignRef,
    operation: &str,
    arguments: Vec<runmat_value::Value>,
) -> Result<crate::execution::RootedValue, crate::RuntimeError> {
    let sequence = invoke_foreign(reference, operation, arguments, 1).await?;
    require_foreign_single(sequence)
}

async fn invoke_foreign(
    reference: runmat_value::ForeignRef,
    operation: &str,
    arguments: Vec<runmat_value::Value>,
    requested_outputs: usize,
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
    let context = crate::context::legacy::active().ok_or_else(|| {
        foreign_error(
            ForeignErrorKind::HostUnavailable,
            "foreign member access has no active runtime context",
        )
    })?;
    let arguments = crate::call::RootedFunctionArguments::ordinary(
        crate::execution::RootedValues::in_context(arguments, &context)?,
    );
    invoke_foreign_rooted(reference, operation, arguments, requested_outputs).await
}

async fn invoke_foreign_rooted(
    reference: runmat_value::ForeignRef,
    operation: &str,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
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
                adapter: reference.type_identity.family.clone(),
                symbol: operation.into(),
                arguments,
                requested_outputs,
            },
        ))
        .await
        .into_invoked_result()
}

fn require_foreign_single(
    sequence: crate::execution::RootedValueSequence,
) -> Result<crate::execution::RootedValue, crate::RuntimeError> {
    use crate::sequence::ResolveValueSequence;
    sequence.try_map_owned(
        |sequence| {
            let values = sequence.resolve(
                runmat_types::SequenceUse::RequireSingle,
                crate::sequence::SequenceResolutionContext::default(),
            )?;
            Ok(crate::sequence::into_first_output(values)?
                .expect("RequireSingle returned one foreign value"))
        },
        |error| error.into_runtime_error(),
    )
}

/// Resolve the modern `clib.<interface>.<function>` namespace through the same
/// session adapter used by the legacy shared-library builtins.
pub async fn try_invoke_clib(
    context: crate::context::RuntimeContext,
    name: &str,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
) -> ForeignAttempt {
    let segments = name.split('.').collect::<Vec<_>>();
    if segments.len() < 3 || segments[0] != "clib" {
        return ForeignAttempt::Continue(arguments);
    }
    if segments.iter().any(|segment| segment.trim().is_empty()) {
        return ForeignAttempt::Break(Err(foreign_error(
            ForeignErrorKind::InvalidCall,
            "clib namespace contains an empty segment",
        )));
    }
    let Some(service) = context.service_ports().foreign().cloned() else {
        return ForeignAttempt::Break(Err(foreign_error(
            ForeignErrorKind::UnsupportedOnWasm,
            format!(
                "native interface `{}` is unavailable on this host",
                segments[1]
            ),
        )));
    };
    let call_arguments = arguments
        .prepend_routing_in_context(
            runmat_value::Value::String(segments[2..].join(".").into()),
            &context,
        )
        .and_then(|arguments| {
            arguments.prepend_routing_in_context(
                runmat_value::Value::String(segments[1].into()),
                &context,
            )
        });
    let call_arguments = match call_arguments {
        Ok(arguments) => arguments,
        Err(error) => return ForeignAttempt::Break(Err(error)),
    };
    ForeignAttempt::Break(
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
            .await
            .into_invoked_result(),
    )
}

/// Route supported interface-generation entrypoints into the native adapter.
pub async fn try_invoke_clibgen(
    context: crate::context::RuntimeContext,
    name: &str,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
) -> ForeignAttempt {
    if name != "clibgen.buildInterface" {
        return ForeignAttempt::Continue(arguments);
    }
    let Some(service) = context.service_ports().foreign().cloned() else {
        return ForeignAttempt::Break(Err(foreign_error(
            ForeignErrorKind::UnsupportedOnWasm,
            "native interface generation is unavailable on this host",
        )));
    };
    ForeignAttempt::Break(
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
            .await
            .into_invoked_result(),
    )
}

/// Resolve a qualified constructor or static call through the session Java
/// adapter after source and package resolution has classified it as an external
/// name. Native-library namespaces retain their own adapter routing.
pub async fn try_invoke_java(
    context: crate::context::RuntimeContext,
    name: &str,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
) -> ForeignAttempt {
    if !is_java_qualified_candidate(name) {
        return ForeignAttempt::Continue(arguments);
    }
    let Some(service) = context.service_ports().foreign().cloned() else {
        return if name.starts_with("java.") || name.starts_with("javax.") {
            ForeignAttempt::Break(Err(foreign_error(
                ForeignErrorKind::UnsupportedOnWasm,
                format!("Java call `{name}` is unavailable on this host"),
            )))
        } else {
            ForeignAttempt::Continue(arguments)
        };
    };
    if !name.starts_with("java.") && !name.starts_with("javax.") {
        let configured = context
            .scope(service.invoke(
                context.clone(),
                crate::context::ForeignCall {
                    adapter: "java".into(),
                    symbol: "has_classpath".into(),
                    arguments: match crate::execution::RootedValues::in_context(
                        Vec::new(),
                        &context,
                    ) {
                        Ok(arguments) => crate::call::RootedFunctionArguments::ordinary(arguments),
                        Err(error) => return ForeignAttempt::Break(Err(error)),
                    },
                    requested_outputs: 1,
                },
            ))
            .await;
        if !matches!(
            configured.into_invoked_result().and_then(require_foreign_single),
            Ok(value) if matches!(value.value(), runmat_value::Value::Bool(true))
        ) {
            return ForeignAttempt::Continue(arguments);
        }
    }
    let call_arguments = match arguments
        .prepend_routing_in_context(runmat_value::Value::String(name.into()), &context)
    {
        Ok(arguments) => arguments,
        Err(error) => return ForeignAttempt::Break(Err(error)),
    };
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
    match result.into_owner_return() {
        Ok(result) => ForeignAttempt::Break(result),
        Err(call) => match call.arguments.remove_first_routing_in_context(&context) {
            Ok(arguments) => ForeignAttempt::Continue(arguments),
            Err(error) => ForeignAttempt::Break(Err(error)),
        },
    }
}

/// Resolve the explicit `py.<module>.<callable>` namespace through the
/// session's Python adapter. No other dotted namespace is claimed here.
pub async fn try_invoke_python(
    context: crate::context::RuntimeContext,
    name: &str,
    arguments: crate::call::RootedFunctionArguments,
    requested_outputs: usize,
) -> ForeignAttempt {
    if !name.starts_with("py.") || name.len() <= 3 {
        return ForeignAttempt::Continue(arguments);
    }
    let Some(service) = context.service_ports().foreign().cloned() else {
        return ForeignAttempt::Break(Err(foreign_error(
            ForeignErrorKind::UnsupportedOnWasm,
            format!("Python call `{name}` is unavailable on this host"),
        )));
    };
    let call_arguments = match arguments
        .prepend_routing_in_context(runmat_value::Value::String(name.into()), &context)
    {
        Ok(arguments) => arguments,
        Err(error) => return ForeignAttempt::Break(Err(error)),
    };
    ForeignAttempt::Break(
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
            .await
            .into_invoked_result(),
    )
}

fn is_java_qualified_candidate(name: &str) -> bool {
    if !name.contains('.') || name.starts_with("clib.") || name.starts_with("clibgen.") {
        return false;
    }
    if name.rsplit_once('.').is_some_and(|(class_name, _)| {
        runmat_types::ClassIdentity::new(class_name)
            .ok()
            .and_then(|identity| crate::class_registry::get_class(&identity))
            .is_some()
    }) {
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
