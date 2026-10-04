//! MATLAB-compatible `matfile` compatibility object.
use runmat_types::MemberAccess;

use std::collections::HashMap;
use std::path::PathBuf;

use runmat_builtins::{BuiltinErrorDescriptor, MatFileOptionName};
use runmat_macros::runtime_builtin;
use runmat_value::{ObjectInstance, StringArray, StructValue, Value};

use super::authority::{issue_matfile, resolve_matfile, MatFsPath};
use super::cooperation::MatCodecCooperation;
use super::encoding::{encode_workspace_to_mat_bytes_for_owner, MatEncodingOwner};
use super::load::read_mat_file_for_builtin_on_with_resource_liveness;
use crate::context::host_effect::{
    HostEffectResourceLivenessLease, HostEffectWorkload, MatAssignmentSuffix, MatDecodeOutput,
    MatDecodeSelection, MatEncodeMode, MatEncodeOutput, MatHandleControl,
};
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError,
    OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD,
};

const BUILTIN_NAME: &str = "matfile";
pub(super) use runmat_builtins::MATFILE_CLASS;
const MATFILE_SUBSREF: &str = "matlab.io.MatFile.subsref";
const MATFILE_SUBSASGN: &str = "matlab.io.MatFile.subsasgn";
mod admission;
pub(super) const PROPERTIES_FIELD: &str = "Properties";
pub(super) const SOURCE_FIELD: &str = "Source";
pub(super) const WRITABLE_FIELD: &str = "Writable";

static MATFILE_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(MATFILE_CLASS);

use runmat_builtins::{
    MATFILE_ERROR_INVALID_ARGUMENT, MATFILE_ERROR_INVALID_OPTION, MATFILE_ERROR_IO,
    MATFILE_ERROR_SELECTION, MATFILE_ERROR_UNSUPPORTED,
};

#[runtime_builtin(
    name = "matfile",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_matfile_class_registered();
    let controls = prepare_writable_options(rest)?;
    let path = normalise_path(&filename)?;
    let public_path = path.to_string_lossy().into_owned();
    let source = std::sync::Arc::new(MatFsPath::capture(path).map_err(|err| {
        matfile_error_with_source(
            &MATFILE_ERROR_IO,
            format!("matfile: failed to resolve path: {err}"),
            err,
        )
    })?);
    let control_facts = controls.identity_controls();
    let workload = HostEffectWorkload::MatHandle {
        source: source.path(),
        controls: &control_facts,
    };
    let receipt = super::admission::select(runtime.as_ref(), &source, BUILTIN_NAME, workload)?;
    let result = async {
        let mut cooperation = MatCodecCooperation::new(
            runtime.as_ref(),
            Some(&source),
            BUILTIN_NAME,
            &MATFILE_ERROR_IO,
        )
        .with_resource_liveness(
            receipt
                .as_ref()
                .map(|selected| selected.resource_liveness_lease()),
        );
        let writable = controls.resolve(&mut cooperation).await?;
        cooperation.completed(0).await?;
        let reference = issue_matfile(
            runtime.as_ref(),
            source.clone(),
            public_path.clone(),
            writable,
        )?;
        Ok(Value::Object(matfile_object(
            &public_path,
            writable,
            reference,
        )))
    }
    .await;
    super::admission::finish(receipt, &result);
    result
}

#[runtime_builtin(
    name = "matlab.io.MatFile.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_subsref(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
) -> BuiltinResult<Value> {
    ensure_matfile_class_registered();
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let first = path.steps().first().expect("parsed nonempty MAT path");
    let payload = first.selector_value()?;
    if first.kind() != crate::object::indexing::ObjectIndexKind::Member {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSREF}: only whole-variable member access is supported"),
        ));
    }
    let field = member_name(&payload, MATFILE_SUBSREF)?;
    reject_internal_field(&field)?;
    let object = into_matfile_object(obj.clone(), MATFILE_SUBSREF)?;
    let owner = resolve_matfile(runtime.as_ref(), &object)?;
    let suffix = admission::suffix_digest(&path)
        .map(MatAssignmentSuffix::Exact)
        .unwrap_or(MatAssignmentSuffix::Unavailable);
    let workload = HostEffectWorkload::MatDecode {
        source: owner.source.path(),
        selection: MatDecodeSelection::MemberPath {
            member: &field,
            suffix,
        },
        output: MatDecodeOutput::SingleMember,
    };
    let receipt =
        super::admission::select(runtime.as_ref(), &owner.source, MATFILE_SUBSREF, workload)?;
    let liveness = receipt
        .as_ref()
        .map(|selected| selected.resource_liveness_lease());
    let read_runtime = runtime.clone();
    let result = crate::object::protocol::execute_owned_subsref(
        obj,
        path,
        move |obj, step| async move {
            check_resource_liveness(liveness.as_ref())?;
            let value =
                matfile_read_step(read_runtime.as_ref(), obj, step, liveness.clone()).await?;
            check_resource_liveness(liveness.as_ref())?;
            Ok(value)
        },
        None,
    )
    .await;
    let result = match result {
        Ok(value) => receipt
            .as_ref()
            .map_or(Ok(()), |selected| selected.checkpoint())
            .map(|()| value),
        Err(error) => Err(error),
    };
    super::admission::finish(receipt, &result);
    result
}

async fn matfile_read_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    resource_liveness: Option<std::sync::Arc<dyn HostEffectResourceLivenessLease>>,
) -> BuiltinResult<Value> {
    let payload = step.selector_value()?;
    let object = into_matfile_object(obj, MATFILE_SUBSREF)?;
    if step.kind() != crate::object::indexing::ObjectIndexKind::Member {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSREF}: only whole-variable member access is supported"),
        ));
    }
    let field = member_name(&payload, MATFILE_SUBSREF)?;
    reject_internal_field(&field)?;
    let owner = resolve_matfile(runtime, &object)?;
    if field == PROPERTIES_FIELD {
        return object
            .properties
            .get(PROPERTIES_FIELD)
            .cloned()
            .ok_or_else(|| matfile_error("matfile object is missing Properties"));
    }
    let entries = read_mat_file_for_builtin_on_with_resource_liveness(
        &owner.source,
        BUILTIN_NAME,
        runtime,
        resource_liveness,
    )
    .await?;
    resolve_matfile(runtime, &object)?;
    entries
        .into_iter()
        .find(|(name, _)| name == &field)
        .map(|(_, value)| value)
        .ok_or_else(|| {
            matfile_error_with(
                &MATFILE_ERROR_SELECTION,
                format!(
                    "matfile: variable '{field}' was not found in '{}'",
                    owner.source.path().display()
                ),
            )
        })
}

#[runtime_builtin(
    name = "matlab.io.MatFile.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_subsasgn(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    ensure_matfile_class_registered();
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let first = path.steps().first().expect("parsed nonempty MAT path");
    let payload = first.selector_value()?;
    if first.kind() != crate::object::indexing::ObjectIndexKind::Member {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSASGN}: partial MAT-file assignment is not supported yet"),
        ));
    }
    let field = member_name(&payload, MATFILE_SUBSASGN)?;
    if field == PROPERTIES_FIELD || field == SOURCE_FIELD || field == WRITABLE_FIELD {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSASGN}: '{field}' is read-only"),
        ));
    }
    reject_internal_field(&field)?;
    validate_variable_name(&field, MATFILE_SUBSASGN)?;
    let object = into_matfile_object(obj.clone(), MATFILE_SUBSASGN)?;
    let owner = resolve_matfile(runtime.as_ref(), &object)?;
    if !owner.writable {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            "matfile: object is not writable; construct it with matfile(filename, 'Writable', true)",
        ));
    }
    let suffix = admission::suffix_digest(&path)
        .map(MatAssignmentSuffix::Exact)
        .unwrap_or(MatAssignmentSuffix::Unavailable);
    let workload = HostEffectWorkload::MatEncode {
        destination: owner.source.path(),
        mode: MatEncodeMode::AssignMember {
            member: &field,
            suffix,
        },
        output: MatEncodeOutput::UpdatedObject,
    };
    let receipt =
        super::admission::select(runtime.as_ref(), &owner.source, MATFILE_SUBSASGN, workload)?;
    let liveness = receipt
        .as_ref()
        .map(|selected| selected.resource_liveness_lease());
    let read_runtime = runtime.clone();
    let read_liveness = liveness.clone();
    let result = crate::object::protocol::execute_owned_subsasgn(
        obj,
        path,
        vec![rhs],
        move |obj, step| async move {
            check_resource_liveness(read_liveness.as_ref())?;
            let result =
                matfile_read_step(read_runtime.as_ref(), obj, step, read_liveness.clone()).await?;
            check_resource_liveness(read_liveness.as_ref())?;
            Ok(result)
        },
        move |obj, step, mut values| async move {
            let rhs = values
                .pop()
                .ok_or_else(|| matfile_error("matfile assignment value is missing"))?;
            matfile_write_step(runtime.as_ref(), obj, step, rhs, liveness).await
        },
        None,
    )
    .await;
    let result = match result {
        Ok(value) => receipt
            .as_ref()
            .map_or(Ok(()), |selected| selected.checkpoint())
            .map(|()| value),
        Err(error) => Err(error),
    };
    super::admission::finish(receipt, &result);
    result
}

fn check_resource_liveness(
    lease: Option<&std::sync::Arc<dyn HostEffectResourceLivenessLease>>,
) -> BuiltinResult<()> {
    if lease.is_some_and(|lease| !lease.is_current()) {
        Err(crate::context::host_effect::stale_resource())
    } else {
        Ok(())
    }
}

async fn matfile_write_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
    resource_liveness: Option<std::sync::Arc<dyn HostEffectResourceLivenessLease>>,
) -> BuiltinResult<Value> {
    let payload = step.selector_value()?;
    let object = into_matfile_object(obj, MATFILE_SUBSASGN)?;
    if step.kind() != crate::object::indexing::ObjectIndexKind::Member {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSASGN}: partial MAT-file assignment is not supported yet"),
        ));
    }
    let field = member_name(&payload, MATFILE_SUBSASGN)?;
    if field == PROPERTIES_FIELD || field == SOURCE_FIELD || field == WRITABLE_FIELD {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSASGN}: '{field}' is read-only"),
        ));
    }
    reject_internal_field(&field)?;
    validate_variable_name(&field, MATFILE_SUBSASGN)?;
    let owner = resolve_matfile(runtime, &object)?;
    if !owner.writable {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            "matfile: object is not writable; construct it with matfile(filename, 'Writable', true)",
        ));
    }

    let mut cooperation = MatCodecCooperation::new(
        runtime,
        Some(&owner.source),
        BUILTIN_NAME,
        &MATFILE_ERROR_IO,
    )
    .with_resource_liveness(resource_liveness);
    cooperation.completed(0).await?;

    let mut entries =
        read_existing_entries_for_assignment(runtime, &object, &owner.source, &mut cooperation)
            .await?;
    resolve_matfile(runtime, &object)?;
    let value = cooperation.await_io(gather_if_needed_async(&rhs)).await??;
    resolve_matfile(runtime, &object)?;
    replace_or_append(&mut entries, field, value);
    cooperation.completed(0).await?;
    let bytes = encode_workspace_to_mat_bytes_for_owner(
        &entries,
        MatEncodingOwner::MatFile,
        Some(&mut cooperation),
    )
    .await?;
    cooperation.completed(bytes.len()).await?;
    resolve_matfile(runtime, &object)?;
    cooperation
        .await_io(owner.source.write(&bytes))
        .await?
        .map_err(|err| {
            matfile_error_with_source(
                &MATFILE_ERROR_IO,
                format!(
                    "matfile: failed to write '{}': {err}",
                    owner.source.path().display()
                ),
                err,
            )
        })?;
    resolve_matfile(runtime, &object)?;
    Ok(Value::Object(object))
}

pub fn ensure_matfile_class_registered() {
    MATFILE_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        properties.insert(
            PROPERTIES_FIELD.into(),
            crate::class_registry::RuntimeProperty {
                name: PROPERTIES_FIELD.into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Private,
                default_value: None,
            },
        );

        let mut methods = HashMap::new();
        methods.insert(
            OBJECT_SUBSREF_METHOD.into(),
            crate::class_registry::RuntimeMethod {
                name: OBJECT_SUBSREF_METHOD.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: MATFILE_SUBSREF.into(),
                implicit_class_argument: None,
            },
        );
        methods.insert(
            OBJECT_SUBSASGN_METHOD.into(),
            crate::class_registry::RuntimeMethod {
                name: OBJECT_SUBSASGN_METHOD.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: MATFILE_SUBSASGN.into(),
                implicit_class_argument: None,
            },
        );

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: MATFILE_CLASS.into(),
            parent: None,
            properties,
            methods,
        });
    });
}

fn matfile_object(
    source: &str,
    writable: bool,
    authority: runmat_value::ForeignRef,
) -> ObjectInstance {
    let mut object = ObjectInstance::new(MATFILE_CLASS.owned()).with_matfile_authority(authority);
    object.properties.insert(
        PROPERTIES_FIELD.to_string(),
        properties_struct(source.to_string(), writable),
    );
    object
}

fn properties_struct(source: String, writable: bool) -> Value {
    let mut st = StructValue::new();
    st.insert(SOURCE_FIELD, Value::String(source));
    st.insert(WRITABLE_FIELD, Value::Bool(writable));
    Value::Struct(st)
}

fn into_matfile_object(value: Value, builtin: &str) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.class_name == MATFILE_CLASS => Ok(object),
        other => Err(matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            format!("{builtin}: receiver must be a {MATFILE_CLASS} object, got {other:?}"),
        )),
    }
}

async fn read_existing_entries_for_assignment(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &ObjectInstance,
    path: &MatFsPath,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<Vec<(String, Value)>> {
    match cooperation.await_io(path.metadata()).await? {
        Ok(_) => {
            resolve_matfile(runtime, object)?;
            let entries = read_mat_file_for_builtin_on_with_resource_liveness(
                path,
                BUILTIN_NAME,
                runtime,
                cooperation.resource_liveness_lease(),
            )
            .await?;
            resolve_matfile(runtime, object)?;
            Ok(entries)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(matfile_error_with_source(
            &MATFILE_ERROR_IO,
            format!(
                "matfile: failed to inspect '{}': {err}",
                path.path().display()
            ),
            err,
        )),
    }
}

enum PreparedWritableValue {
    Known(bool),
    Resident(runmat_accelerate_api::GpuTensorHandle),
}

struct PreparedWritablePair {
    value: PreparedWritableValue,
}

struct PreparedWritableOptions(Vec<PreparedWritablePair>);

fn prepare_writable_options(rest: Vec<Value>) -> BuiltinResult<PreparedWritableOptions> {
    if !rest.len().is_multiple_of(2) {
        return Err(matfile_error_with(
            &MATFILE_ERROR_INVALID_OPTION,
            "matfile: name-value arguments must come in pairs",
        ));
    }
    let mut pairs = Vec::with_capacity(rest.len() / 2);
    let mut values = rest.into_iter();
    while let Some(name) = values.next() {
        let value = values.next().expect("validated complete Writable pair");
        let name = value_to_string_scalar(&name).ok_or_else(|| {
            matfile_error_with(
                &MATFILE_ERROR_INVALID_OPTION,
                "matfile: option names must be character vectors or string scalars",
            )
        })?;
        if MatFileOptionName::parse(&name) != Some(MatFileOptionName::Writable) {
            return Err(matfile_error_with(
                &MATFILE_ERROR_INVALID_OPTION,
                format!("matfile: unsupported option '{name}'"),
            ));
        }
        let value = match value {
            Value::GpuTensor(handle) => PreparedWritableValue::Resident(handle),
            value => {
                PreparedWritableValue::Known(value_to_bool_scalar(&value).ok_or_else(|| {
                    matfile_error_with(
                        &MATFILE_ERROR_INVALID_OPTION,
                        "matfile: Writable must be a scalar logical or numeric value",
                    )
                })?)
            }
        };
        pairs.push(PreparedWritablePair { value });
    }
    Ok(PreparedWritableOptions(pairs))
}

impl PreparedWritableOptions {
    fn identity_controls(&self) -> Vec<MatHandleControl> {
        self.0
            .iter()
            .map(|pair| match &pair.value {
                PreparedWritableValue::Known(value) => MatHandleControl::KnownWritable(*value),
                PreparedWritableValue::Resident(handle) => {
                    MatHandleControl::ResidentWritable(handle.wire_descriptor())
                }
            })
            .collect()
    }

    async fn resolve(self, cooperation: &mut MatCodecCooperation<'_>) -> BuiltinResult<bool> {
        let mut writable = false;
        for pair in self.0 {
            writable = match pair.value {
                PreparedWritableValue::Known(value) => value,
                PreparedWritableValue::Resident(handle) => {
                    let resident = Value::GpuTensor(handle);
                    let gathered = cooperation
                        .await_io(gather_if_needed_async(&resident))
                        .await??;
                    value_to_bool_scalar(&gathered).ok_or_else(|| {
                        matfile_error_with(
                            &MATFILE_ERROR_INVALID_OPTION,
                            "matfile: Writable must be a scalar logical or numeric value",
                        )
                    })?
                }
            };
        }
        Ok(writable)
    }
}

fn normalise_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw = value_to_string_scalar(value).ok_or_else(|| {
        matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            "matfile: filename must be a character vector or string scalar",
        )
    })?;
    if raw.trim().is_empty() {
        return Err(matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            "matfile: filename must not be empty",
        ));
    }
    let mut path = PathBuf::from(raw);
    if path.extension().is_none() {
        path.set_extension("mat");
    }
    Ok(path)
}

fn member_name(value: &Value, builtin: &str) -> BuiltinResult<String> {
    value_to_string_scalar(value).ok_or_else(|| {
        matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            format!("{builtin}: member name must be a character vector or string scalar"),
        )
    })
}

fn reject_internal_field(field: &str) -> BuiltinResult<()> {
    if field.starts_with("__runmat_") {
        return Err(matfile_error_with(
            &MATFILE_ERROR_SELECTION,
            format!("matfile: variable '{field}' was not found"),
        ));
    }
    Ok(())
}

fn validate_variable_name(field: &str, builtin: &str) -> BuiltinResult<()> {
    let mut chars = field.chars();
    let Some(first) = chars.next() else {
        return Err(matfile_error_with(
            &MATFILE_ERROR_SELECTION,
            format!("{builtin}: variable name must not be empty"),
        ));
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || chars.any(|ch| !(ch == '_' || ch.is_ascii_alphanumeric()))
    {
        return Err(matfile_error_with(
            &MATFILE_ERROR_SELECTION,
            format!("{builtin}: unsupported MAT-file variable name '{field}'"),
        ));
    }
    Ok(())
}

fn replace_or_append(entries: &mut Vec<(String, Value)>, field: String, value: Value) {
    if let Some((_, existing)) = entries.iter_mut().find(|(name, _)| name == &field) {
        *existing = value;
    } else {
        entries.push((field, value));
    }
}

fn value_to_string_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(StringArray { data, .. }) if data.len() == 1 => Some(data[0].clone()),
        _ => None,
    }
}

fn value_to_bool_scalar(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Num(n) if n.is_finite() => Some(*n != 0.0),
        Value::Int(i) => Some(!i.is_zero()),
        _ => None,
    }
}

pub(super) fn matfile_error(message: impl Into<String>) -> RuntimeError {
    matfile_error_with(&MATFILE_ERROR_INVALID_ARGUMENT, message)
}

fn matfile_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn matfile_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
#[path = "matfile/tests.rs"]
mod tests;
