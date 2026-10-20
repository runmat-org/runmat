use super::receiver::replace_or_append;
use super::*;
use crate::execution::{RootedValue, RootedValueLease};

pub(super) async fn matfile_read_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    resource_liveness: Option<std::sync::Arc<dyn HostEffectResourceLivenessLease>>,
) -> BuiltinResult<RootedValue> {
    let _input_roots = RootedValueLease::for_values(std::iter::once(&obj))
        .map_err(|error| error.into_runtime_error())?;
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
        let value = object
            .properties()
            .get(PROPERTIES_FIELD)
            .cloned()
            .ok_or_else(|| matfile_error("matfile object is missing Properties"))?;
        return RootedValue::new(value).map_err(|error| error.into_runtime_error());
    }
    let entries = read_mat_file_for_builtin_on_with_resource_liveness(
        &owner.source,
        BUILTIN_NAME,
        runtime,
        resource_liveness,
    )
    .await?;
    resolve_matfile(runtime, &object)?;
    entries.try_map_owned(
        |mut entries| {
            let index = entries
                .0
                .iter()
                .position(|(name, _)| name == &field)
                .ok_or_else(|| {
                    matfile_error_with(
                        &MATFILE_ERROR_SELECTION,
                        format!(
                            "matfile: variable '{field}' was not found in '{}'",
                            owner.source.path().display()
                        ),
                    )
                })?;
            Ok(entries.0.swap_remove(index).1)
        },
        |error| error.into_runtime_error(),
    )
}

pub(super) async fn matfile_write_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
    resource_liveness: Option<std::sync::Arc<dyn HostEffectResourceLivenessLease>>,
) -> BuiltinResult<RootedValue> {
    let _input_roots = RootedValueLease::for_values([&obj, &rhs].into_iter())
        .map_err(|error| error.into_runtime_error())?;
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
    let _value_root = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|error| error.into_runtime_error())?;
    resolve_matfile(runtime, &object)?;
    entries = entries.try_map_owned(
        |mut entries| {
            replace_or_append(&mut entries.0, field, value);
            Ok::<_, crate::RuntimeError>(entries)
        },
        |error| error.into_runtime_error(),
    )?;
    cooperation.completed(0).await?;
    let bytes = encode_workspace_to_mat_bytes_for_owner(
        &entries.value().0,
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
    RootedValue::new(Value::Object(object)).map_err(|error| error.into_runtime_error())
}

async fn read_existing_entries_for_assignment(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &ObjectInstance,
    path: &MatFsPath,
    cooperation: &mut MatCodecCooperation<'_>,
) -> BuiltinResult<super::super::load::RootedMatEntries> {
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
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            super::super::load::RootedMatEntries::new(super::super::load::MatEntries(Vec::new()))
                .map_err(|error| error.into_runtime_error())
        }
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
