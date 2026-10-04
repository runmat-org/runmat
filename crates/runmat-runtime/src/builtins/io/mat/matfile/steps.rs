use super::receiver::replace_or_append;
use super::*;

pub(super) async fn matfile_read_step(
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

pub(super) async fn matfile_write_step(
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
