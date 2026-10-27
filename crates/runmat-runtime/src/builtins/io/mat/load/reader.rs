use super::*;
use crate::builtins::io::mat::format::{MatArray, MatClass, MatData};
use futures::future::LocalBoxFuture;

pub(crate) async fn read_mat_file_for_builtin(
    path: &Path,
    builtin: &str,
) -> crate::BuiltinResult<RootedMatEntries> {
    match read_mat_file(path).await {
        Ok(entries) => Ok(entries),
        Err(err) => {
            let message = err.message().replacen("load:", &format!("{builtin}:"), 1);
            let mut builder = build_runtime_error(message).with_builtin(builtin);
            if let Some(identifier) = err.identifier() {
                builder = builder.with_identifier(identifier);
            }
            Err(builder.with_source(err).build())
        }
    }
}

pub(crate) async fn read_mat_file_for_builtin_on_with_resource_liveness(
    source: &MatFsPath,
    builtin: &str,
    runtime: Option<&crate::context::RuntimeContext>,
    resource_liveness: Option<std::sync::Arc<dyn HostEffectResourceLivenessLease>>,
) -> crate::BuiltinResult<RootedMatEntries> {
    match read_mat_file_on_with_resource_liveness(source, runtime, resource_liveness).await {
        Ok(entries) => Ok(entries),
        Err(err) => {
            let message = err.message().replacen("load:", &format!("{builtin}:"), 1);
            let mut builder = build_runtime_error(message).with_builtin(builtin);
            if let Some(identifier) = err.identifier() {
                builder = builder.with_identifier(identifier);
            }
            Err(builder.with_source(err).build())
        }
    }
}

pub(crate) async fn read_mat_file(path: &Path) -> BuiltinResult<RootedMatEntries> {
    let source = MatFsPath::capture(path.to_path_buf()).map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to resolve MAT path: {err}"),
            err,
        )
    })?;
    read_mat_file_on(&source).await
}

pub(crate) async fn read_mat_file_on(source: &MatFsPath) -> BuiltinResult<RootedMatEntries> {
    read_mat_file_on_with_context(source, None).await
}

pub(crate) async fn read_mat_file_on_with_context(
    source: &MatFsPath,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<RootedMatEntries> {
    read_mat_file_on_with_resource_liveness(source, runtime, None).await
}

pub(crate) async fn read_mat_file_on_with_resource_liveness(
    source: &MatFsPath,
    runtime: Option<&crate::context::RuntimeContext>,
    resource_liveness: Option<std::sync::Arc<dyn HostEffectResourceLivenessLease>>,
) -> BuiltinResult<RootedMatEntries> {
    let mut cooperation =
        decoder::MatCodecCooperation::new(runtime, Some(source), "load", &LOAD_ERROR_IO)
            .with_resource_liveness(resource_liveness);
    let file = cooperation.await_io(source.open()).await?.map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!("load: failed to open '{}': {err}", source.path().display()),
            err,
        )
    })?;
    let mut reader = BufReader::new(file);
    let arrays = decoder::read_mat_arrays_cooperative(&mut reader, &mut cooperation).await?;
    check_source_current(source)?;
    let entries = restore_mat_arrays(arrays, cooperation.runtime()).await?;
    restore_loaded_entries(entries, Some(source), Some(&mut cooperation)).await
}

pub fn decode_workspace_from_mat_bytes(bytes: &[u8]) -> BuiltinResult<RootedMatEntries> {
    let mut cursor = Cursor::new(bytes);
    let arrays = decoder::read_mat_arrays(&mut cursor)?;
    let entries = futures::executor::block_on(restore_mat_arrays(arrays, None))?;
    futures::executor::block_on(restore_loaded_entries(entries, None, None))
}

fn contains_mat_object(array: &MatArray) -> bool {
    match &array.data {
        MatData::Object { .. } => true,
        MatData::Cell { elements } => elements.iter().any(contains_mat_object),
        MatData::Struct { field_values, .. } => field_values.iter().any(contains_mat_object),
        _ => false,
    }
}

fn restore_mat_array<'a>(
    array: MatArray,
    runtime: Option<&'a crate::context::RuntimeContext>,
) -> LocalBoxFuture<'a, BuiltinResult<RootedValue>> {
    Box::pin(async move {
        if !contains_mat_object(&array) {
            return RootedValue::new(decoder::mat_array_to_value(array)?)
                .map_err(|error| error.into_runtime_error());
        }
        let MatArray { class, dims, data } = array;
        match data {
            MatData::Object {
                class_name,
                field_names,
                mut field_values,
            } => {
                let count = dims
                    .iter()
                    .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                    .ok_or_else(|| load_error("load: object shape exceeds platform limits"))?;
                if class != MatClass::Object
                    || field_names.len() != 2
                    || field_names[0] != "__runmat_saved_kind"
                    || field_names[1] != "__runmat_saved_payload"
                    || count.checked_mul(2) != Some(field_values.len())
                {
                    return Err(load_error("load: unsupported object MAT field layout"));
                }
                let (class, product) =
                    crate::builtins::introspection::object_serialization::resolve_mat_object_class(
                        &class_name,
                        runtime,
                    )
                    .await?;
                let payloads = field_values.split_off(count);
                let mut restored = Vec::with_capacity(count);
                for (kind, payload) in field_values.into_iter().zip(payloads) {
                    let MatData::Logical { data } = kind.data else {
                        return Err(load_error("load: object element kind is not logical"));
                    };
                    if kind.dims != [1, 1] || data.len() != 1 || data[0] > 1 {
                        return Err(load_error("load: object element kind is invalid"));
                    }
                    if let crate::context::CurrentClassProductResolution::Selected(facts) = &product
                    {
                        if facts.is_handle_class() != (data[0] != 0) {
                            return Err(load_error(
                                "load: object element kind differs from the current selected class",
                            ));
                        }
                    }
                    let payload = restore_mat_array(payload, runtime).await?;
                    crate::builtins::introspection::object_serialization::validate_mat_object_product_current(
                        &product, runtime,
                    )?;
                    let object = payload
                        .with_owned_async(|value| {
                            crate::builtins::introspection::object_serialization::restore_mat_object_element(
                                &class,
                                &product,
                                data[0] != 0,
                                value,
                                runtime,
                            )
                        })
                        .await?;
                    crate::builtins::introspection::object_serialization::validate_mat_object_product_current(
                        &product, runtime,
                    )?;
                    restored.push(
                        crate::builtins::introspection::object_serialization::reconcile_mat_object_product(
                            object, &class, &product, data[0] != 0, runtime,
                        )?,
                    );
                }
                let result = if count == 1 && dims == [1, 1] {
                    restored.pop().expect("one object element")
                } else {
                    let provenance = match &product {
                        crate::context::CurrentClassProductResolution::Selected(facts) => {
                            Some(facts)
                        }
                        crate::context::CurrentClassProductResolution::RegisteredBuiltin(_) => None,
                    };
                    let mut output = crate::execution::object_array::ObjectArrayConstruction::new(
                        runtime,
                        crate::execution::object_array::Class::Name(class.display_name()),
                        count,
                        &dims,
                        provenance.into_iter(),
                        provenance.is_none(),
                    )?;
                    for child in &restored {
                        output.push(child.value().clone())?;
                    }
                    output.finish()?
                };
                crate::builtins::introspection::object_serialization::validate_mat_object_product_current(
                    &product,runtime)?;
                Ok(result)
            }
            MatData::Cell { elements } => {
                let mut restored = Vec::with_capacity(elements.len());
                for element in elements {
                    restored.push(restore_mat_array(element, runtime).await?);
                }
                let values = restored
                    .into_iter()
                    .map(|value| value.with_owned(std::convert::identity))
                    .collect();
                let cell =
                    runmat_value::CellArray::from_column_major(values, dims).map_err(load_error)?;
                RootedValue::new(Value::Cell(cell)).map_err(|error| error.into_runtime_error())
            }
            MatData::Struct {
                field_names,
                field_values,
            } => {
                let mut restored = Vec::with_capacity(field_values.len());
                for field in field_values {
                    restored.push(restore_mat_array(field, runtime).await?);
                }
                let values = restored
                    .into_iter()
                    .map(|value| value.with_owned(std::convert::identity))
                    .collect();
                let value =
                    runmat_value::StructArray::normalize_field_major(field_names, values, dims)
                        .map_err(load_error)?;
                RootedValue::new(value).map_err(|error| error.into_runtime_error())
            }
            _ => unreachable!("object descendant requires aggregate MAT node"),
        }
    })
}

#[cfg(test)]
pub(crate) fn restore_typed_mat_array_for_test<'a>(
    array: MatArray,
    runtime: &'a crate::context::RuntimeContext,
) -> LocalBoxFuture<'a, BuiltinResult<RootedValue>> {
    restore_mat_array(array, Some(runtime))
}

#[cfg(test)]
pub(crate) fn restore_typed_mat_entry_for_test<'a>(
    array: MatArray,
    runtime: &'a crate::context::RuntimeContext,
) -> LocalBoxFuture<'a, BuiltinResult<RootedValue>> {
    Box::pin(async move {
        let decoded = restore_mat_array(array, Some(runtime)).await?;
        decoded
            .with_owned_async(|value| restore_value_from_mat_load_in_context(value, Some(runtime)))
            .await
    })
}

async fn restore_mat_arrays(
    arrays: Vec<(String, MatArray)>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Vec<(String, Value)>> {
    let mut restored = Vec::with_capacity(arrays.len());
    for (name, array) in arrays {
        restored.push((name, restore_mat_array(array, runtime).await?));
    }
    Ok(restored
        .into_iter()
        .map(|(name, value)| (name, value.with_owned(std::convert::identity)))
        .collect())
}

pub(super) async fn restore_loaded_entries(
    entries: Vec<(String, Value)>,
    source: Option<&MatFsPath>,
    mut cooperation: Option<&mut decoder::MatCodecCooperation<'_>>,
) -> BuiltinResult<RootedMatEntries> {
    let _decoded_roots =
        crate::execution::RootedValueLease::for_values(entries.iter().map(|(_, value)| value))
            .map_err(|error| error.into_runtime_error())?;
    let mut restored: Vec<(String, Value)> = Vec::new();
    let mut retained = Vec::new();
    restored.try_reserve_exact(entries.len()).map_err(|error| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            "load: restored values cannot be allocated",
            error,
        )
    })?;
    for (name, value) in entries {
        if let Some(cooperation) = cooperation.as_deref_mut() {
            cooperation.completed(0).await?;
        }
        if let Some(source) = source {
            check_source_current(source)?;
        }
        let value = match classify_mat_load_restoration(value) {
            MatLoadRestoration::Ready(leaf) => {
                if let Some(cooperation) = cooperation.as_deref_mut() {
                    cooperation.completed(0).await?;
                }
                let rooted = leaf.into_rooted()?;
                if let Some(cooperation) = cooperation.as_deref_mut() {
                    cooperation.completed(0).await?;
                }
                rooted
            }
            MatLoadRestoration::Recursive(value) => match cooperation.as_deref_mut() {
                Some(cooperation) => {
                    let runtime = cooperation.runtime();
                    cooperation
                        .await_io(restore_value_from_mat_load_in_context(value, runtime))
                        .await??
                }
                None => restore_value_from_mat_load(value).await?,
            },
        };
        value.with_owned(|value| {
            let roots = crate::execution::RootedValueLease::for_values(std::iter::once(&value))
                .map_err(|error| error.into_runtime_error())?;
            restored.push((name, value));
            retained.push(roots);
            Ok::<_, crate::RuntimeError>(())
        })?;
        if let Some(source) = source {
            check_source_current(source)?;
        }
        if let Some(cooperation) = cooperation.as_deref_mut() {
            cooperation.completed(1).await?;
        }
    }
    if let Some(cooperation) = cooperation {
        cooperation.completed(0).await?;
    }
    let entries =
        RootedMatEntries::new(MatEntries(restored)).map_err(|error| error.into_runtime_error())?;
    drop(retained);
    Ok(entries)
}

fn check_source_current(source: &MatFsPath) -> BuiltinResult<()> {
    source.check_current().map_err(|err| {
        load_error_with_source(
            &LOAD_ERROR_IO,
            format!(
                "load: MAT source authority for '{}' changed: {err}",
                source.path().display()
            ),
            err,
        )
    })
}
