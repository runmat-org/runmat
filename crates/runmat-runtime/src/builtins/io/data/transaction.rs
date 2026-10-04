//! Managed data receiver runtime bindings.

use super::*;

#[runtime_builtin(
    name = "DataTransaction.id",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_id_builtin(base: Value) -> BuiltinResult<Value> {
    let (_, tx_id) = authority::resolve_transaction(&base)?;
    Ok(Value::String(tx_id))
}

#[runtime_builtin(
    name = "DataTransaction.write",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_write_builtin(
    base: Value,
    array_name: Value,
    slice: Value,
    values: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.write")?;
    let array_name = parse_string(&array_name, "DataTransaction.write arrayName")?;
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error("DataTransaction.write: transaction is not open"));
        }
        tx.writes.push(PendingWrite {
            array: array_name,
            slice_spec: Some(slice),
            value: values,
        });
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.set_attr",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_set_attr_builtin(base: Value, key: Value, value: Value) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.set_attr")?;
    let key = parse_string(&key, "DataTransaction.set_attr key")?;
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error(
                "DataTransaction.set_attr: transaction is not open",
            ));
        }
        tx.attrs.insert(key, value);
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.set_attrs",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_set_attrs_builtin(base: Value, attrs: Value) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.set_attrs")?;
    let Value::Struct(incoming) = attrs else {
        return Err(data_error(
            "DataTransaction.set_attrs: attrs must be struct",
        ));
    };
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error(
                "DataTransaction.set_attrs: transaction is not open",
            ));
        }
        for (k, v) in incoming.fields {
            tx.attrs.insert(k, v);
        }
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.resize",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_resize_builtin(
    base: Value,
    array_name: Value,
    new_shape: Value,
    _rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.resize")?;
    let array_name = parse_string(&array_name, "DataTransaction.resize arrayName")?;
    let shape = parse_shape_from_value(&new_shape)?;
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error(
                "DataTransaction.resize: transaction is not open",
            ));
        }
        tx.resizes.push(PendingResize {
            array: array_name,
            shape,
        });
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.fill",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_fill_builtin(
    base: Value,
    array_name: Value,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.fill")?;
    let array_name = parse_string(&array_name, "DataTransaction.fill arrayName")?;
    let slice_spec = rest.first().cloned();
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error("DataTransaction.fill: transaction is not open"));
        }
        tx.fills.push(PendingFill {
            array: array_name,
            slice_spec,
            value,
        });
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.delete_array",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_delete_array_builtin(base: Value, array_name: Value) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.delete_array")?;
    let array_name = parse_string(&array_name, "DataTransaction.delete_array arrayName")?;
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error(
                "DataTransaction.delete_array: transaction is not open",
            ));
        }
        tx.delete_arrays.push(array_name);
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.create_array",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_create_array_builtin(
    base: Value,
    array_name: Value,
    meta: Value,
) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.create_array")?;
    let array_name = parse_string(&array_name, "DataTransaction.create_array arrayName")?;
    let meta = parse_array_meta(&array_name, &meta)?;
    with_tx_mut(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error(
                "DataTransaction.create_array: transaction is not open",
            ));
        }
        tx.create_arrays.push(PendingCreateArray {
            array: array_name,
            meta,
        });
        Ok(())
    })?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.commit",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_commit_builtin(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let options = match rest.as_slice() {
        [] => None,
        [Value::Struct(options)] => Some(options),
        [_] => {
            return Err(data_error(
                "DataTransaction.commit: options must be a scalar struct",
            ))
        }
        _ => {
            return Err(data_error(
                "DataTransaction.commit: expected at most one options struct",
            ))
        }
    };
    let (owner, tx_id) = authority::resolve_transaction(&base)?;
    let (dataset_path, base_sequence, writes, resizes, fills, create_arrays, delete_arrays, attrs) =
        with_tx(&tx_id, |tx| {
            if tx.status != TxnStatus::Open {
                return Err(data_error(
                    "DataTransaction.commit: transaction is not open",
                ));
            }
            Ok((
                tx.dataset_path.clone(),
                tx.base_sequence,
                tx.writes.clone(),
                tx.resizes.clone(),
                tx.fills.clone(),
                tx.create_arrays.clone(),
                tx.delete_arrays.clone(),
                tx.attrs.clone(),
            ))
        })?;

    let write_ops = writes.len();
    let resize_ops = resizes.len();
    let fill_ops = fills.len();
    let create_ops = create_arrays.len();
    let delete_ops = delete_arrays.len();
    let attr_updates = attrs.len();

    if dataset_path != owner.public_path {
        return Err(data_error(
            "DataTransaction.commit: transaction owner changed",
        ));
    }
    let root = &owner.root;
    let provider = owner.filesystem.as_ref();
    let mut manifest = crate::data::read_manifest_on(provider, root).await?;
    ensure_manifest_sequence(base_sequence, &manifest)?;
    if let Some(options) = options {
        if let Some(expected) = options.fields.get("if_manifest") {
            let expected = parse_string(expected, "DataTransaction.commit if_manifest")?;
            let actual = manifest_version_token(&manifest);
            if expected != actual {
                tracing::warn!(
                    target: "runmat.data",
                    tx_id = tx_id,
                    expected_manifest = expected,
                    actual_manifest = actual,
                    "data transaction manifest conflict"
                );
                return Err(data_error(
                    "MANIFEST_CONFLICT: if_manifest precondition failed",
                ));
            }
        }
    }
    for create in create_arrays {
        create_array_in_manifest(provider, root, &mut manifest, &create.array, create.meta).await?;
    }
    for resize in resizes {
        resize_array_in_manifest(provider, root, &mut manifest, &resize.array, resize.shape)
            .await?;
    }
    for fill in fills {
        fill_array_in_manifest(
            provider,
            root,
            &mut manifest,
            &fill.array,
            fill.slice_spec.as_ref(),
            &fill.value,
        )
        .await?;
    }
    for write in writes {
        apply_write_to_manifest_on(
            provider,
            root,
            &mut manifest,
            &write.array,
            write.slice_spec.as_ref(),
            &write.value,
        )
        .await?;
    }
    for array_name in delete_arrays {
        delete_array_in_manifest_on(provider, root, &mut manifest, &array_name).await?;
    }
    for (k, v) in attrs {
        manifest.attrs.insert(k, value_to_json(&v));
    }
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    crate::data::write_manifest_on(provider, root, &manifest).await?;
    with_tx_mut(&tx_id, |tx| {
        tx.status = TxnStatus::Committed;
        Ok(())
    })?;
    tracing::info!(
        target: "runmat.data",
        dataset = dataset_path,
        tx_id = tx_id,
        write_ops = write_ops,
        resize_ops = resize_ops,
        fill_ops = fill_ops,
        create_ops = create_ops,
        delete_ops = delete_ops,
        attr_updates = attr_updates,
        next_sequence = manifest.txn_sequence,
        "data transaction commit"
    );
    remove_tx(&tx_id)?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "commit",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_commit_alias_builtin(base: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    match &base {
        Value::Object(obj)
            if obj
                .class_name
                .is(runmat_types::StaticClassIdentity::new("DataTransaction")) =>
        {
            data_tx_commit_builtin(base, rest).await
        }
        _ => Err(data_error(
            "commit: receiver must be a DataTransaction (use tx = ds.begin())",
        )),
    }
}

#[runtime_builtin(
    name = "DataTransaction.abort",
    binding_variant = "default",
    sink = true,
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_abort_builtin(base: Value) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.abort")?;
    with_tx_mut(&tx_id, |tx| {
        tx.status = TxnStatus::Aborted;
        Ok(())
    })?;
    tracing::info!(
        target: "runmat.data",
        tx_id = tx_id,
        "data transaction abort"
    );
    remove_tx(&tx_id)?;
    Ok(Value::Bool(true))
}

#[runtime_builtin(
    name = "DataTransaction.status",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::data"
)]
async fn data_tx_status_builtin(base: Value) -> BuiltinResult<Value> {
    let tx_id = tx_id_from_object(&base, "DataTransaction.status")?;
    with_tx(&tx_id, |tx| {
        let status = match tx.status {
            TxnStatus::Open => "open",
            TxnStatus::Committed => "committed",
            TxnStatus::Aborted => "aborted",
        };
        Ok(Value::String(status.to_string()))
    })
}
