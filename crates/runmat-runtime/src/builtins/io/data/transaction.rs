//! Managed data receiver runtime bindings.

use super::*;
use crate::context::{host_effect::DatasetHandleOperation as Op, RuntimeContext};

#[runtime_builtin(
    name = "DataTransaction.id",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_id_builtin(runtime: Option<RuntimeContext>, base: Value) -> BuiltinResult<Value> {
    let (owner, tx_id) = authority::resolve_transaction(runtime.as_ref(), &base)?;
    handle_admission::execute(runtime.as_ref(), &owner, Op::TransactionId, &[], async {
        Ok(Value::string(tx_id))
    })
    .await
}

#[runtime_builtin(
    name = "DataTransaction.write",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_write_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    array_name: Value,
    slice: Value,
    values: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(data_error(
            "DataTransaction.write does not accept trailing options",
        ));
    }
    let (owner, tx_id) = authority::resolve_transaction(runtime.as_ref(), &base)?;
    let array_name_text = parse_string(&array_name, "DataTransaction.write arrayName")?;
    let facts = handle_admission::inputs(&[&array_name, &slice, &values]);
    handle_admission::execute_facts(
        runtime.as_ref(),
        &owner,
        Op::TransactionWrite,
        &facts,
        async {
            with_tx_mut(&tx_id, |tx| {
                if tx.status != TxnStatus::Open {
                    return Err(data_error("DataTransaction.write: transaction is not open"));
                }
                tx.writes.push(PendingWrite {
                    array: array_name_text,
                    slice_spec: Some(slice),
                    value: values,
                });
                Ok(())
            })?;
            Ok(Value::Bool(true))
        },
    )
    .await
}

#[runtime_builtin(
    name = "DataTransaction.set_attr",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_set_attr_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    key: Value,
    value: Value,
) -> BuiltinResult<Value> {
    let (owner, tx_id) = authority::resolve_transaction(runtime.as_ref(), &base)?;
    let key_text = parse_string(&key, "DataTransaction.set_attr key")?;
    let facts = handle_admission::inputs(&[&key, &value]);
    handle_admission::execute_facts(
        runtime.as_ref(),
        &owner,
        Op::TransactionSetAttr,
        &facts,
        async {
            with_tx_mut(&tx_id, |tx| {
                if tx.status != TxnStatus::Open {
                    return Err(data_error(
                        "DataTransaction.set_attr: transaction is not open",
                    ));
                }
                tx.attrs.insert(key_text, value);
                Ok(())
            })?;
            Ok(Value::Bool(true))
        },
    )
    .await
}

#[runtime_builtin(
    name = "DataTransaction.set_attrs",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_set_attrs_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    attrs: Value,
) -> BuiltinResult<Value> {
    let (owner, tx_id) = authority::resolve_transaction(runtime.as_ref(), &base)?;
    let facts = handle_admission::inputs(&[&attrs]);
    let Value::Struct(incoming) = attrs else {
        return Err(data_error(
            "DataTransaction.set_attrs: attrs must be struct",
        ));
    };
    handle_admission::execute_facts(
        runtime.as_ref(),
        &owner,
        Op::TransactionSetAttrs,
        &facts,
        async {
            with_tx_mut(&tx_id, |tx| {
                if tx.status != TxnStatus::Open {
                    return Err(data_error(
                        "DataTransaction.set_attrs: transaction is not open",
                    ));
                }
                for (k, v) in incoming.into_fields_unbounded() {
                    tx.attrs.insert(k, v);
                }
                Ok(())
            })?;
            Ok(Value::Bool(true))
        },
    )
    .await
}

#[runtime_builtin(
    name = "DataTransaction.resize",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_resize_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    array_name: Value,
    new_shape: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(data_error(
            "DataTransaction.resize does not accept trailing options",
        ));
    }
    let (owner, tx_id) = authority::resolve_transaction(runtime.as_ref(), &base)?;
    let array_name_text = parse_string(&array_name, "DataTransaction.resize arrayName")?;
    let shape = parse_shape_from_value(&new_shape)?;
    let facts = handle_admission::inputs(&[&array_name, &new_shape]);
    handle_admission::execute_facts(
        runtime.as_ref(),
        &owner,
        Op::TransactionResize,
        &facts,
        async {
            with_tx_mut(&tx_id, |tx| {
                if tx.status != TxnStatus::Open {
                    return Err(data_error(
                        "DataTransaction.resize: transaction is not open",
                    ));
                }
                tx.resizes.push(PendingResize {
                    array: array_name_text,
                    shape,
                });
                Ok(())
            })?;
            Ok(Value::Bool(true))
        },
    )
    .await
}

#[runtime_builtin(
    name = "DataTransaction.fill",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_fill_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    array_name: Value,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(data_error("DataTransaction.fill expects at most one slice"));
    }
    let (owner, tx_id) = authority::resolve_transaction(runtime.as_ref(), &base)?;
    let array_name_text = parse_string(&array_name, "DataTransaction.fill arrayName")?;
    let mut controls = vec![&array_name, &value];
    controls.extend(rest.iter());
    let facts = handle_admission::inputs(&controls);
    let slice_spec = rest.first().cloned();
    handle_admission::execute_facts(
        runtime.as_ref(),
        &owner,
        Op::TransactionFill,
        &facts,
        async {
            with_tx_mut(&tx_id, |tx| {
                if tx.status != TxnStatus::Open {
                    return Err(data_error("DataTransaction.fill: transaction is not open"));
                }
                tx.fills.push(PendingFill {
                    array: array_name_text,
                    slice_spec,
                    value,
                });
                Ok(())
            })?;
            Ok(Value::Bool(true))
        },
    )
    .await
}

#[runtime_builtin(
    name = "DataTransaction.delete_array",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_delete_array_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    array_name: Value,
) -> BuiltinResult<Value> {
    let (owner, tx_id) = authority::resolve_transaction(runtime.as_ref(), &base)?;
    let array_name_text = parse_string(&array_name, "DataTransaction.delete_array arrayName")?;
    let facts = handle_admission::inputs(&[&array_name]);
    handle_admission::execute_facts(
        runtime.as_ref(),
        &owner,
        Op::TransactionDeleteArray,
        &facts,
        async {
            with_tx_mut(&tx_id, |tx| {
                if tx.status != TxnStatus::Open {
                    return Err(data_error(
                        "DataTransaction.delete_array: transaction is not open",
                    ));
                }
                tx.delete_arrays.push(array_name_text);
                Ok(())
            })?;
            Ok(Value::Bool(true))
        },
    )
    .await
}

#[runtime_builtin(
    name = "DataTransaction.create_array",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_create_array_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    array_name: Value,
    meta: Value,
) -> BuiltinResult<Value> {
    let (owner, tx_id) = authority::resolve_transaction(runtime.as_ref(), &base)?;
    let array_name_text = parse_string(&array_name, "DataTransaction.create_array arrayName")?;
    let facts = handle_admission::inputs(&[&array_name, &meta]);
    let meta = parse_array_meta(&array_name_text, &meta)?;
    handle_admission::execute_facts(
        runtime.as_ref(),
        &owner,
        Op::TransactionCreateArray,
        &facts,
        async {
            with_tx_mut(&tx_id, |tx| {
                if tx.status != TxnStatus::Open {
                    return Err(data_error(
                        "DataTransaction.create_array: transaction is not open",
                    ));
                }
                tx.create_arrays.push(PendingCreateArray {
                    array: array_name_text,
                    meta,
                });
                Ok(())
            })?;
            Ok(Value::Bool(true))
        },
    )
    .await
}

#[runtime_builtin(
    name = "DataTransaction.commit",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_commit_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    commit(runtime, base, rest, Op::TransactionCommit).await
}

async fn commit(
    runtime: Option<RuntimeContext>,
    base: Value,
    rest: crate::call::BuiltinArguments,
    operation: Op,
) -> BuiltinResult<Value> {
    let mut identity_budget = super::handle_identity::HashBudget::new();
    let mut input_facts = rest
        .iter()
        .map(|value| handle_admission::input_with_budget(value, &mut identity_budget))
        .collect::<Vec<_>>();
    let options = match (rest.len(), rest.first()) {
        (0, _) => None,
        (1, Some(Value::Struct(options))) => Some(options),
        (1, _) => {
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
    let (owner, tx_id) = authority::resolve_transaction(runtime.as_ref(), &base)?;
    let (
        dataset_path,
        base_sequence,
        writes,
        resizes,
        fills,
        create_arrays,
        delete_arrays,
        attrs,
        queue_facts,
    ) = with_tx(&tx_id, |tx| {
        if tx.status != TxnStatus::Open {
            return Err(data_error(
                "DataTransaction.commit: transaction is not open",
            ));
        }
        let mut queue_facts = vec![handle_admission::queue_input(tx)];
        for write in &tx.writes {
            if identity_budget.is_exhausted() {
                break;
            }
            if let Some(slice) = &write.slice_spec {
                queue_facts.push(handle_admission::input_with_budget(
                    slice,
                    &mut identity_budget,
                ));
            }
            queue_facts.push(handle_admission::input_with_budget(
                &write.value,
                &mut identity_budget,
            ));
        }
        for fill in &tx.fills {
            if identity_budget.is_exhausted() {
                break;
            }
            if let Some(slice) = &fill.slice_spec {
                queue_facts.push(handle_admission::input_with_budget(
                    slice,
                    &mut identity_budget,
                ));
            }
            queue_facts.push(handle_admission::input_with_budget(
                &fill.value,
                &mut identity_budget,
            ));
        }
        for value in tx.attrs.values() {
            if identity_budget.is_exhausted() {
                break;
            }
            queue_facts.push(handle_admission::input_with_budget(
                value,
                &mut identity_budget,
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
            queue_facts,
        ))
    })?;
    input_facts.extend(queue_facts);

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
    handle_admission::execute_facts(runtime.as_ref(), &owner, operation, &input_facts, async {
        let mut budget = runtime.as_ref().map(RuntimeContext::cpu_cooperation);
        let root = &owner.root;
        let provider = owner.filesystem.as_ref();
        crate::data::cooperate_data_work(&mut budget, 0).await?;
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
            create_array_in_manifest(
                provider,
                root,
                &mut manifest,
                &create.array,
                create.meta,
                &mut budget,
            )
            .await?;
        }
        for resize in resizes {
            resize_array_in_manifest(
                provider,
                root,
                &mut manifest,
                &resize.array,
                resize.shape,
                &mut budget,
            )
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
                &mut budget,
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
                &mut budget,
            )
            .await?;
        }
        for array_name in delete_arrays {
            delete_array_in_manifest_on(provider, root, &mut manifest, &array_name, &mut budget)
                .await?;
        }
        for (k, v) in attrs {
            crate::data::cooperate_data_work(&mut budget, 1).await?;
            manifest.attrs.insert(k, value_to_json(&v)?);
        }
        crate::data::cooperate_data_work(&mut budget, 0).await?;
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
    })
    .await
}

#[runtime_builtin(
    name = "commit",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_commit_alias_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    match &base {
        Value::Object(obj) if obj.class_name().is(runmat_builtins::DATA_TRANSACTION_CLASS) => {
            commit(runtime, base, rest, Op::CommitAlias).await
        }
        _ => Err(data_error(
            "commit: receiver must be a DataTransaction (use tx = ds.begin())",
        )),
    }
}

#[runtime_builtin(
    name = "DataTransaction.abort",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_abort_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let (owner, tx_id) = authority::resolve_transaction(runtime.as_ref(), &base)?;
    handle_admission::execute(runtime.as_ref(), &owner, Op::TransactionAbort, &[], async {
        with_tx_mut(&tx_id, |tx| {
            tx.status = TxnStatus::Aborted;
            Ok(())
        })?;
        tracing::info!(target: "runmat.data", tx_id = tx_id, "data transaction abort");
        remove_tx(&tx_id)?;
        Ok(Value::Bool(true))
    })
    .await
}

#[runtime_builtin(
    name = "DataTransaction.status",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::data::transaction"
)]
async fn data_tx_status_builtin(
    runtime: Option<RuntimeContext>,
    base: Value,
) -> BuiltinResult<Value> {
    let (owner, tx_id) = authority::resolve_transaction(runtime.as_ref(), &base)?;
    handle_admission::execute(
        runtime.as_ref(),
        &owner,
        Op::TransactionStatus,
        &[],
        async {
            with_tx(&tx_id, |tx| {
                let status = match tx.status {
                    TxnStatus::Open => "open",
                    TxnStatus::Committed => "committed",
                    TxnStatus::Aborted => "aborted",
                };
                Ok(Value::string(status.to_string()))
            })
        },
    )
    .await
}
