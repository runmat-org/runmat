use super::*;

pub(super) fn apply_hgload_overrides(
    handles: &[FigureHandle],
    overrides: Option<&StructValue>,
) -> BuiltinResult<Value> {
    let mut cells = Vec::with_capacity(handles.len());
    for handle in handles {
        let mut old = runmat_value::UnboundedStructValueBuilder::new();
        if let Some(overrides) = overrides {
            let mut set_args = Vec::new();
            for (name, value) in &overrides.fields {
                if let Ok(previous) =
                    get_properties(PlotHandle::Figure(*handle), Some(name), HGLOAD)
                {
                    old.insert(name, previous);
                    set_args.push(Value::string(name));
                    set_args.push(value.clone());
                }
            }
            if !set_args.is_empty() {
                set_properties(PlotHandle::Figure(*handle), &set_args, HGLOAD)?;
            }
        }
        cells.push(Value::Struct(old.finish()));
    }
    CellArray::new(cells, 1, handles.len())
        .map(Value::Cell)
        .map_err(|err| persistence_error(&ERROR_INTERNAL, HGLOAD, &err))
}

pub(super) fn handles_value(handles: &[FigureHandle]) -> Value {
    if handles.len() == 1 {
        Value::Num(handles[0].as_u32() as f64)
    } else {
        Value::Tensor(
            Tensor::new(
                handles.iter().map(|h| h.as_u32() as f64).collect(),
                vec![1, handles.len()],
            )
            .expect("figure handle row"),
        )
    }
}

pub(super) fn scalar_string(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(runmat_value::decode_utf16_code_units(
            ca.data(),
        )
        .map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                builtin,
                source.to_string(),
                source,
            )
        })?),
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        Value::CharArray(_) => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: character vector arguments must be 1-by-N"),
        )),
        Value::StringArray(_) => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: string array arguments must be scalar"),
        )),
        other => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: expected a string argument, got {other:?}"),
        )),
    }
}

pub(super) fn validate_filename(filename: &str, builtin: &'static str) -> BuiltinResult<()> {
    if filename.trim().is_empty() {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: filename must not be empty"),
        ));
    }
    if filename.contains('\0') {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: filename must not contain NUL bytes"),
        ));
    }
    Ok(())
}
