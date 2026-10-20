use runmat_value::Value;

use crate::BuiltinResult;

const IDENTITY: &str = "fileparts";

pub(super) fn run(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    if requested.is_some_and(|count| count > 3) {
        return Err(super::super::error::catalog(
            &runmat_builtins::FILEPARTS_ERROR_OUTPUTS,
            IDENTITY,
        ));
    }
    if args.len() != 1 {
        return Err(super::super::error::catalog(
            &runmat_builtins::FILEPARTS_ERROR_ARITY,
            IDENTITY,
        ));
    }
    let input = super::super::text::TextContainer::decode(
        &args[0],
        IDENTITY,
        &runmat_builtins::FILEPARTS_ERROR_TYPE,
    )?;
    let count = requested.unwrap_or(3);
    if count == 0 {
        return Ok(Value::output_list(Vec::new()));
    }
    let mut folders = Vec::with_capacity(input.values.len());
    let mut names = (count >= 2).then(|| Vec::with_capacity(input.values.len()));
    let mut extensions = (count >= 3).then(|| Vec::with_capacity(input.values.len()));
    for value in &input.values {
        let (folder, name, extension) = super::super::lexical::split(value);
        folders.push(folder);
        if let Some(names) = names.as_mut() {
            names.push(name);
        }
        if let Some(extensions) = extensions.as_mut() {
            extensions.push(extension);
        }
    }
    let outputs = std::iter::once(folders)
        .chain(names)
        .chain(extensions)
        .map(|values| {
            super::super::text::output(
                input.representation,
                values,
                &input.shape,
                IDENTITY,
                &runmat_builtins::FILEPARTS_ERROR_TYPE,
            )
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(match requested {
        Some(count) => crate::output_count::output_list_with_padding(count, outputs),
        None => Value::output_list(outputs),
    })
}
