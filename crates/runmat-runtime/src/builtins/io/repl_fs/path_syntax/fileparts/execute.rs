use runmat_value::Value;

use crate::BuiltinResult;

const IDENTITY: &str = "fileparts";

pub(super) fn run(args: Vec<Value>) -> BuiltinResult<Value> {
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
    let mut folders = Vec::with_capacity(input.values.len());
    let mut names = Vec::with_capacity(input.values.len());
    let mut extensions = Vec::with_capacity(input.values.len());
    for value in &input.values {
        let (folder, name, extension) = super::super::lexical::split(value);
        folders.push(folder);
        names.push(name);
        extensions.push(extension);
    }
    let outputs = [folders, names, extensions]
        .into_iter()
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
    Ok(match crate::output_count::current_output_count() {
        Some(count) => crate::output_count::output_list_with_padding(count, outputs),
        None => Value::OutputList(outputs),
    })
}
