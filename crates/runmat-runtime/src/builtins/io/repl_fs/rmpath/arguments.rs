pub(super) fn parse(tokens: Vec<String>) -> crate::BuiltinResult<Vec<String>> {
    let directories: Vec<String> = tokens
        .iter()
        .map(String::as_str)
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .flat_map(super::super::path_mutation::segments::split)
        .collect();
    if directories.is_empty() {
        return Err(super::errors::descriptor(
            &runmat_builtins::RMPATH_ERROR_TOO_FEW_ARGUMENTS,
        ));
    }
    Ok(directories)
}
