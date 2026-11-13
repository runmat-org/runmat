use runmat_value::Value;

pub(super) async fn run(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::GENPATH_DESCRIPTOR,
        &runmat_builtins::GENPATH_ERROR_TOO_MANY_OUTPUTS,
        "genpath",
    )?;
    let request = super::input::decode(args).await?;
    let root = super::root::resolve(request.root.as_deref()).await?;
    let exclusions =
        super::exclusions::Exclusions::resolve(request.excludes.as_deref(), &root).await;
    let folders = super::traversal::collect(&root, &exclusions).await?;
    Ok(super::output::path_list(&folders))
}
