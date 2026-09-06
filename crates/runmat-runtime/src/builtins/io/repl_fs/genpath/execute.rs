use runmat_value::Value;

pub(super) async fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let request = super::input::decode(args).await?;
    let root = super::root::resolve(request.root.as_deref()).await?;
    let exclusions =
        super::exclusions::Exclusions::resolve(request.excludes.as_deref(), &root).await;
    let folders = super::traversal::collect(&root, &exclusions).await?;
    Ok(super::output::path_list(&folders))
}
