use runmat_value::Value;

pub(super) async fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let request = super::input::decode(args).await?;
    let requested_outputs = request.requested_outputs;
    Ok(evaluate(request).await?.value(requested_outputs))
}

pub(super) async fn evaluate(
    request: super::input::Request,
) -> crate::BuiltinResult<super::result::Outcome> {
    let target = match super::target::resolve(request.filename.as_deref()).await {
        Ok(target) => target,
        Err(super::target::ResolveError::Status(failure)) => {
            return Ok(super::result::Outcome::failure(failure));
        }
        Err(super::target::ResolveError::Compatibility(error)) => return Err(error),
    };
    let path = crate::builtins::common::path_state::current_path_string();
    let contents = super::contents::build(&path);
    match super::persistence::write(&target, &contents).await {
        Ok(()) => Ok(super::result::Outcome::success()),
        Err(failure) => Ok(super::result::Outcome::failure(failure)),
    }
}
