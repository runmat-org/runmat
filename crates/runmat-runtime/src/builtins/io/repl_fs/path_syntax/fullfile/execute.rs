use runmat_value::Value;

use crate::BuiltinResult;

const IDENTITY: &str = "fullfile";

pub(super) async fn run(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(super::super::error::catalog(
            &runmat_builtins::FULLFILE_ERROR_NOT_ENOUGH_INPUTS,
            IDENTITY,
        ));
    }
    let mut inputs = Vec::with_capacity(args.len());
    for argument in &args {
        inputs.push(super::input::decode(argument).await?);
    }
    let shape = super::super::text::target_shape(&inputs).map_err(|()| {
        super::super::error::catalog(&runmat_builtins::FULLFILE_ERROR_SHAPE, IDENTITY)
    })?;
    let count = inputs
        .iter()
        .find(|input| !input.is_scalar())
        .map_or(1, |input| input.values.len());
    let representation = super::super::text::result_representation(&inputs);
    let mut joined = Vec::with_capacity(count);
    for index in 0..count {
        let parts: Vec<&str> = inputs.iter().map(|input| input.value_at(index)).collect();
        joined.push(super::super::lexical::join(&parts));
    }
    super::super::text::output(
        representation,
        joined,
        &shape,
        IDENTITY,
        &runmat_builtins::FULLFILE_ERROR_SHAPE,
    )
}
