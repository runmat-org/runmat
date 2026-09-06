use runmat_builtins::BuiltinErrorDescriptor;
use runmat_filesystem::{self as vfs, FsMetadata};
use runmat_value::{LogicalArray, Value};

use crate::{build_runtime_error, BuiltinResult};

use super::input::{self, PathInputs};

pub(super) async fn evaluate(
    args: &[Value],
    builtin: &'static str,
    arity_error: &'static BuiltinErrorDescriptor,
    path_error: &'static BuiltinErrorDescriptor,
    predicate: fn(&FsMetadata) -> bool,
) -> BuiltinResult<Value> {
    match input::parse(args, builtin, arity_error, path_error)? {
        PathInputs::Scalar(path) => Ok(Value::Bool(
            vfs::metadata_async(&path)
                .await
                .is_ok_and(|metadata| predicate(&metadata)),
        )),
        PathInputs::Shaped { paths, shape } => {
            let mut values = Vec::with_capacity(paths.len());
            for path in paths {
                values.push(
                    vfs::metadata_async(&path)
                        .await
                        .is_ok_and(|metadata| predicate(&metadata)),
                );
            }
            LogicalArray::new(values.into_iter().map(u8::from).collect(), shape)
                .map(Value::LogicalArray)
                .map_err(|error| {
                    build_runtime_error(format!("{builtin}: {error}"))
                        .with_builtin(builtin)
                        .build()
                })
        }
    }
}
