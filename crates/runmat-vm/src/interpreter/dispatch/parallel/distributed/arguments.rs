use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub(super) fn decode<const N: usize>(arguments: Vec<Value>) -> Result<[Value; N], RuntimeError> {
    arguments.try_into().map_err(|_| {
        crate::interpreter::errors::mex(
            "InvalidDistributedInstruction",
            "distributed instruction operand count does not match its bytecode contract",
        )
    })
}
