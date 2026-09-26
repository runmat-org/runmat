mod codec;
mod envelope;
mod error;

pub use codec::{
    decode_interpreter_program_v2, decode_interpreter_script_v2, encode_interpreter_program_v2,
    encode_interpreter_script_v2,
};
pub use error::{InterpreterPayloadError, InterpreterPayloadForm, InterpreterRevisionField};

#[cfg(test)]
#[path = "artifact/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "artifact/float_tests.rs"]
mod float_tests;
