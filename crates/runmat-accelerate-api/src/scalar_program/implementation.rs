//! Descriptive physical lowering evidence, never source execution authority.
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ScalarImplementation {
    Binary32Product48RneV1,
    Binary32Square48RneV1,
    Binary32IdentityWordsV1,
    Binary32ZeroPowerWordsV1,
    Binary32WidenToBinary64WordsV1,
}

/// One actually emitted SSA implementation. Slicing may omit semantic nodes;
/// consumers must not infer these records from a program's possible recipes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct EmittedScalarImplementation {
    pub output: u32,
    pub implementation: ScalarImplementation,
    pub invocations_per_element: u32,
    /// Cold algorithmic stage prior, not ISA instructions or measured time.
    pub algorithmic_units_per_invocation: u64,
}

/// Storage transport is independent of numerical and accumulator precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ScalarTransport {
    NativeFloat,
    Binary32Words,
}
