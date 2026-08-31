use serde::Serialize;

/// Execution lane used by the standalone documentation-example verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinExampleHarness {
    /// Must agree in the native runtime and the browser/WASM runtime.
    Portable,
    Native,
    Browser,
    BrowserGraphics,
    NativeFilesystem,
    NativeLoopbackNetwork,
    Wgpu,
    NativeForeignRuntime,
    InteractiveHost,
}

/// Semantic oracle for a documentation example. Presentation output is kept
/// separately so formatting changes do not silently redefine correctness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinExampleVerification {
    Succeeds,
    Assertions {
        source: &'static str,
    },
    ExpectedError {
        identifier: &'static str,
    },
    Figure {
        minimum_figures: usize,
        assertions: &'static str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinExample {
    /// Stable within the owning builtin identity; export combines both values.
    pub id: &'static str,
    pub title: &'static str,
    pub program: &'static str,
    pub display_output: Option<&'static str>,
    pub harness: BuiltinExampleHarness,
    pub verification: BuiltinExampleVerification,
}
