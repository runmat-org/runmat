mod ancestry;
mod isolation;
#[cfg(not(target_family = "wasm"))]
mod native_lane;
mod services;
mod session;

pub(crate) use ancestry::MexCallbackAncestry;

/// Runtime-side ownership of decoded MEX outputs before output selection.
/// Wire and native MEX crates keep their portable, untraced result formats.
#[derive(Debug)]
pub struct RootedMexInvocation {
    pub outputs: crate::execution::RootedValues,
    pub warnings: Vec<runmat_mex::MexDiagnostic>,
    pub console: String,
}

pub use isolation::*;
pub use services::RuntimeMexHostServices;
pub use session::{
    MexArtifactExpectation, MexNameSelection, MexNativeLaneChildFactory, MexRuntimeSession,
    MexSessionRetirement, PreparedMexArtifactCatalog, ReadyMexArtifactCatalog,
    ReadyMexIsolationPolicy,
};
