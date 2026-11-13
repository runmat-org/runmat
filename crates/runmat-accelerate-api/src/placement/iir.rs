//! Physical IIR route identity used by provider feasibility and cost queries.
use serde::{Deserialize, Serialize};

/// Stable presentation token at the capability-query boundary. Numerical
/// dispatch uses the closed route enum, never a builtin spelling comparison.
pub const PROVIDER_IIR_PREPARED_PASS_OPERATION: &str = "signal.iir.prepared_pass";
pub const PROVIDER_IIR_CAUSAL_PASS_OPERATION: &str = "signal.iir.causal_pass";
pub const PROVIDER_IIR_ZERO_PHASE_OPERATION: &str = "signal.iir.zero_phase";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderIirRoute {
    /// One causal pass over the admitted signal axes and order-first state.
    CausalPass,
    /// One provider pass with one previously prepared resident coefficient set.
    PreparedPass,
    /// Reflected two-pass filtering with state scaling and trim on the device.
    ZeroPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderIirWorkload {
    pub route: ProviderIirRoute,
    /// The zero-based filtered axis after host-admitted shape extension.
    pub dim: u32,
    /// Exact number of state elements per filtered channel.
    pub state_len: u64,
    /// Requested reflection extent on each end; zero for a causal pass.
    #[serde(default)]
    pub reflection_len: u64,
}
