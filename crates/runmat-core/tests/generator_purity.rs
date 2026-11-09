#![cfg(not(target_arch = "wasm32"))]
//! A `.m` function built only from elementwise arithmetic, comparisons and pure math
//! builtins is a deterministic value function with certified local suspension, so FEA
//! admits it as a signal generator.

use runmat_core::{ExecutableSource, RunMatSession};

const IMPACT_PULSE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/simulation/examples/cantilever-impact/impactPulse.m"
));

#[test]
fn elementwise_pulse_generator_is_a_deterministic_value_function() {
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let source = ExecutableSource::new("impact-pulse@1", "impactPulse.m", IMPACT_PULSE);
    let unit = futures::executor::block_on(session.compile_executable_unit(source, None))
        .expect("compile impactPulse.m");
    let interface = unit
        .primary_function_interface("impactPulse")
        .expect("impactPulse interface");
    // Signal generators admit MaySuspend only with the compiler's proof that every
    // suspension origin is argument-local numerical work (sin, min and arithmetic here).
    assert!(
        interface.effects.is_deterministic_value_function() && interface.cooperative_suspension,
        "effects {:?}, cooperative_suspension {}",
        interface.effects,
        interface.cooperative_suspension
    );
    assert!(
        interface.capabilities.0.is_empty(),
        "capabilities {:?}",
        interface.capabilities
    );
}
