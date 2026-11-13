pub(crate) mod auto_promote;
#[cfg(feature = "native-accel")]
pub mod fusion;
#[cfg(feature = "native-accel")]
pub(crate) mod fusion_declaration;
#[cfg(feature = "native-accel")]
pub(crate) mod graph;
#[cfg(all(test, feature = "native-accel"))]
mod residency_tests;
#[cfg(feature = "native-accel")]
pub(crate) mod stack_layout;
