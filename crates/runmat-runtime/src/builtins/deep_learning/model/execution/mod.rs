//! Executable deep-learning kernels. Public dispatch and network admission
//! remain with `model`; this subtree owns numerical execution only.

pub(in crate::builtins::deep_learning) mod host;
