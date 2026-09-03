//! Reusable exact algorithms for discrete number-theory builtins.

pub(super) mod binary;
mod primality;

pub(super) use primality::{is_prime, prime_factors};
