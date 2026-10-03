use super::*;
use futures::executor::block_on;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

mod fixtures;
mod admission;
mod extensible;
mod metadata;
mod samples;
use fixtures::*;
