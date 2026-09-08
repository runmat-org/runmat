mod classify;
mod collector;
mod contract;
mod empty;
mod finish;
mod upload;

#[cfg(test)]
pub(in crate::builtins::acceleration::gpu::arrayfun) use classify::{
    classify_value as classify_for_test, ClassifiedValue as ClassifiedValueForTest,
};
pub(super) use collector::UniformCollector;
pub(super) use contract::OutputContract;
pub(super) use empty::empty_uniform;
pub(super) use upload::maybe_upload_uniform;
