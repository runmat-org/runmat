//! Admission and selection of one complete network invocation. Numerical
//! execution remains in the existing host and mixed-route owners.

mod observation;
mod request;

pub(in crate::builtins::deep_learning) use observation::{
    is_provider_download_failure, PendingWholeWorkObservation,
};
pub(in crate::builtins::deep_learning) use request::{select, SelectedNetworkRoute};

#[cfg(test)]
mod tests;
