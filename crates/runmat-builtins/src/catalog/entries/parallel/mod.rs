#[macro_use]
mod shared;

mod codistributors;
mod collectives;
mod context;
mod current_execution;
mod distributed_arrays;
mod documentation;
mod futures;
mod pools;

use shared::*;

pub use codistributors::*;
pub use collectives::*;
pub use context::*;
pub use current_execution::*;
pub use distributed_arrays::*;
pub use futures::*;
pub use pools::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(
        entries,
        &[
            codistributors::ENTRIES,
            collectives::ENTRIES,
            context::ENTRIES,
            current_execution::ENTRIES,
            distributed_arrays::ENTRIES,
            futures::ENTRIES,
            pools::ENTRIES,
        ],
    );
}
