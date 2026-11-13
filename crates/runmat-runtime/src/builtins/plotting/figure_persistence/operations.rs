use super::*;

mod load;
mod save;
mod support;

pub(super) use load::{execute_hgload, execute_openfig};
pub(super) use save::{execute_hgsave, execute_saveas, execute_savefig};

async fn in_selected_runtime<T>(
    runtime: Option<&crate::context::RuntimeContext>,
    future: impl std::future::Future<Output = T>,
) -> T {
    match runtime {
        Some(runtime) => runtime.scope(future).await,
        None => future.await,
    }
}
