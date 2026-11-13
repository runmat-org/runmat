//! Public Python builtin identities. The foreign adapter remains the sole
//! authority for Python execution, conversion, and session ownership.

mod contract;
mod descriptor;
mod documentation_support;
mod inference_support;
pub mod pyargs;
pub mod pyenv;
pub mod pyrun;
pub mod pyrunfile;
pub mod terminate;
#[cfg(test)]
mod tests;

pub use contract::PYTHON_ENVIRONMENT_CLASS;
pub use descriptor::{
    PYTHON_ERROR_HOST_UNAVAILABLE, PYTHON_ERROR_INVALID, PYTHON_ERROR_UNAVAILABLE,
};
pub use pyargs::PYARGS_CATALOG_ENTRY;
pub use pyargs::PYARGS_DESCRIPTOR;
pub use pyenv::PYENV_CATALOG_ENTRY;
pub use pyenv::PYENV_DESCRIPTOR;
pub use pyrun::PYRUN_CATALOG_ENTRY;
pub use pyrun::PYRUN_DESCRIPTOR;
pub use pyrunfile::PYRUNFILE_CATALOG_ENTRY;
pub use pyrunfile::PYRUNFILE_DESCRIPTOR;
pub use terminate::TERMINATE_CATALOG_ENTRY;
pub use terminate::TERMINATE_DESCRIPTOR;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &PYARGS_CATALOG_ENTRY,
    &PYENV_CATALOG_ENTRY,
    &PYRUN_CATALOG_ENTRY,
    &PYRUNFILE_CATALOG_ENTRY,
    &TERMINATE_CATALOG_ENTRY,
];
