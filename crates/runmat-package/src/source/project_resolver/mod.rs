mod error;
mod loader;
mod model;
mod resolver;
mod selection;
mod source;

pub use error::ProjectResolveError;
pub use model::{GitPackageMount, GitPackageProvider, ProjectResolveOptions, ResolvedProject};
pub use resolver::resolve_project_async;
