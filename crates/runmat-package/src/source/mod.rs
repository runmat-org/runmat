mod catalog;
mod project;

pub use catalog::{
    FrozenProject, FrozenSourceDescriptor, PackageMount, PackageSourceCatalog, SourceCatalog,
    StableSourceId,
};
pub use project::{
    build_frozen_project, build_frozen_project_async, discover_frozen_project_from,
    discover_frozen_project_from_async, FrozenProjectError,
};
