mod dependency;
mod model;
mod target;

pub use dependency::{DependencyGroup, DependencyLocator, DependencySpec, GitSelector};
pub use model::{PackageManifest, PublicationDeclaration, RegistryDeclaration, SourceReplacement};
pub use target::{HostCapability, TargetEnvironment, TargetPredicate};
