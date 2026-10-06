mod component;
mod dependencies;
mod envelope;
mod identity;
mod manifest;
mod revision;
mod section;

pub use component::{
    ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
};
pub use dependencies::{
    ExecutableDependencyDescriptor, ExecutableDependencyProduct, EXECUTABLE_DEPENDENCY_MAX_PRODUCTS,
};
pub use envelope::{ExecutableUnitEnvelope, EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES};
pub use identity::{ExecutableEntrypointKind, ExecutableIdentity};
pub use manifest::{ExecutableUnitManifest, EXECUTABLE_UNIT_SCHEMA_VERSION};
pub use revision::ExecutableComponentRevisions;
pub use section::{ExecutableOptionalSection, ExecutableSectionSupport, SectionRequirement};
