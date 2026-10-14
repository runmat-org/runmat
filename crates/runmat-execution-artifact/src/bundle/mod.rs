mod builder;
mod closure;
mod foreign;
mod manifest;
mod selected_foreign;
mod selected_unit;
mod validator;

pub use builder::{ExecutionBundleBuilder, SourceReader};
pub use closure::{BundleCodeClosure, CompiledPackageClosure};
pub use foreign::ForeignArtifactClosure;
pub use manifest::{
    BuildResourceDeclaration, BundleCallable, BundleManifest, ExecutionBundle,
    ProjectRevisionRecord, EXECUTION_BUNDLE_SCHEMA_VERSION,
};
pub use selected_foreign::{
    PreparedForeignParentView, SelectedForeignProjection, SelectedPreparedForeignMaterial,
};
