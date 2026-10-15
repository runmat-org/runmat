use runmat_execution_artifact::target_pack::ComponentId;
use runmat_package::NormalizedRelativePath;

use super::invalid;
use crate::AotResult;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct NativeInstalledComponent {
    pub component: ComponentId,
    pub path: NormalizedRelativePath,
}

/// Explicit target-relative layout. The complete sysroot is retained beneath
/// `sysroot`; frameworks, resource directories and link topology are not
/// flattened or rewritten to accommodate a build-host path.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct NativeInstallationLayout {
    pub executable: NormalizedRelativePath,
    pub sysroot: NormalizedRelativePath,
    pub components: Vec<NativeInstalledComponent>,
}

impl NativeInstallationLayout {
    pub(super) fn validate(&self, selected: &[ComponentId]) -> AotResult<()> {
        if self.components.len() != selected.len()
            || !self
                .components
                .iter()
                .map(|entry| &entry.component)
                .eq(selected)
            || self
                .components
                .windows(2)
                .any(|pair| pair[0].component >= pair[1].component)
        {
            return Err(invalid(
                "installation layout must retain the exact selected runtime closure",
            ));
        }
        // All path collision, ancestor and link checks are performed by the
        // shared tree validator after assembling the complete payload.
        Ok(())
    }
}
