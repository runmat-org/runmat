use runmat_execution_artifact::target_pack::ComponentId;
use runmat_package::NormalizedRelativePath;

use super::invalid;
use crate::AotResult;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeInstalledComponent {
    pub component: ComponentId,
    pub path: NormalizedRelativePath,
}

/// Explicit target-relative layout. The complete sysroot is retained beneath
/// `sysroot`; frameworks, resource directories and link topology are not
/// flattened or rewritten to accommodate a build-host path.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
        if let Some(notice) = self.components.iter().find(|entry| {
            entry.component.as_str() == runmat_runtime::third_party_notices::FILE_NAME
        }) {
            let executable = self.executable.as_str();
            let parent = executable.rsplit_once('/').map(|(parent, _)| parent);
            let expected = parent.map_or_else(
                || runmat_runtime::third_party_notices::FILE_NAME.to_string(),
                |parent| {
                    format!(
                        "{parent}/{}",
                        runmat_runtime::third_party_notices::FILE_NAME
                    )
                },
            );
            if notice.path.as_str() != expected {
                return Err(invalid(
                    "selected runtime notice must be installed beside the executable",
                ));
            }
        }
        // All path collision, ancestor and link checks are performed by the
        // shared tree validator after assembling the complete payload.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_notice_must_accompany_installed_executable() {
        let id = ComponentId::new(runmat_runtime::third_party_notices::FILE_NAME).unwrap();
        let selected = vec![id.clone()];
        let mut layout = NativeInstallationLayout {
            executable: NormalizedRelativePath::new("bin/runmat").unwrap(),
            sysroot: NormalizedRelativePath::new("sysroot").unwrap(),
            components: vec![NativeInstalledComponent {
                component: id,
                path: NormalizedRelativePath::new("other/THIRD_PARTY_NOTICES.txt").unwrap(),
            }],
        };
        assert!(layout.validate(&selected).is_err());
        layout.components[0].path =
            NormalizedRelativePath::new("bin/THIRD_PARTY_NOTICES.txt").unwrap();
        layout.validate(&selected).unwrap();
    }
}
