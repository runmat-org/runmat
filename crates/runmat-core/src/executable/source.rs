use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// Request-scoped authority for one explicitly selected primary file that is
/// inside the project workspace but outside its indexed source roots. The
/// access path is only for the current host; portable source maps carry the
/// root-relative name and captured bytes, never this path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CapturedPrimarySource {
    pub(crate) owner_identity: String,
    pub(crate) relative_path: String,
    pub(crate) access_path: String,
    workspace_root: String,
    content_digest: runmat_package::ContentDigest,
    graph_digest: runmat_package::ContentDigest,
    source_revision: runmat_package::ContentDigest,
}

pub(crate) enum AdmittedProjectPrimary {
    Indexed {
        source: runmat_package::StableSourceId,
        access_path: String,
    },
    Captured(CapturedPrimarySource),
}

impl AdmittedProjectPrimary {
    pub(crate) async fn issue(
        project: &runmat_package::FrozenProject,
        access_path: &Path,
        text: &str,
    ) -> Result<Self, String> {
        let selected = canonical_access_path(access_path).await?;
        let mut indexed = None;
        for (descriptor, path) in project.all_sources() {
            if canonical_access_path(path).await? == selected
                && indexed.replace((descriptor, path)).is_some()
            {
                return Err("primary source has ambiguous indexed project ownership".into());
            }
        }
        if let Some((descriptor, path)) = indexed {
            if runmat_package::ContentDigest::sha256(text.as_bytes())
                != descriptor.id.content_digest
            {
                return Err("source text differs from its frozen project descriptor".into());
            }
            return Ok(Self::Indexed {
                source: descriptor.id.clone(),
                access_path: path.to_string_lossy().replace('\\', "/"),
            });
        }
        CapturedPrimarySource::capture(project, &selected, text)
            .await
            .map(Self::Captured)
    }

    pub(crate) fn into_request_parts(self) -> (Option<CapturedPrimarySource>, String) {
        match self {
            Self::Indexed { access_path, .. } => (None, access_path),
            Self::Captured(source) => {
                let access_path = source.access_path.clone();
                (Some(source), access_path)
            }
        }
    }
}

impl CapturedPrimarySource {
    async fn capture(
        project: &runmat_package::FrozenProject,
        selected: &Path,
        text: &str,
    ) -> Result<Self, String> {
        let workspace = canonical_access_path(&project.workspace_root).await?;
        let relative = selected
            .strip_prefix(&workspace)
            .map_err(|_| "explicit primary source is outside the captured project workspace")?;
        let relative = runmat_package::NormalizedRelativePath::new(relative)
            .map_err(|error| error.to_string())?;
        if relative.as_str() == "." {
            return Err("explicit primary source has no project-relative file name".into());
        }
        Ok(Self {
            owner_identity: project.graph.root.to_string(),
            relative_path: relative.to_string(),
            access_path: selected.to_string_lossy().replace('\\', "/"),
            workspace_root: project.workspace_root.to_string_lossy().replace('\\', "/"),
            content_digest: runmat_package::ContentDigest::sha256(text.as_bytes()),
            graph_digest: project.graph.graph_digest.clone(),
            source_revision: project.sources.revision.clone(),
        })
    }

    pub(crate) fn validate(
        &self,
        handoff: &runmat_package::FrozenProjectHandoff,
        source: &ExecutableSource,
    ) -> Result<(), String> {
        handoff.validate().map_err(|error| error.to_string())?;
        if self.owner_identity != handoff.project.graph.root.to_string()
            || self.graph_digest != handoff.project.graph.graph_digest
            || self.source_revision != handoff.project.sources.revision
            || self.workspace_root
                != handoff
                    .project
                    .workspace_root
                    .to_string_lossy()
                    .replace('\\', "/")
            || self.owner_identity != source.owner_identity
            || self.relative_path != source.relative_path
            || self.content_digest != runmat_package::ContentDigest::sha256(source.text.as_bytes())
        {
            return Err(
                "captured primary source differs from its installed project or bytes".into(),
            );
        }
        Ok(())
    }

    pub(crate) fn matches_access(&self, access: &str, text: &str) -> bool {
        self.access_path == access.replace('\\', "/")
            && self.content_digest == runmat_package::ContentDigest::sha256(text.as_bytes())
    }
}

async fn canonical_access_path(path: &Path) -> Result<PathBuf, String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        runmat_filesystem::canonicalize_async(path)
            .await
            .map_err(|error| format!("cannot resolve primary source access: {error}"))
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Browser providers use rooted virtual paths; `Path::is_absolute` is
        // false for every path on wasm32-unknown-unknown (no prefix grammar).
        if !crate::diagnostic_path::host_path_is_absolute(path)
            || path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err("primary source access must be an absolute normalized virtual path".into());
        }
        Ok(path.to_path_buf())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableSource {
    pub owner_identity: String,
    pub relative_path: String,
    pub text: String,
    lookup_path: Option<String>,
    captured_primary: Option<CapturedPrimarySource>,
}

impl<'a> From<ExecutableSource> for Cow<'a, ExecutableSource> {
    fn from(source: ExecutableSource) -> Self {
        Self::Owned(source)
    }
}

impl<'a> From<&'a ExecutableSource> for Cow<'a, ExecutableSource> {
    fn from(source: &'a ExecutableSource) -> Self {
        Self::Borrowed(source)
    }
}

impl ExecutableSource {
    /// Select an exact indexed source, or issue a distinct primary-only origin
    /// for a direct/declared path outside `[sources].roots`.
    pub async fn capture_project_primary(
        project: &runmat_package::FrozenProject,
        access_path: &Path,
    ) -> Result<Self, String> {
        let selected = canonical_access_path(access_path).await?;
        let text = runmat_runtime::builtins::io::repl_fs::pcode::read_source_text_async(&selected)
            .await
            .map_err(|error| format!("cannot read selected primary source: {error}"))?;
        Self::from_project_primary_text(project, &selected, text).await
    }

    async fn from_project_primary_text(
        project: &runmat_package::FrozenProject,
        selected: &Path,
        text: String,
    ) -> Result<Self, String> {
        match AdmittedProjectPrimary::issue(project, selected, &text).await? {
            AdmittedProjectPrimary::Indexed { source, .. } => {
                Self::from_project_source(&source, text)
            }
            AdmittedProjectPrimary::Captured(captured) => {
                Self::with_captured_primary(text, captured)
            }
        }
    }

    pub(crate) fn with_captured_primary(
        text: impl Into<String>,
        captured: CapturedPrimarySource,
    ) -> Result<Self, String> {
        let text = text.into();
        if captured.content_digest != runmat_package::ContentDigest::sha256(text.as_bytes()) {
            return Err("selected primary text differs from captured bytes".into());
        }
        Ok(Self {
            owner_identity: captured.owner_identity.clone(),
            relative_path: captured.relative_path.clone(),
            text,
            lookup_path: None,
            captured_primary: Some(captured),
        })
    }

    pub(crate) fn captured_primary(&self) -> Option<&CapturedPrimarySource> {
        self.captured_primary.as_ref()
    }

    /// Bind source text to an exact frozen project descriptor. This pure
    /// admission is shared by native and browser producers; it performs no I/O.
    pub fn from_project_source(
        source: &runmat_package::StableSourceId,
        text: impl Into<String>,
    ) -> Result<Self, String> {
        let text = text.into();
        if runmat_package::ContentDigest::sha256(text.as_bytes()) != source.content_digest {
            return Err("source text differs from its frozen project descriptor".into());
        }
        Ok(Self::new(
            source.package_instance.to_string(),
            source.relative_path.to_string(),
            text,
        ))
    }

    pub fn new(
        owner_identity: impl Into<String>,
        relative_path: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            owner_identity: owner_identity.into(),
            relative_path: relative_path.into(),
            text: text.into(),
            lookup_path: None,
            captured_primary: None,
        }
    }

    /// Construct a source whose portable identity is logical while project discovery reads the
    /// already-frozen physical file selected by the caller.
    pub fn frozen(
        owner_identity: impl Into<String>,
        relative_path: impl Into<String>,
        lookup_path: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            owner_identity: owner_identity.into(),
            relative_path: relative_path.into(),
            text: text.into(),
            lookup_path: Some(lookup_path.into()),
            captured_primary: None,
        }
    }

    pub(crate) fn lookup_path(&self) -> &str {
        self.lookup_path.as_deref().unwrap_or(&self.relative_path)
    }

    pub(crate) fn explicit_lookup_path(&self) -> Option<&str> {
        self.lookup_path.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn explicit_primary_has_exact_bytes_without_becoming_an_indexed_source() {
        let fixture = tempfile::tempdir().expect("project fixture");
        std::fs::create_dir_all(fixture.path().join("src")).expect("source root");
        std::fs::create_dir_all(fixture.path().join("scripts")).expect("entrypoint directory");
        std::fs::write(
            fixture.path().join("src/helper.m"),
            "function y = helper(); y = 2; end\n",
        )
        .expect("indexed helper");
        let primary = fixture.path().join("scripts/report.m");
        std::fs::write(&primary, "answer = helper() + 40;\n").expect("explicit entrypoint");
        let manifest = fixture.path().join("runmat.toml");
        std::fs::write(
            &manifest,
            "[package]\nname = \"captured-primary\"\n[sources]\nroots = [\"src\"]\n[entrypoints.report]\npath = \"scripts/report.m\"\n",
        )
        .expect("project manifest");
        let project = runmat_package::build_frozen_project(&manifest, BTreeSet::new())
            .expect("frozen project");
        let source = futures::executor::block_on(ExecutableSource::capture_project_primary(
            &project, &primary,
        ))
        .expect("captured explicit primary");
        let captured = source
            .captured_primary()
            .expect("distinct primary authority");
        let handoff = runmat_package::FrozenProjectHandoff::new(project.clone());
        captured.validate(&handoff, &source).expect("exact source");
        assert_eq!(source.relative_path, "scripts/report.m");
        assert!(!project.all_sources().any(|(descriptor, _)| descriptor
            .id
            .relative_path
            .to_string()
            == source.relative_path));

        let mut session = crate::RunMatSession::with_options(false, false).expect("session");
        futures::executor::block_on(session.install_project_handoff(handoff.clone()))
            .expect("installed project");
        let unit = futures::executor::block_on(session.compile_executable_unit(&source, None))
            .expect("captured primary with its indexed helper");
        assert!(unit
            .source_map()
            .entries()
            .iter()
            .any(|entry| entry.relative_path == "src/helper.m"));

        let altered = ExecutableSource::new(
            source.owner_identity.clone(),
            source.relative_path.clone(),
            "answer = helper() + 41;\n",
        );
        assert!(captured.validate(&handoff, &altered).is_err());
        let external = tempfile::NamedTempFile::new().expect("external source path");
        assert!(futures::executor::block_on(AdmittedProjectPrimary::issue(
            &project,
            external.path(),
            "answer = 1;\n"
        ))
        .is_err());
    }
}
