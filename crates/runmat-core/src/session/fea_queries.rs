//! Bounded, session-owned snapshots for semantic browsing. A query never reads
//! a provider, recompiles a study, or executes a function.
use super::CheckedFeaSessionSource;
use runmat_runtime::analysis::{
    FeaBrowsePage, FeaBrowseRequest, FeaQueryError, FeaSourceCheckResult, SourceDigest,
};
use std::{collections::VecDeque, sync::Arc};

const MAX_SNAPSHOTS: usize = 4;
const MAX_BYTES: usize = 128 * 1024 * 1024;

#[derive(Default)]
pub(super) struct CheckedSourceCache {
    snapshots: VecDeque<(FeaSourceCheckResult, Arc<CheckedFeaSessionSource>, usize)>,
    bytes: usize,
    provider: Option<Arc<dyn runmat_filesystem::FsProvider>>,
    project: Option<runmat_package::ProjectRevision>,
}

impl CheckedSourceCache {
    fn belongs_to(
        &self,
        provider: &Arc<dyn runmat_filesystem::FsProvider>,
        project: &Option<runmat_package::ProjectRevision>,
    ) -> bool {
        self.provider
            .as_ref()
            .is_some_and(|owner| Arc::ptr_eq(owner, provider))
            && &self.project == project
    }

    fn admit(
        &mut self,
        provider: Arc<dyn runmat_filesystem::FsProvider>,
        project: Option<runmat_package::ProjectRevision>,
    ) {
        if !self.belongs_to(&provider, &project) {
            *self = Self::default();
            self.provider = Some(provider);
            self.project = project;
        }
    }
    pub fn insert(&mut self, result: &FeaSourceCheckResult, source: Arc<CheckedFeaSessionSource>) {
        if self
            .snapshots
            .iter()
            .any(|(existing, _, _)| existing.revision == result.revision)
        {
            return;
        }
        let Ok(report_bytes) = serde_json::to_vec(result).map(|bytes| bytes.len()) else {
            return;
        };
        // Bound retained input and derived facts. Captures are shared between
        // roots, but charging each entry conservatively prevents unbounded retention.
        let input_bytes = source.captured_sources().map_or_else(
            || {
                source
                    .program
                    .bound
                    .graph
                    .modules
                    .values()
                    .map(|module| module.source.source.len())
                    .sum::<usize>()
            },
            |files| files.values().map(Vec::len).sum::<usize>(),
        );
        let Some(bytes) = report_bytes
            .checked_mul(4)
            .and_then(|derived| derived.checked_add(input_bytes))
        else {
            return;
        };
        if bytes > MAX_BYTES {
            return;
        }
        while self.snapshots.len() >= MAX_SNAPSHOTS || self.bytes + bytes > MAX_BYTES {
            if let Some((_, _, size)) = self.snapshots.pop_front() {
                self.bytes -= size;
            } else {
                break;
            }
        }
        self.bytes += bytes;
        self.snapshots.push_back((result.clone(), source, bytes));
    }

    pub fn source(
        &mut self,
        revision: SourceDigest,
    ) -> Result<Arc<CheckedFeaSessionSource>, FeaQueryError> {
        let index = self
            .snapshots
            .iter()
            .position(|(report, _, _)| report.revision == revision)
            .ok_or(FeaQueryError::StaleRevision)?;
        let entry = self
            .snapshots
            .remove(index)
            .expect("located checked source");
        let source = entry.1.clone();
        self.snapshots.push_back(entry);
        Ok(source)
    }

    pub fn browse(&mut self, request: &FeaBrowseRequest) -> Result<FeaBrowsePage, FeaQueryError> {
        let index = self
            .snapshots
            .iter()
            .position(|(value, _, _)| value.revision == request.revision)
            .ok_or(FeaQueryError::StaleRevision)?;
        let entry = self
            .snapshots
            .remove(index)
            .expect("located cached revision");
        let result = entry.0.browse(request);
        self.snapshots.push_back(entry);
        result
    }
}

impl super::RunMatSession {
    pub(super) fn retain_checked_fea_source(
        &mut self,
        result: &FeaSourceCheckResult,
        source: Arc<CheckedFeaSessionSource>,
    ) {
        let provider = runmat_filesystem::current_provider();
        let project = self
            .project_handoff
            .as_ref()
            .map(|handoff| handoff.project.revision());
        self.checked_fea_sources.admit(provider, project);
        self.checked_fea_sources.insert(result, source);
    }

    fn checked_fea_scope_matches(&self) -> bool {
        self.checked_fea_sources.belongs_to(
            &runmat_filesystem::current_provider(),
            &self
                .project_handoff
                .as_ref()
                .map(|handoff| handoff.project.revision()),
        )
    }
    /// Retrieves the checked product, never recompiles a live file as a fallback.
    pub fn checked_fea_source(
        &mut self,
        revision: SourceDigest,
    ) -> Result<Arc<CheckedFeaSessionSource>, String> {
        if !self.checked_fea_scope_matches() {
            return Err(FeaQueryError::StaleRevision.to_string());
        }
        let source = self
            .checked_fea_sources
            .source(revision)
            .map_err(|error| error.to_string())?;
        if source.program_environment() != &crate::program_environment(self.compat_mode()) {
            return Err("the compiler environment changed after source checking".into());
        }
        Ok(source)
    }

    pub fn browse_fea_source(
        &mut self,
        request: &FeaBrowseRequest,
    ) -> Result<FeaBrowsePage, FeaQueryError> {
        if !self.checked_fea_scope_matches() {
            return Err(FeaQueryError::StaleRevision);
        }
        self.checked_fea_sources.browse(request)
    }
}
