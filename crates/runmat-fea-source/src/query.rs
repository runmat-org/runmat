//! Revision-fenced navigation over checked domain facts. Queries never discover
//! or parse another source view and do not realize geometry or run a solve.
use crate::{
    FeaDeclaredMaterial, FeaModelMemberKind, FeaSourceCheckResult, FeaSourceIdentity,
    FeaSourceIndexEntry, FeaStudyMemberKind,
};
use runmat_analysis_language::{FeaSymbolKind, LogicalSourcePath, SourceDigest};
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaQueryDetail {
    Navigation,
    Definition,
}

#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeaQueryFilter {
    All,
    Identity {
        identity: FeaSourceIdentity,
    },
    /// Admit an exact bounded named-region selection, returning only its owner
    /// navigation entry. Validation stays beside the immutable domain facts.
    GeometryRegionSelection {
        geometry: FeaSourceIdentity,
        region_ids: Vec<String>,
        required_role: Option<runmat_geometry_core::GeometryRegionRole>,
    },
    Declaration {
        kind: FeaSymbolKind,
    },
    ModelMember {
        kind: FeaModelMemberKind,
    },
    StudyMember {
        kind: FeaStudyMemberKind,
    },
    Function,
}

#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaQueryCursor {
    pub revision: SourceDigest,
    pub offset: u32,
    pub filter: FeaQueryFilter,
    pub module_path: Option<LogicalSourcePath>,
    pub detail: FeaQueryDetail,
}

#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaBrowseRequest {
    pub revision: SourceDigest,
    pub module_path: Option<LogicalSourcePath>,
    pub filter: FeaQueryFilter,
    pub after: Option<FeaQueryCursor>,
    pub limit: u16,
    pub detail: FeaQueryDetail,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaBrowsePage {
    pub revision: SourceDigest,
    pub entries: Vec<FeaSourceIndexEntry>,
    pub next: Option<FeaQueryCursor>,
    pub total: u32,
    pub details: Vec<crate::FeaObjectDetails>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeaBrowseOutcome {
    Page { page: Box<FeaBrowsePage> },
    Rejected { reason: FeaQueryError },
}

impl From<Result<FeaBrowsePage, FeaQueryError>> for FeaBrowseOutcome {
    fn from(result: Result<FeaBrowsePage, FeaQueryError>) -> Self {
        match result {
            Ok(page) => Self::Page {
                page: Box::new(page),
            },
            Err(reason) => Self::Rejected { reason },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum FeaQueryError {
    #[error("checked source revision has changed")]
    StaleRevision,
    #[error("query limit must be between 1 and 100")]
    InvalidLimit,
    #[error("query cursor is outside this result")]
    InvalidCursor,
    #[error("object is not present in this checked source")]
    NotFound,
    #[error("requested definitions exceed the 64 KiB response budget; narrow the query")]
    ResponseLimit,
    #[error("this declaration has no resolved definition; inspect source diagnostics or query its owner")]
    DefinitionUnavailable,
    #[error("geometry selection must contain 1 to 256 distinct bounded region identities and request navigation only")]
    InvalidRegionSelection,
    #[error("a selected region does not have the required geometry role")]
    IncompatibleRegionRole,
}

fn matches(entry: &FeaSourceIndexEntry, request: &FeaBrowseRequest) -> bool {
    let module = match &entry.identity {
        FeaSourceIdentity::Declaration { module_path, .. }
        | FeaSourceIdentity::ModelMember { module_path, .. }
        | FeaSourceIdentity::StudyMember { module_path, .. }
        | FeaSourceIdentity::Function { module_path, .. } => module_path,
    };
    if request
        .module_path
        .as_ref()
        .is_some_and(|path| path != module)
    {
        return false;
    }
    match (&request.filter, &entry.identity) {
        (FeaQueryFilter::All, _) => true,
        (FeaQueryFilter::Identity { identity }, actual) => identity == actual,
        (FeaQueryFilter::GeometryRegionSelection { geometry, .. }, actual) => geometry == actual,
        (FeaQueryFilter::Declaration { kind }, FeaSourceIdentity::Declaration { symbol, .. }) => {
            *kind == symbol.kind
        }
        (
            FeaQueryFilter::ModelMember { kind },
            FeaSourceIdentity::ModelMember { member_kind, .. },
        ) => kind == member_kind,
        (
            FeaQueryFilter::StudyMember { kind },
            FeaSourceIdentity::StudyMember { member_kind, .. },
        ) => kind == member_kind,
        (FeaQueryFilter::Function, FeaSourceIdentity::Function { .. }) => true,
        _ => false,
    }
}

impl FeaSourceCheckResult {
    pub fn browse(&self, request: &FeaBrowseRequest) -> Result<FeaBrowsePage, FeaQueryError> {
        let revision = self.revision;
        if request.revision != revision
            || request
                .after
                .as_ref()
                .is_some_and(|cursor| cursor.revision != revision)
        {
            return Err(FeaQueryError::StaleRevision);
        }
        if request.after.as_ref().is_some_and(|cursor| {
            cursor.filter != request.filter
                || cursor.module_path != request.module_path
                || cursor.detail != request.detail
        }) {
            return Err(FeaQueryError::InvalidCursor);
        }
        if !(1..=100).contains(&request.limit) {
            return Err(FeaQueryError::InvalidLimit);
        }
        self.validate_region_query(request)?;
        let offset = request
            .after
            .as_ref()
            .map_or(0, |cursor| cursor.offset as usize);
        let total = self
            .source_index
            .entries
            .iter()
            .filter(|entry| matches(entry, request))
            .count();
        if offset > total {
            return Err(FeaQueryError::InvalidCursor);
        }
        let entries = self
            .source_index
            .entries
            .iter()
            .filter(|entry| matches(entry, request))
            .skip(offset)
            .take(request.limit as usize)
            .cloned()
            .collect::<Vec<_>>();
        let next_offset = offset + entries.len();
        let total = u32::try_from(total).map_err(|_| FeaQueryError::InvalidCursor)?;
        let details = match request.detail {
            FeaQueryDetail::Navigation => Vec::new(),
            FeaQueryDetail::Definition => entries
                .iter()
                .map(|entry| {
                    self.definition(&entry.identity).map_err(|error| {
                        if error == FeaQueryError::NotFound {
                            // The source index already proves the declaration exists.
                            // Failed semantic lowering must not masquerade as a name mismatch.
                            FeaQueryError::DefinitionUnavailable
                        } else {
                            error
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        let page = FeaBrowsePage {
            revision,
            entries,
            total,
            details,
            next: (next_offset < total as usize).then_some(FeaQueryCursor {
                revision,
                offset: next_offset as u32,
                filter: request.filter.clone(),
                module_path: request.module_path.clone(),
                detail: request.detail,
            }),
        };
        if serde_json::to_vec(&page)
            .map_err(|_| FeaQueryError::ResponseLimit)?
            .len()
            > 64 * 1024
        {
            return Err(FeaQueryError::ResponseLimit);
        }
        Ok(page)
    }

    pub fn material(
        &self,
        revision: SourceDigest,
        module: &LogicalSourcePath,
        id: &str,
    ) -> Result<&FeaDeclaredMaterial, FeaQueryError> {
        if self.revision != revision {
            return Err(FeaQueryError::StaleRevision);
        }
        self.program
            .materials
            .iter()
            .find(|value| &value.module_path == module && value.material.material_id == id)
            .ok_or(FeaQueryError::NotFound)
    }
}
