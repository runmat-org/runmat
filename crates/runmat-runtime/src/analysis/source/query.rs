pub use runmat_fea_source::{
    FeaBrowseOutcome, FeaBrowsePage, FeaBrowseRequest, FeaQueryCursor, FeaQueryDetail,
    FeaQueryError, FeaQueryFilter,
};
#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_language::{FeaSymbolKind, LogicalSourcePath, SourceDigest};
    #[test]
    fn bounded_navigation_preserves_source_locations_and_rejects_stale_or_reused_cursors() {
        let checked = super::super::FeaSourceSnapshotBuilder::new(
            std::path::Path::new("/project"),
            super::super::FeaSourceSnapshotBuilder::source_module(
                LogicalSourcePath::new("library.fea").unwrap(),
                b"model first {}\nmodel second {}\nmodel third {}\n",
            ),
        )
        .unwrap()
        .prepare()
        .unwrap()
        .check(vec![], vec![])
        .unwrap()
        .check_result();
        let revision = checked.revision;
        let mut request = FeaBrowseRequest {
            revision,
            module_path: None,
            filter: FeaQueryFilter::Declaration {
                kind: FeaSymbolKind::Model,
            },
            after: None,
            limit: 2,
            detail: FeaQueryDetail::Navigation,
        };
        let first = checked.browse(&request).unwrap();
        assert_eq!(first.total, 3);
        assert_eq!(first.entries.len(), 2);
        assert!(first
            .entries
            .iter()
            .all(|entry| entry.location.path == checked.root_module));
        request.after = first.next;
        let second = checked.browse(&request).unwrap();
        assert_eq!(second.entries.len(), 1);
        assert!(second.next.is_none());
        request.filter = FeaQueryFilter::All;
        assert_eq!(checked.browse(&request), Err(FeaQueryError::InvalidCursor));
        request.revision = SourceDigest([0; 32]);
        assert_eq!(checked.browse(&request), Err(FeaQueryError::StaleRevision));
        request.revision = revision;
        request.after = None;
        request.limit = 101;
        assert_eq!(checked.browse(&request), Err(FeaQueryError::InvalidLimit));
    }
}
