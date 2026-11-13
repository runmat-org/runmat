//! Source-file columns and selected output columns have different identities.
//! Detected metadata may itself begin at a nonzero source column.

#[derive(Clone, Copy, Debug)]
pub(crate) struct ImportColumnProjection {
    metadata_source_start: Option<usize>,
    selected_source_start: usize,
}

impl ImportColumnProjection {
    pub(crate) fn new(metadata_source_start: Option<usize>, selected_source_start: usize) -> Self {
        Self {
            metadata_source_start,
            selected_source_start,
        }
    }

    /// Legacy options can contain metadata for only an explicitly selected
    /// range, without a recorded source origin. A matching declared width
    /// identifies that layout; wider metadata stays in source-file order.
    pub(crate) fn with_selected_width(
        mut self,
        metadata_count: usize,
        declared_selected_width: Option<usize>,
    ) -> Self {
        if self.metadata_source_start.is_none()
            && self.selected_source_start > 0
            && declared_selected_width == Some(metadata_count)
        {
            self.metadata_source_start = Some(self.selected_source_start);
        }
        self
    }

    pub(crate) fn metadata_index_for_source(self, source_column: usize) -> Option<usize> {
        source_column.checked_sub(self.metadata_source_start.unwrap_or(0))
    }

    pub(crate) fn source_for_output(self, output_column: usize) -> Option<usize> {
        self.selected_source_start.checked_add(output_column)
    }

    pub(crate) fn metadata_index_for_output(self, output_column: usize) -> Option<usize> {
        self.metadata_index_for_source(self.source_for_output(output_column)?)
    }
}
