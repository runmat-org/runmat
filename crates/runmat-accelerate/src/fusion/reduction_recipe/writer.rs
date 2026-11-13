//! Count and emit the same recipe text with one exact admitted allocation.
use anyhow::{ensure, Result};
use runmat_execution::resource::ProviderHostMetadataAdmission;
use std::fmt;

use crate::fusion::wgsl_source::CountedWgslSource;
use crate::wgsl::host_buffers::ClaimedString;

pub(super) fn render(
    metadata: Option<&dyn ProviderHostMetadataAdmission>,
    write: impl Fn(&mut dyn fmt::Write) -> Result<()>,
) -> Result<ClaimedString> {
    let mut count = CountedWgslSource::with_preamble_bytes(0);
    write(&mut count)?;
    let limit = count.bytes();
    let mut source = ClaimedString::with_capacity(limit, metadata)?;
    struct Fixed<'a> {
        source: &'a mut ClaimedString,
        limit: usize,
    }
    impl fmt::Write for Fixed<'_> {
        fn write_str(&mut self, text: &str) -> fmt::Result {
            if self
                .source
                .len()
                .checked_add(text.len())
                .is_none_or(|n| n > self.limit)
            {
                return Err(fmt::Error);
            }
            // The count already admitted this exact backing; growth is forbidden.
            self.source.push_str(text, None).map_err(|_| fmt::Error)
        }
    }
    write(&mut Fixed {
        source: &mut source,
        limit,
    })?;
    ensure!(
        source.len() == limit,
        "reduction source changed between count and emission"
    );
    Ok(source)
}
