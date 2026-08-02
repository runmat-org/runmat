mod filter;
mod merge;
mod model;

pub use filter::{CoverageFilter, CoveragePathClass};
pub use merge::{merge_aggregates, merge_coverage, CoverageMergeError};
pub use model::{
    CoverageAggregate, CoverageBackend, CoverageBackendSupport, CoverageFileSummary,
    CoverageFragment, CoverageMetric, CoverageSite, CoverageSummary,
};
