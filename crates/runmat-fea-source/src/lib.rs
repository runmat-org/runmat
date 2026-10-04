//! Canonical checked FEA data and revision-fenced queries. This crate has no
//! filesystem provider, session, execution engine, or mutable workspace state.
pub use runmat_analysis_language::{LogicalSourcePath, SourceDigest};
mod changes;
mod inspection;
mod navigation;
mod object;
mod program;
mod query;
mod query_regions;
mod report;
mod workspace;
pub use changes::*;
pub use navigation::*;
pub use object::*;
pub use program::*;
pub use query::*;
pub use report::*;
pub use workspace::*;
