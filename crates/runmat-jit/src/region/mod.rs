mod placement;
mod plan;

pub(crate) use placement::choose_vectorized;
pub(crate) use plan::{derive_plans, OptimizedRegionPlan, RegionOutputSource, SiteIdentity};
