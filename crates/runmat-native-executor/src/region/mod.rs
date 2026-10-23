mod placement;
mod plan;

#[cfg(test)]
pub(crate) use placement::test_support as placement_test_support;
pub(crate) use placement::{choose_region, RegionObservation, RegionPlacement};
pub(crate) use plan::{
    derive_plans, OptimizedRegionPlan, OptimizedRegionPlanLease, RegionOutputSource, SiteIdentity,
};
mod slots;
pub(crate) use slots::{NumericRegionInputViews, NumericRegionReferences};
