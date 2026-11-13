//! Public warm-start data contract; numerical state remains the actual solver's.
mod descriptor;
mod documentation;
mod entry;
mod inference;
pub use descriptor::*;
pub use entry::*;
pub const LSQLIN_WARM_START_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("LsqlinWarmStart");
pub const QUADPROG_WARM_START_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("QuadprogWarmStart");
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimWarmStartProperty {
    X,
    Options,
    MaxLinearEqualities,
    MaxLinearInequalities,
}
impl OptimWarmStartProperty {
    pub const ALL: [Self; 4] = [
        Self::X,
        Self::Options,
        Self::MaxLinearEqualities,
        Self::MaxLinearInequalities,
    ];
    pub const fn name(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Options => "Options",
            Self::MaxLinearEqualities => "MaxLinearEqualities",
            Self::MaxLinearInequalities => "MaxLinearInequalities",
        }
    }
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }
    pub fn parse_argument(name: &str) -> Option<Self> {
        [Self::MaxLinearEqualities, Self::MaxLinearInequalities]
            .into_iter()
            .find(|p| p.name().eq_ignore_ascii_case(name))
    }
}
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &OPTIMWARMSTART_CATALOG_ENTRY,
    &LSQLIN_WARM_SUBSREF_CATALOG_ENTRY,
    &LSQLIN_WARM_SUBSASGN_CATALOG_ENTRY,
    &QUADPROG_WARM_SUBSREF_CATALOG_ENTRY,
    &QUADPROG_WARM_SUBSASGN_CATALOG_ENTRY,
];
