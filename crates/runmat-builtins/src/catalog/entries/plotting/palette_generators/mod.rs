use crate::BuiltinCatalogEntry;
mod shared;
pub use shared::{MAX_PALETTE_LENGTH, PALETTE_INVALID_ARGUMENT};
#[cfg(test)]
mod tests;
mod autumn;
mod bone;
mod colorcube;
mod cool;
mod copper;
mod gray;
mod hot;
mod inferno;
mod jet;
mod lines;
mod magma;
mod parula;
mod pink;
mod plasma;
mod spring;
mod summer;
mod turbo;
mod viridis;
mod winter;

pub use autumn::AUTUMN_CATALOG_ENTRY;
pub use bone::BONE_CATALOG_ENTRY;
pub use colorcube::COLORCUBE_CATALOG_ENTRY;
pub use cool::COOL_CATALOG_ENTRY;
pub use copper::COPPER_CATALOG_ENTRY;
pub use gray::GRAY_CATALOG_ENTRY;
pub use hot::HOT_CATALOG_ENTRY;
pub use inferno::INFERNO_CATALOG_ENTRY;
pub use jet::JET_CATALOG_ENTRY;
pub use lines::LINES_CATALOG_ENTRY;
pub use magma::MAGMA_CATALOG_ENTRY;
pub use parula::PARULA_CATALOG_ENTRY;
pub use pink::PINK_CATALOG_ENTRY;
pub use plasma::PLASMA_CATALOG_ENTRY;
pub use spring::SPRING_CATALOG_ENTRY;
pub use summer::SUMMER_CATALOG_ENTRY;
pub use turbo::TURBO_CATALOG_ENTRY;
pub use viridis::VIRIDIS_CATALOG_ENTRY;
pub use winter::WINTER_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &autumn::AUTUMN_CATALOG_ENTRY,
    &bone::BONE_CATALOG_ENTRY,
    &colorcube::COLORCUBE_CATALOG_ENTRY,
    &cool::COOL_CATALOG_ENTRY,
    &copper::COPPER_CATALOG_ENTRY,
    &gray::GRAY_CATALOG_ENTRY,
    &hot::HOT_CATALOG_ENTRY,
    &inferno::INFERNO_CATALOG_ENTRY,
    &jet::JET_CATALOG_ENTRY,
    &lines::LINES_CATALOG_ENTRY,
    &magma::MAGMA_CATALOG_ENTRY,
    &parula::PARULA_CATALOG_ENTRY,
    &pink::PINK_CATALOG_ENTRY,
    &plasma::PLASMA_CATALOG_ENTRY,
    &spring::SPRING_CATALOG_ENTRY,
    &summer::SUMMER_CATALOG_ENTRY,
    &turbo::TURBO_CATALOG_ENTRY,
    &viridis::VIRIDIS_CATALOG_ENTRY,
    &winter::WINTER_CATALOG_ENTRY,
];
