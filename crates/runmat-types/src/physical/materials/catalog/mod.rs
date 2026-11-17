//! The library's materials, one file per category. Polymers, composites, ceramics, electronics,
//! construction materials and fluids are empty until the MCAD and ECAD material lists fold in.
use super::{Material, MaterialCategory};

pub mod ceramics;
pub mod composites;
pub mod construction;
pub mod electronics;
pub mod fluids;
pub mod metals;
pub mod polymers;

/// Every category's rows, in the order a card offers them.
pub const CATEGORIES: &[(MaterialCategory, &[Material])] = &[
    (MaterialCategory::Metal, metals::MATERIALS),
    (MaterialCategory::Polymer, polymers::MATERIALS),
    (MaterialCategory::Composite, composites::MATERIALS),
    (MaterialCategory::Ceramic, ceramics::MATERIALS),
    (MaterialCategory::Electronic, electronics::MATERIALS),
    (MaterialCategory::Construction, construction::MATERIALS),
    (MaterialCategory::Fluid, fluids::MATERIALS),
];
