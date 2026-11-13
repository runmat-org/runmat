//! Canonical physical constants. Values retain the solver's existing numerical
//! convention; semantic kinds prevent mixing electric and magnetic coefficients.
use super::{ElectricPermittivity, MagneticPermeability};
pub const VACUUM_ELECTRIC_PERMITTIVITY: ElectricPermittivity =
    ElectricPermittivity::new(8.854_187_818_8e-12);
pub const VACUUM_MAGNETIC_PERMEABILITY: MagneticPermeability =
    MagneticPermeability::new(1.256_637_061_27e-6);
