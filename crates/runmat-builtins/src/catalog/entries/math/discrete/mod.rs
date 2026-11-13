mod factor;
mod factorial;
mod gcd;
mod isprime;
mod lcm;
mod primes;

pub use factor::*;
pub use factorial::*;
pub use gcd::*;
pub use isprime::*;
pub use lcm::*;
pub use primes::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &FACTOR_CATALOG_ENTRY,
    &FACTORIAL_CATALOG_ENTRY,
    &GCD_CATALOG_ENTRY,
    &ISPRIME_CATALOG_ENTRY,
    &LCM_CATALOG_ENTRY,
    &PRIMES_CATALOG_ENTRY,
];
