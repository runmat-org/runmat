mod descriptor;
mod inference;
mod policy;
#[cfg(test)]
mod test_support;
mod normpdf;
mod normcdf;
mod norminv;
mod tpdf;
mod tcdf;
mod tinv;
mod binocdf;
mod chi2cdf;
mod wblinv;
mod icdf;

pub use binocdf::*;
pub use chi2cdf::*;
pub use descriptor::*;
pub use icdf::*;
pub use normcdf::*;
pub use norminv::*;
pub use normpdf::*;
pub use policy::*;
pub use tcdf::*;
pub use tinv::*;
pub use tpdf::*;
pub use wblinv::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(normpdf::ENTRIES.iter().copied());
    values.extend(normcdf::ENTRIES.iter().copied());
    values.extend(norminv::ENTRIES.iter().copied());
    values.extend(tpdf::ENTRIES.iter().copied());
    values.extend(tcdf::ENTRIES.iter().copied());
    values.extend(tinv::ENTRIES.iter().copied());
    values.extend(binocdf::ENTRIES.iter().copied());
    values.extend(chi2cdf::ENTRIES.iter().copied());
    values.extend(wblinv::ENTRIES.iter().copied());
    values.extend(icdf::ENTRIES.iter().copied());
}
