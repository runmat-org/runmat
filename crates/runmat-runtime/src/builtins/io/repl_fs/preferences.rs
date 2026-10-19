//! Preference controls over one application-owned settings service.

mod controls;
pub(crate) mod getpref;
pub(crate) mod ispref;
mod preparation;
mod service;
pub(crate) mod setpref;

#[cfg(test)]
pub(super) use getpref::getpref_builtin;
#[cfg(test)]
pub(super) use ispref::ispref_builtin;
#[cfg(test)]
pub(super) use setpref::setpref_builtin;
