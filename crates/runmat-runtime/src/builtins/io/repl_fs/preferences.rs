//! Preference controls over one application-owned settings service.

mod controls;
mod getpref;
mod ispref;
mod preparation;
mod service;
mod setpref;

#[cfg(test)]
pub(super) use getpref::getpref_builtin;
#[cfg(test)]
pub(super) use ispref::ispref_builtin;
#[cfg(test)]
pub(super) use setpref::setpref_builtin;
