mod error;
pub(crate) mod matlabroot;
mod root;

pub(crate) use root::configured_factory_root;
pub(crate) use root::resolve as root;
