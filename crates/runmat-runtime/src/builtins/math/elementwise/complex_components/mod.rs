//! Complex-component extraction and conjugation.

pub(crate) mod conj;
pub(crate) mod imag;
mod projection;
pub(crate) mod real;

pub(crate) use conj::conjugate_integer_imaginary_storage;
