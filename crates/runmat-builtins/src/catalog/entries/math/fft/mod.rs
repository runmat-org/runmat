// FFT family topology only; semantic contracts live in identity leaves.
mod inference;
#[cfg(test)]
mod tests;
#[path = "fft/mod.rs"]
mod forward;
mod fft2;
mod fftn;
mod ifft;
mod ifft2;
mod ifftn;
mod fftshift;
mod ifftshift;

pub use self::fft2::*;
pub use self::fftn::*;
pub use self::fftshift::*;
pub use self::forward::*;
pub use self::ifft::*;
pub use self::ifft2::*;
pub use self::ifftn::*;
pub use self::ifftshift::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(forward::ENTRIES.iter().copied());
    values.extend(fft2::ENTRIES.iter().copied());
    values.extend(fftn::ENTRIES.iter().copied());
    values.extend(ifft::ENTRIES.iter().copied());
    values.extend(ifft2::ENTRIES.iter().copied());
    values.extend(ifftn::ENTRIES.iter().copied());
    values.extend(fftshift::ENTRIES.iter().copied());
    values.extend(ifftshift::ENTRIES.iter().copied());
}
