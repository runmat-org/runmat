//! Signal processing builtins.

pub(crate) mod blackman;
pub mod butter;
pub mod buttord;
pub(crate) mod common;
pub(crate) mod conv;
pub(crate) mod conv2;
pub mod deconv;
pub mod filter;
pub mod filtfilt;
pub mod fir1;
pub mod freqz;
pub(crate) mod gauspuls;
pub(crate) mod hamming;
pub(crate) mod hann;
pub(crate) mod hilbert;
pub(crate) mod pulstran;
pub(crate) mod pwelch;
pub(crate) mod rectpuls;
pub(crate) mod sample_rate;
pub(crate) mod sawtooth;
pub(crate) mod sinc;
pub(crate) mod square;
pub(crate) mod tripuls;
pub(crate) mod type_resolvers;
pub(crate) mod unwrap;
pub(crate) mod zplane;
