//! Image color and format conversion builtins.

pub(crate) mod common;
pub(crate) mod color_presentation;
pub(crate) mod conversion_mode;
pub mod gray2rgb;
pub mod hsv2rgb;
pub mod im2double;
pub mod im2uint16;
pub mod im2uint8;
pub mod ind2rgb;
pub mod lab2rgb;
pub mod rgb2gray;
pub mod rgb2hsv;
pub mod rgb2lab;
pub(crate) mod rgb_input;
pub(crate) mod rgb_presentation;
pub(crate) mod resident_owner;

#[cfg(test)]
mod preparation_tests;
