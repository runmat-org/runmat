//! Sound static facts for the encoding family. Unknown class lists and
//! resident controls stay dynamic; they are not guessed from argument shape.
mod common;
mod dummyvar;
mod encode;
mod decode;
#[cfg(test)]
#[path = "inference/tests.rs"]
mod tests;
pub(super) use decode::decode;
pub(super) use dummyvar::dummyvar;
pub(super) use encode::encode;
