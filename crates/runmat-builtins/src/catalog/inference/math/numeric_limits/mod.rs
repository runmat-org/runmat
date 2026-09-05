//! Static inference for numeric-limit constructors.

mod call;
mod output;

pub(in crate::catalog::inference) use call::infer;

#[cfg(test)]
mod tests;
