mod two_parameter;

pub(super) use two_parameter::{infer_binornd, infer_gamrnd, infer_wblrnd};

#[cfg(test)]
mod tests;
