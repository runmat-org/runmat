mod activation;
mod conv;
mod dense;
mod normalization;
mod numeric;
mod pooling;
mod recurrent;
mod softmax;

pub(in crate::builtins::deep_learning) mod network;

pub(in crate::builtins::deep_learning) use activation::Activation;
pub(in crate::builtins::deep_learning) use conv::ConvOptions;
pub(in crate::builtins::deep_learning) use normalization::NormalizationOptions;
pub(in crate::builtins::deep_learning) use recurrent::RecurrentOptions;
