use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpmdHeader<T> {
    Default,
    One(T),
    Two(T, T),
    Three(T, T, T),
}
