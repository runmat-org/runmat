//! One in-memory preference owner for query, mutation, and host consumers.

mod store;

pub(crate) use store::{
    all_groups, group_values, has_group, has_preference, read_many, read_or_create,
    session_pref_text, set_many, MissingPreference,
};
