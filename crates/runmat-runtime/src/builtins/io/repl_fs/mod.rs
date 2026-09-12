//! REPL-facing filesystem builtins.

pub mod addpath;
pub mod cd;
pub mod compat;
pub mod delete;
pub(crate) mod directory_lifecycle;
pub(crate) mod directory_listing;
pub(crate) mod environment;
pub mod exist;
pub(crate) mod file_dialog;
pub(crate) mod file_transfer;
pub mod genpath;
pub(crate) mod installation_path;
pub mod open;
pub mod opentoline;
pub mod path;
mod path_list;
mod path_mutation;
pub(crate) mod path_predicate;
pub(crate) mod path_root;
pub(crate) mod path_syntax;
pub mod pcode;
pub mod pwd;
pub mod rmpath;
pub mod run;
pub mod savepath;
pub(crate) mod source_inventory;
pub(crate) mod temporary_path;
pub(crate) mod test_support;
pub(crate) mod text_conversion;
pub mod uigetdir;
pub mod uigetfile;
pub mod uiputfile;
mod working_directory;
pub mod xml;

pub use test_support::REPL_FS_TEST_LOCK;
