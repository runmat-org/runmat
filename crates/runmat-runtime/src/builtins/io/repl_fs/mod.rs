//! REPL-facing filesystem builtins.

pub mod addpath;
pub mod cd;
pub mod copyfile;
pub mod delete;
pub mod dir;
pub mod exist;
pub mod genpath;
pub mod getenv;
pub mod ls;
pub mod mkdir;
pub mod movefile;
pub mod path;
pub mod pwd;
pub mod rmdir;
pub mod rmpath;
pub mod savepath;
pub mod setenv;
pub mod tempdir;
pub mod tempname;
pub mod which;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

#[cfg(test)]
pub(crate) static REPL_FS_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
