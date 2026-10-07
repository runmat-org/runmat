mod lifecycle;
mod limits;
mod process_tree;
mod spawn;
mod startup;
mod stderr;

pub use lifecycle::{ChildProcess, ChildStdio, ProcessExit};
pub use limits::ResourceLimits;
pub use process_tree::{is_alive as is_process_alive, terminate_id as terminate_process_tree};
pub use spawn::spawn;
pub use startup::release_startup_stderr;
pub use stderr::{CapturedStderr, StderrCompletion, StderrSnapshot};
