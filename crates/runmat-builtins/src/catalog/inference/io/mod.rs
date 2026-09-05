mod change_directory;
mod clear_console;
mod current_directory;

pub(in crate::catalog::inference) use change_directory::infer as change_directory;
pub(in crate::catalog::inference) use clear_console::infer as clear_console;
pub(in crate::catalog::inference) use current_directory::infer as current_directory;
