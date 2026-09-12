use once_cell::sync::Lazy;
use std::sync::Mutex;

pub static REPL_FS_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
