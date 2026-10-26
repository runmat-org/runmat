//! Target linker spelling for closed-world unused-section removal.
use runmat_execution::host::NativeOperatingSystem;

use super::LinkerFamily;

pub(super) fn argument(family: LinkerFamily, system: NativeOperatingSystem) -> &'static str {
    match (family, system) {
        (LinkerFamily::Msvc, _) => "/OPT:REF",
        (LinkerFamily::UnixCc, NativeOperatingSystem::Macos) => "-Wl,-dead_strip",
        (LinkerFamily::UnixCc, NativeOperatingSystem::Linux | NativeOperatingSystem::Windows) => {
            "-Wl,--gc-sections"
        }
    }
}
