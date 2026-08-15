//! Process-image runtime host linked into standalone RunMat programs.

mod execute;
mod input;
mod output;
mod program;

pub use input::AotProcessInput;

pub const EXIT_SUCCESS: i32 = 0;
pub const EXIT_INVALID_PROGRAM: i32 = 64;
pub const EXIT_RUNTIME_FAILURE: i32 = 70;

/// C entry called by the launcher in a verified RunMat user object.
///
/// All pointers originate from immutable symbols in that same linked object.
/// The boundary copies and validates each bounded payload before decoding it;
/// no borrowed process-image bytes escape this call.
///
/// # Safety
///
/// `function_resolver` must address a linked resolver with the exact
/// `AotFunctionResolver` ABI. Every non-null address it returns must identify a
/// function with Runtime's exact `NativeEntryPoint` ABI. Each payload pointer
/// must address an immutable allocation of its corresponding declared length
/// for the duration of this call. When `argc` is positive, `argv` must be a
/// valid C argument vector.
#[no_mangle]
pub unsafe extern "C" fn runmat_aot_main(
    argc: i32,
    argv: *const *const std::ffi::c_char,
    function_resolver: *const std::ffi::c_void,
    native_ir: *const u8,
    native_ir_len: u64,
    program: *const u8,
    program_len: u64,
    resume_points: *const u8,
    resume_points_len: u64,
) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: validation and bounded copies happen before any payload is
        // decoded or retained. The generated launcher supplies these symbols.
        let linked = input::LinkedProcessImage {
            argc,
            argv,
            function_resolver,
            native_ir,
            native_ir_len,
            program,
            program_len,
            resume_points,
            resume_points_len,
        };
        // SAFETY: the raw pointers were checked and converted to bounded borrows
        // above; the function resolver contract is documented on this entry.
        let input = unsafe { AotProcessInput::copy_from_linked(linked) }?;
        execute::execute(input)
    }));
    match result {
        Ok(Ok(())) => EXIT_SUCCESS,
        Ok(Err(error)) => {
            output::error(&error);
            EXIT_INVALID_PROGRAM
        }
        Err(_) => {
            output::error("standalone runtime aborted after an internal panic");
            EXIT_RUNTIME_FAILURE
        }
    }
}
