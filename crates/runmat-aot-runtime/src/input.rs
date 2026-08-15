const MAX_NATIVE_IR_BYTES: usize = 512 * 1024 * 1024;
const MAX_PROGRAM_BYTES: usize = 512 * 1024 * 1024;
const MAX_RESUME_POINT_BYTES: usize = 64 * 1024 * 1024;
const MAX_ARGUMENTS: i32 = 16_384;
pub type AotFunctionResolver = unsafe extern "C" fn(function: u32) -> *const std::ffi::c_void;

/// Borrowed symbols passed across the generated launcher's C ABI boundary.
pub struct LinkedProcessImage {
    pub argc: i32,
    pub argv: *const *const std::ffi::c_char,
    pub function_resolver: *const std::ffi::c_void,
    pub native_ir: *const u8,
    pub native_ir_len: u64,
    pub program: *const u8,
    pub program_len: u64,
    pub resume_points: *const u8,
    pub resume_points_len: u64,
}

pub struct AotProcessInput {
    pub function_resolver: AotFunctionResolver,
    pub native_ir: Vec<u8>,
    pub program: Vec<u8>,
    pub resume_points: Vec<u8>,
}

impl AotProcessInput {
    /// Copy process-image inputs into bounded owned storage.
    ///
    /// # Safety
    ///
    /// `function_resolver` must address a linked resolver with the exact
    /// `AotFunctionResolver` ABI. Each non-null payload pointer must address at
    /// least its declared length of readable immutable bytes. A positive
    /// `argc` requires a valid `argv` pointer.
    pub unsafe fn copy_from_linked(input: LinkedProcessImage) -> Result<Self, String> {
        if !(0..=MAX_ARGUMENTS).contains(&input.argc)
            || (input.argc > 0 && input.argv.is_null())
            || input.function_resolver.is_null()
        {
            return Err("standalone launcher arguments are invalid".into());
        }
        // SAFETY: the generated launcher supplies the address of the resolver
        // emitted with the exact AotFunctionResolver signature.
        let function_resolver = unsafe {
            std::mem::transmute::<*const std::ffi::c_void, AotFunctionResolver>(
                input.function_resolver,
            )
        };
        let native_ir = unsafe {
            copy_bounded(
                input.native_ir,
                input.native_ir_len,
                MAX_NATIVE_IR_BYTES,
                "Native IR",
            )
        }?;
        let program = unsafe {
            copy_bounded(
                input.program,
                input.program_len,
                MAX_PROGRAM_BYTES,
                "program",
            )
        }?;
        let resume_points = unsafe {
            copy_bounded(
                input.resume_points,
                input.resume_points_len,
                MAX_RESUME_POINT_BYTES,
                "resume-point",
            )
        }?;
        Ok(Self {
            function_resolver,
            native_ir,
            program,
            resume_points,
        })
    }
}

unsafe fn copy_bounded(
    pointer: *const u8,
    length: u64,
    maximum: usize,
    label: &str,
) -> Result<Vec<u8>, String> {
    let length = usize::try_from(length)
        .map_err(|_| format!("standalone {label} payload length exceeds this host"))?;
    if pointer.is_null() || length == 0 || length > maximum {
        return Err(format!(
            "standalone {label} payload is absent or exceeds its bound"
        ));
    }
    // SAFETY: the generated launcher passes an immutable linked data symbol
    // with exactly this validated length. The bytes are copied immediately.
    Ok(unsafe { std::slice::from_raw_parts(pointer, length) }.to_vec())
}

#[cfg(test)]
mod tests {
    use super::{AotProcessInput, LinkedProcessImage};

    #[test]
    fn raw_boundary_rejects_null_and_oversized_payloads_before_reading() {
        let one = [1_u8];
        let invalid = unsafe {
            AotProcessInput::copy_from_linked(LinkedProcessImage {
                argc: 0,
                argv: std::ptr::null(),
                function_resolver: std::ptr::null(),
                native_ir: std::ptr::null(),
                native_ir_len: 1,
                program: one.as_ptr(),
                program_len: 1,
                resume_points: one.as_ptr(),
                resume_points_len: 1,
            })
        };
        assert!(invalid.is_err());

        let oversized = unsafe {
            AotProcessInput::copy_from_linked(LinkedProcessImage {
                argc: 0,
                argv: std::ptr::null(),
                function_resolver: std::ptr::dangling(),
                native_ir: one.as_ptr(),
                native_ir_len: (512_u64 * 1024 * 1024) + 1,
                program: one.as_ptr(),
                program_len: 1,
                resume_points: one.as_ptr(),
                resume_points_len: 1,
            })
        };
        assert!(oversized.is_err());
    }
}
