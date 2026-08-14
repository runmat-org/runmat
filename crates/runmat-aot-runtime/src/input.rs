const MAX_NATIVE_IR_BYTES: usize = 512 * 1024 * 1024;
const MAX_PROGRAM_BYTES: usize = 512 * 1024 * 1024;
const MAX_RESUME_POINT_BYTES: usize = 64 * 1024 * 1024;
const MAX_ARGUMENTS: i32 = 16_384;

pub struct AotProcessInput {
    pub entrypoint: runmat_runtime::native::NativeEntryPoint,
    pub native_ir: Vec<u8>,
    pub program: Vec<u8>,
    pub resume_points: Vec<u8>,
}

impl AotProcessInput {
    /// Copy process-image inputs into bounded owned storage.
    ///
    /// # Safety
    ///
    /// `entrypoint` must have Runtime's exact native entry ABI. Each non-null
    /// payload pointer must address at least its declared length of readable
    /// immutable bytes. A positive `argc` requires a valid `argv` pointer.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn copy_from_linked(
        argc: i32,
        argv: *const *const std::ffi::c_char,
        entrypoint: *const std::ffi::c_void,
        native_ir: *const u8,
        native_ir_len: u64,
        program: *const u8,
        program_len: u64,
        resume_points: *const u8,
        resume_points_len: u64,
    ) -> Result<Self, String> {
        if !(0..=MAX_ARGUMENTS).contains(&argc)
            || (argc > 0 && argv.is_null())
            || entrypoint.is_null()
        {
            return Err("standalone launcher arguments are invalid".into());
        }
        let native_ir =
            unsafe { copy_bounded(native_ir, native_ir_len, MAX_NATIVE_IR_BYTES, "Native IR") }?;
        let program = unsafe { copy_bounded(program, program_len, MAX_PROGRAM_BYTES, "program") }?;
        let resume_points = unsafe {
            copy_bounded(
                resume_points,
                resume_points_len,
                MAX_RESUME_POINT_BYTES,
                "resume-point",
            )
        }?;
        // SAFETY: the pointer is non-null and the generated object declares
        // `runmat_aot_entry` with Runtime's exact NativeEntryPoint signature.
        let entrypoint = unsafe {
            std::mem::transmute::<*const std::ffi::c_void, runmat_runtime::native::NativeEntryPoint>(
                entrypoint,
            )
        };
        Ok(Self {
            entrypoint,
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
    use super::AotProcessInput;

    #[test]
    fn raw_boundary_rejects_null_and_oversized_payloads_before_reading() {
        let one = [1_u8];
        let invalid = unsafe {
            AotProcessInput::copy_from_linked(
                0,
                std::ptr::null(),
                std::ptr::dangling(),
                std::ptr::null(),
                1,
                one.as_ptr(),
                1,
                one.as_ptr(),
                1,
            )
        };
        assert!(invalid.is_err());

        let oversized = unsafe {
            AotProcessInput::copy_from_linked(
                0,
                std::ptr::null(),
                std::ptr::dangling(),
                one.as_ptr(),
                (512_u64 * 1024 * 1024) + 1,
                one.as_ptr(),
                1,
                one.as_ptr(),
                1,
            )
        };
        assert!(oversized.is_err());
    }
}
