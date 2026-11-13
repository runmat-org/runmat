//! Process-image runtime host linked into standalone RunMat programs.

mod accelerator_profile;
mod builtin;
mod clock_service;
mod execute;
mod external_service;
mod foreign_policy;
mod foreign_retirement;
mod input;
mod java_artifacts;
mod local_service;
mod mex;
mod mex_artifacts;
mod native_interfaces;
mod network_service;
mod output;
mod process_mode;
mod program;
mod python_artifacts;
#[cfg(feature = "source-evaluation")]
mod source_service;
mod target_test;

/// Owned embedding input remains host-unselected; it cannot mint a selected
/// linked-image grant from decoded bytes.
pub use execute::execute;
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
/// `function_resolver` and `builtin_resolver` must address linked resolvers with
/// the exact `AotFunctionResolver` and `AotBuiltinResolver` ABIs. Every non-null
/// function address must identify Runtime's exact `NativeEntryPoint` ABI, and
/// every non-null builtin address must identify an immutable
/// `RuntimeBuiltinBinding`. A non-null `source_service_factory` must identify
/// the optional factory symbol from the exact verified linked runtime archive.
/// A non-null `network_service_factory` likewise identifies the exact linked
/// network factory, never a decoded capability claim.
/// Each payload pointer must address an immutable
/// allocation of its corresponding declared length for the duration of this
/// call. When `argc` is positive, `argv` must be a valid C argument vector.
#[no_mangle]
pub unsafe extern "C" fn runmat_aot_main(
    argc: i32,
    argv: *const *const std::ffi::c_char,
    function_resolver: *const std::ffi::c_void,
    builtin_resolver: *const std::ffi::c_void,
    source_service_factory: *const std::ffi::c_void,
    network_service_factory: *const std::ffi::c_void,
    ui_service_factory: *const std::ffi::c_void,
    callback_service_factory: *const std::ffi::c_void,
    dynamic_service_factory: *const std::ffi::c_void,
    native_ir: *const u8,
    native_ir_len: u64,
    program: *const u8,
    program_len: u64,
    resume_points: *const u8,
    resume_points_len: u64,
    native_interfaces: *const u8,
    native_interfaces_len: u64,
    mex_artifacts: *const u8,
    mex_artifacts_len: u64,
    java_artifacts: *const u8,
    java_artifacts_len: u64,
    python_artifacts: *const u8,
    python_artifacts_len: u64,
) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the generated launcher forwards the process argument vector
        // received by its C `main` entrypoint.
        let arguments = unsafe { input::copy_process_arguments(argc, argv) }?;
        if process_mode::run_if_requested(arguments)? {
            return Ok(());
        }
        // SAFETY: validation and bounded copies happen before any payload is
        // decoded or retained. The generated launcher supplies these symbols.
        let linked = input::LinkedProcessImage {
            argc,
            argv,
            function_resolver,
            builtin_resolver,
            source_service_factory,
            network_service_factory,
            external_factories: input::LinkedExternalFactories {
                ui: ui_service_factory,
                callback: callback_service_factory,
                dynamic: dynamic_service_factory,
            },
            native_ir,
            native_ir_len,
            program,
            program_len,
            resume_points,
            resume_points_len,
            native_interfaces,
            native_interfaces_len,
            mex_artifacts,
            mex_artifacts_len,
            java_artifacts,
            java_artifacts_len,
            python_artifacts,
            python_artifacts_len,
        };
        // SAFETY: the raw pointers were checked and converted to bounded borrows
        // above; both resolver contracts are documented on this entry.
        let input = unsafe { AotProcessInput::copy_from_linked(linked) }?;
        // SAFETY: this entry is only called by the generated launcher in the
        // qualified linked image described by this function's safety contract.
        let linked = unsafe {
            input::LinkedAotProcessInput::from_verified_link(
                input,
                source_service_factory,
                network_service_factory,
                input::LinkedExternalFactories {
                    ui: ui_service_factory,
                    callback: callback_service_factory,
                    dynamic: dynamic_service_factory,
                },
            )
        };
        execute::execute_linked(linked)
    }));
    match result {
        Ok(Ok(())) => EXIT_SUCCESS,
        Ok(Err(error)) => {
            output::error(&error.to_string());
            error.exit_code()
        }
        Err(_) => {
            output::error("standalone runtime aborted after an internal panic");
            EXIT_RUNTIME_FAILURE
        }
    }
}

/// Distinct generated test-image entry. Its immutable paired header is an
/// additional linked symbol; stdin carries exactly one bounded framed request
/// and stdout carries exactly one framed receiver result.
///
/// # Safety
/// The ordinary linked-image arguments have the same contracts as `runmat_aot_main`.
/// `paired_header` points to immutable linked bytes of `paired_header_len` for
/// the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn runmat_aot_test_main(
    argc: i32,
    argv: *const *const std::ffi::c_char,
    function_resolver: *const std::ffi::c_void,
    builtin_resolver: *const std::ffi::c_void,
    source_service_factory: *const std::ffi::c_void,
    network_service_factory: *const std::ffi::c_void,
    ui_service_factory: *const std::ffi::c_void,
    callback_service_factory: *const std::ffi::c_void,
    dynamic_service_factory: *const std::ffi::c_void,
    native_ir: *const u8,
    native_ir_len: u64,
    program: *const u8,
    program_len: u64,
    resume_points: *const u8,
    resume_points_len: u64,
    native_interfaces: *const u8,
    native_interfaces_len: u64,
    mex_artifacts: *const u8,
    mex_artifacts_len: u64,
    java_artifacts: *const u8,
    java_artifacts_len: u64,
    python_artifacts: *const u8,
    python_artifacts_len: u64,
    paired_header: *const u8,
    paired_header_len: u64,
) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the generated test launcher forwards its C main arguments.
        let arguments = unsafe { input::copy_process_arguments(argc, argv) }?;
        if process_mode::run_if_requested(arguments)? {
            return Ok(());
        }
        let image = input::LinkedProcessImage {
            argc,
            argv,
            function_resolver,
            builtin_resolver,
            source_service_factory,
            network_service_factory,
            external_factories: input::LinkedExternalFactories {
                ui: ui_service_factory,
                callback: callback_service_factory,
                dynamic: dynamic_service_factory,
            },
            native_ir,
            native_ir_len,
            program,
            program_len,
            resume_points,
            resume_points_len,
            native_interfaces,
            native_interfaces_len,
            mex_artifacts,
            mex_artifacts_len,
            java_artifacts,
            java_artifacts_len,
            python_artifacts,
            python_artifacts_len,
        };
        // SAFETY: the generated launcher supplies the same verified linked
        // symbols as the ordinary entry plus its immutable paired header.
        let input = unsafe { AotProcessInput::copy_from_linked(image) }?;
        let header_bytes = unsafe {
            input::copy_bounded(
                paired_header,
                paired_header_len,
                runmat_execution_artifact::target_test::MAX_NATIVE_TARGET_TEST_WIRE_BYTES,
                "paired header",
            )
        }?;
        let header = runmat_execution_artifact::target_test::NativePairedProductHeader::from_canonical_bytes(&header_bytes)
            .map_err(|error| format!("paired test header is invalid: {error}"))?;
        // SAFETY: this call is available only from the qualified generated
        // test launcher and retains the selected link provenance.
        let linked = unsafe {
            input::LinkedAotProcessInput::from_verified_link(
                input,
                source_service_factory,
                network_service_factory,
                input::LinkedExternalFactories {
                    ui: ui_service_factory,
                    callback: callback_service_factory,
                    dynamic: dynamic_service_factory,
                },
            )
        };
        target_test::run(linked, header).map_err(execute::ProcessFailure::from)
    }));
    match result {
        Ok(Ok(())) => EXIT_SUCCESS,
        Ok(Err(error)) => {
            output::error(&error.to_string());
            error.exit_code()
        }
        Err(_) => {
            output::error("standalone target-test runtime aborted after an internal panic");
            EXIT_RUNTIME_FAILURE
        }
    }
}
