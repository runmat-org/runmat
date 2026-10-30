//! Native MEX build and target builtin adapters.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

pub use runmat_builtins::{
    MEXCUDA_DESCRIPTOR, MEXEXT_DESCRIPTOR, MEX_DESCRIPTOR, MEX_ERROR_BUILD_ARGUMENTS,
    MEX_ERROR_BUILD_FAILED, MEX_ERROR_OUTPUTS, MEX_ERROR_UNAVAILABLE,
};

#[runtime_builtin(
    name = "mex",
    binding_variant = "default",
    builtin_path = "crate::builtins::interop::mex"
)]
fn mex_builtin(arguments: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    build_mex(arguments, "mex", false)
}

#[runtime_builtin(
    name = "mexcuda",
    binding_variant = "default",
    builtin_path = "crate::builtins::interop::mex"
)]
fn mexcuda_builtin(arguments: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    build_mex(arguments, "mexcuda", true)
}

fn build_mex(
    arguments: crate::call::BuiltinArguments,
    builtin: &'static str,
    cuda: bool,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        if cuda {
            &MEXCUDA_DESCRIPTOR
        } else {
            &MEX_DESCRIPTOR
        },
        &MEX_ERROR_OUTPUTS,
        builtin,
    )?;
    if arguments.is_empty() {
        return Err(mex_build_error_with_source(
            builtin,
            &MEX_ERROR_BUILD_ARGUMENTS,
            runmat_mex::MexArgumentError::MissingSource,
        ));
    }
    let arguments = arguments
        .iter()
        .map(|argument| {
            String::try_from(argument).map_err(|error| {
                build_runtime_error(format!("{builtin}: expected text arguments: {error}"))
                    .with_builtin(builtin)
                    .with_identifier(
                        MEX_ERROR_BUILD_ARGUMENTS
                            .identifier
                            .expect("MEX argument error has an identifier"),
                    )
                    .build()
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    // A native target is a prerequisite to touching CWD, sources, or the SDK.
    // The compiler adapter still validates the exact target and artifact at
    // plan/publication time; this preflight only fixes unsupported-host order.
    let target =
        runmat_mex::MexTarget::current().map_err(|error| map_mex_build_error(builtin, error))?;
    if cuda {
        target
            .validate_cuda_build()
            .map_err(|error| map_mex_build_error(builtin, error))?;
    }
    let working_directory = runmat_filesystem::current_dir().map_err(|error| {
        build_runtime_error(format!(
            "{builtin}: could not resolve the current directory: {error}"
        ))
        .with_builtin(builtin)
        .with_identifier(
            MEX_ERROR_BUILD_FAILED
                .identifier
                .expect("MEX build identifier"),
        )
        .with_source(error)
        .build()
    })?;
    let invocation = if cuda {
        runmat_mex::MexBuildInvocation::parse_cuda(&arguments, &working_directory)
    } else {
        runmat_mex::MexBuildInvocation::parse(&arguments, &working_directory)
    }
    .map_err(|error| mex_build_error_with_source(builtin, &MEX_ERROR_BUILD_ARGUMENTS, error))?;
    if invocation.verbose {
        let command = invocation
            .build
            .plan()
            .map_err(|error| map_mex_build_error(builtin, error))?
            .command()
            .join(" ");
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout, command)?;
    }
    let output = invocation
        .compile()
        .map_err(|error| map_mex_build_error(builtin, error))?;
    crate::console::record_console_line(
        crate::console::ConsoleStream::Stdout,
        format!("Built {}", output.module.display()),
    )?;
    Ok(Value::output_list(Vec::new()))
}

fn map_mex_build_error(
    builtin: &'static str,
    error: runmat_mex::MexBuildError,
) -> crate::RuntimeError {
    match error {
        error @ (runmat_mex::MexBuildError::UnsupportedTarget
        | runmat_mex::MexBuildError::UnsupportedCudaTarget
        | runmat_mex::MexBuildError::UnsupportedTargetIdentity { .. }) => {
            mex_build_error_with_source(builtin, &MEX_ERROR_UNAVAILABLE, error)
        }
        other => mex_build_error_with_source(builtin, &MEX_ERROR_BUILD_FAILED, other),
    }
}

fn mex_build_error_with_source<E: crate::runtime_error::RuntimeErrorSourceBound>(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    source: E,
) -> crate::RuntimeError {
    build_runtime_error(source.to_string())
        .with_builtin(builtin)
        .with_identifier(descriptor.identifier.expect("MEX error identifier"))
        .with_source(source)
        .build()
}

#[runtime_builtin(
    name = "mexext",
    binding_variant = "default",
    builtin_path = "crate::builtins::interop::mex"
)]
fn mexext_builtin() -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(&MEXEXT_DESCRIPTOR, &MEX_ERROR_OUTPUTS, "mexext")?;
    runmat_mex::mex_suffix().map(Value::from).ok_or_else(|| {
        build_runtime_error(MEX_ERROR_UNAVAILABLE.message)
            .with_builtin("mexext")
            .with_identifier(
                MEX_ERROR_UNAVAILABLE
                    .identifier
                    .expect("MEX unavailable error has an identifier"),
            )
            .build()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mexext_matches_the_adapter_target_suffix() {
        match runmat_mex::mex_suffix() {
            Some(expected) => assert_eq!(mexext_builtin().unwrap(), Value::from(expected)),
            None => assert_eq!(
                mexext_builtin()
                    .unwrap_err()
                    .identifier()
                    .expect("identifier"),
                "RunMat:MEX:UnsupportedTarget"
            ),
        }
    }

    #[test]
    fn mex_reports_missing_sources_as_an_argument_error() {
        let error = mex_builtin(crate::builtins::common::test_support::builtin_arguments(
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:MEX:InvalidBuildArguments"));
        assert!(error.source.is_some());
    }

    #[test]
    fn mexcuda_invalid_argument_class_precedes_target_check() {
        let error = mexcuda_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::output_list(Vec::new())],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:MEX:InvalidBuildArguments"));
    }

    #[test]
    fn requested_build_output_is_rejected_before_argument_or_filesystem_work() {
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let error = mex_builtin(crate::builtins::common::test_support::builtin_arguments(
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:MEX:TooManyOutputs"));
    }

    #[test]
    fn mexext_rejects_excess_outputs_before_target_lookup() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = mexext_builtin().unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:MEX:TooManyOutputs"));
    }

    #[test]
    fn build_error_keeps_typed_source_at_target_or_parser_boundary() {
        let error = mexcuda_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::from("gateway.cpp")],
        ))
        .unwrap_err();
        let expected = if runmat_mex::MexTarget::current()
            .and_then(|target| target.validate_cuda_build())
            .is_ok()
        {
            "RunMat:MEX:InvalidBuildArguments"
        } else {
            "RunMat:MEX:UnsupportedTarget"
        };
        assert_eq!(error.identifier(), Some(expected));
        assert!(error.source.is_some());
    }

    #[test]
    fn scalar_numeric_and_logical_build_arguments_still_convert_to_text() {
        assert_eq!(String::try_from(&Value::Num(17.0)).unwrap(), "17");
        assert_eq!(String::try_from(&Value::Bool(true)).unwrap(), "true");
        assert_eq!(
            String::try_from(&Value::Int(runmat_value::IntValue::U64(u64::MAX))).unwrap(),
            u64::MAX.to_string()
        );
    }

    #[cfg(target_family = "wasm")]
    #[test]
    fn valid_build_text_rejects_unsupported_target_before_source_access() {
        let error = mex_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::from("missing.c")],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:MEX:UnsupportedTarget"));
        assert!(error.source.is_some());
        let cuda_error = mexcuda_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::from("missing.cu")],
        ))
        .unwrap_err();
        assert_eq!(
            cuda_error.identifier(),
            Some("RunMat:MEX:UnsupportedTarget")
        );
        assert!(cuda_error.source.is_some());
    }

    #[cfg(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64")
    ))]
    #[test]
    fn mexcuda_requires_a_cuda_translation_unit_on_supported_targets() {
        let error = mexcuda_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::from("gateway.cpp")],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:MEX:InvalidBuildArguments"));
        assert!(error.source.is_some());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mexcuda_rejects_unsupported_native_target_before_source_access() {
        let error = mexcuda_builtin(crate::builtins::common::test_support::builtin_arguments(
            vec![Value::from("missing.cu")],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:MEX:UnsupportedTarget"));
        assert!(error.source.is_some());
    }

    #[cfg(any(
        all(
            target_os = "macos",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64")
    ))]
    #[test]
    fn mex_builds_source_with_the_shared_adapter_service() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("runtime_build_fixture.c");
        std::fs::write(
            &source,
            r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
}
"#,
        )
        .unwrap();
        let arguments = vec![
            Value::from("-outdir"),
            Value::from(directory.path().display().to_string()),
            Value::from("-output"),
            Value::from("runtime_build_fixture"),
            Value::from(source.display().to_string()),
        ];
        let result = mex_builtin(crate::builtins::common::test_support::builtin_arguments(
            arguments,
        ))
        .unwrap();
        assert_eq!(result, Value::output_list(Vec::new()));
        assert!(directory
            .path()
            .join(format!(
                "runtime_build_fixture.{}",
                runmat_mex::mex_suffix().unwrap()
            ))
            .is_file());
    }

    #[test]
    fn mexcuda_reports_an_unsupported_host_as_unavailable() {
        let error =
            map_mex_build_error("mexcuda", runmat_mex::MexBuildError::UnsupportedCudaTarget);
        assert_eq!(
            error.identifier().expect("identifier"),
            "RunMat:MEX:UnsupportedTarget"
        );
    }
}
