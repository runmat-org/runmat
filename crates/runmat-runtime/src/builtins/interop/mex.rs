//! MATLAB-compatible MEX target introspection.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "extension",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Platform-specific MEX filename extension without a leading dot.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "extension = mexext",
    inputs: &[],
    outputs: &OUTPUTS,
}];

const BUILD_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Native source files and compatible MEX build options.",
}];

const BUILD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mex(arguments)",
    inputs: &BUILD_INPUTS,
    outputs: &[],
}];

const CUDA_BUILD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "mexcuda(arguments)",
    inputs: &BUILD_INPUTS,
    outputs: &[],
}];

const ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEX.UNAVAILABLE_TARGET",
    identifier: Some("RunMat:MEX:UnsupportedTarget"),
    when: "The current target cannot load native MEX modules.",
    message: "MEX modules are unavailable on this target.",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_UNAVAILABLE];

const ERROR_BUILD_ARGUMENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEX.INVALID_BUILD_ARGUMENTS",
    identifier: Some("RunMat:MEX:InvalidBuildArguments"),
    when: "The MEX command receives no source file or an invalid build option.",
    message: "mex: invalid build arguments.",
};

const ERROR_BUILD_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEX.BUILD_FAILED",
    identifier: Some("RunMat:MEX:BuildFailed"),
    when: "The selected compiler cannot produce the requested MEX module.",
    message: "mex: C MEX compilation failed.",
};

const BUILD_ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_UNAVAILABLE, ERROR_BUILD_ARGUMENTS, ERROR_BUILD_FAILED];

pub const MEXEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const MEXEXT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "mexext accepts no inputs and returns a platform extension string.",
};

pub const MEX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BUILD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BUILD_ERRORS,
};

pub const MEX_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "mex accepts source paths and compiler options; it does not operate on numeric values.",
};

pub const MEXCUDA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CUDA_BUILD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BUILD_ERRORS,
};

pub const MEXCUDA_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes:
        "mexcuda accepts source paths and compiler options; it does not operate on numeric values.",
};

#[runtime_builtin(
    name = "mex",
    category = "interop/mex",
    summary = "Build MATLAB-compatible C MEX source into a RunMat-loadable module.",
    keywords = "mex,compile,c,native,extension,build",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::interop::mex::MEX_DESCRIPTOR),
    integer_audit(crate::builtins::interop::mex::MEX_INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::mex"
)]
fn mex_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    build_mex(arguments, "mex", false)
}

#[runtime_builtin(
    name = "mexcuda",
    category = "interop/mex",
    summary = "Build a CUDA MEX module with the configured nvcc toolchain.",
    keywords = "mexcuda,mex,cuda,gpu,native,extension,build",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::interop::mex::MEXCUDA_DESCRIPTOR),
    integer_audit(crate::builtins::interop::mex::MEXCUDA_INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::mex"
)]
fn mexcuda_builtin(arguments: Vec<Value>) -> BuiltinResult<Value> {
    build_mex(arguments, "mexcuda", true)
}

fn build_mex(arguments: Vec<Value>, builtin: &'static str, cuda: bool) -> BuiltinResult<Value> {
    let arguments = arguments
        .iter()
        .map(|argument| {
            String::try_from(argument).map_err(|error| {
                build_runtime_error(format!("{builtin}: expected text arguments: {error}"))
                    .with_builtin(builtin)
                    .with_identifier(
                        ERROR_BUILD_ARGUMENTS
                            .identifier
                            .expect("MEX argument error has an identifier"),
                    )
                    .build()
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let working_directory = runmat_filesystem::current_dir().map_err(|error| {
        mex_build_error(
            builtin,
            &ERROR_BUILD_FAILED,
            format!("{builtin}: could not resolve the current directory: {error}"),
        )
    })?;
    let invocation = if cuda {
        runmat_mex::MexBuildInvocation::parse_cuda(&arguments, &working_directory)
    } else {
        runmat_mex::MexBuildInvocation::parse(&arguments, &working_directory)
    }
    .map_err(|error| mex_build_error(builtin, &ERROR_BUILD_ARGUMENTS, error.to_string()))?;
    if invocation.verbose {
        let command = invocation
            .build
            .plan()
            .map_err(|error| map_mex_build_error(builtin, error))?
            .command()
            .join(" ");
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout, command);
    }
    let output = invocation
        .compile()
        .map_err(|error| map_mex_build_error(builtin, error))?;
    crate::console::record_console_line(
        crate::console::ConsoleStream::Stdout,
        format!("Built {}", output.module.display()),
    );
    Ok(Value::OutputList(Vec::new()))
}

fn map_mex_build_error(
    builtin: &'static str,
    error: runmat_mex::MexBuildError,
) -> crate::RuntimeError {
    match error {
        runmat_mex::MexBuildError::UnsupportedTarget
        | runmat_mex::MexBuildError::UnsupportedCudaTarget => {
            mex_build_error(builtin, &ERROR_UNAVAILABLE, error.to_string())
        }
        other => mex_build_error(builtin, &ERROR_BUILD_FAILED, other.to_string()),
    }
}

fn mex_build_error(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    message: String,
) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(
            descriptor
                .identifier
                .expect("MEX build error has an identifier"),
        )
        .build()
}

#[runtime_builtin(
    name = "mexext",
    category = "interop/mex",
    summary = "Return the MEX filename extension for the current native platform.",
    keywords = "mexext,mex,native,extension,platform",
    descriptor(crate::builtins::interop::mex::MEXEXT_DESCRIPTOR),
    integer_audit(crate::builtins::interop::mex::MEXEXT_INTEGER_AUDIT),
    builtin_path = "crate::builtins::interop::mex"
)]
fn mexext_builtin() -> BuiltinResult<Value> {
    runmat_mex::mex_suffix().map(Value::from).ok_or_else(|| {
        build_runtime_error(ERROR_UNAVAILABLE.message)
            .with_builtin("mexext")
            .with_identifier(
                ERROR_UNAVAILABLE
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
        let error = mex_builtin(Vec::new()).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:MEX:InvalidBuildArguments"));
    }

    #[test]
    fn mexcuda_requires_a_cuda_translation_unit() {
        let error = mexcuda_builtin(vec![Value::from("gateway.cpp")]).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:MEX:InvalidBuildArguments"));
    }

    #[cfg(not(target_family = "wasm"))]
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
        let result = mex_builtin(arguments).unwrap();
        assert_eq!(result, Value::OutputList(Vec::new()));
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
