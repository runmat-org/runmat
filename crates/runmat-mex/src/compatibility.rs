//! Source-compatibility catalog for the C, Fortran, and GPU Matrix and MEX APIs.
//!
//! This catalog is the adapter's public support contract. It intentionally
//! excludes the C++ Data API, engine API, MAT-file API, and undocumented
//! entrypoints; those belong to separate compatibility tiers.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MexApiAvailability {
    R2017b,
    R2018a,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MexApiSymbol {
    pub name: &'static str,
    pub availability: MexApiAvailability,
}

macro_rules! api_symbols {
    ($($availability:ident => [$($name:literal),+ $(,)?]),+ $(,)?) => {
        &[$($(MexApiSymbol {
            name: $name,
            availability: MexApiAvailability::$availability,
        }),+),+]
    };
}

pub const C_MATRIX_API: &[MexApiSymbol] = api_symbols! {
    R2017b => [
        "mxIsNumeric", "mxIsComplex", "mxGetNumberOfDimensions",
        "mxGetElementSize", "mxGetDimensions", "mxSetDimensions",
        "mxGetNumberOfElements", "mxCalcSingleSubscript", "mxGetM", "mxSetM",
        "mxGetN", "mxSetN", "mxIsEmpty", "mxIsFromGlobalWS",
        "mxCreateDoubleMatrix", "mxCreateDoubleScalar", "mxCreateNumericMatrix",
        "mxCreateNumericArray", "mxCreateUninitNumericMatrix",
        "mxCreateUninitNumericArray", "mxIsScalar", "mxGetScalar", "mxIsDouble",
        "mxIsSingle", "mxGetPr", "mxSetPr", "mxIsInt8", "mxIsUint8", "mxIsInt16",
        "mxIsUint16", "mxIsInt32", "mxIsUint32", "mxIsInt64", "mxIsUint64",
        "mxGetPi", "mxSetPi", "mxGetImagData", "mxSetImagData", "mxCreateSparse",
        "mxCreateSparseLogicalMatrix", "mxIsSparse", "mxGetNzmax", "mxSetNzmax",
        "mxGetIr", "mxSetIr", "mxGetJc", "mxSetJc", "mxGetData", "mxSetData",
        "mxCreateString", "mxCreateCharMatrixFromStrings", "mxCreateCharArray",
        "mxIsChar", "mxGetChars", "mxIsLogical", "mxIsLogicalScalar",
        "mxIsLogicalScalarTrue", "mxCreateLogicalArray", "mxCreateLogicalMatrix",
        "mxCreateLogicalScalar", "mxGetLogicals", "mxIsClass", "mxGetClassID",
        "mxGetClassName", "mxSetClassName", "mxGetProperty", "mxSetProperty",
        "mxCreateStructMatrix", "mxCreateStructArray", "mxIsStruct", "mxGetField",
        "mxSetField", "mxGetNumberOfFields", "mxGetFieldNameByNumber",
        "mxGetFieldNumber", "mxGetFieldByNumber", "mxSetFieldByNumber", "mxAddField",
        "mxRemoveField", "mxCreateCellMatrix", "mxCreateCellArray", "mxIsCell",
        "mxGetCell", "mxSetCell", "mxDestroyArray", "mxDuplicateArray",
        "mxArrayToString", "mxArrayToUTF8String", "mxGetString", "mxCalloc",
        "mxMalloc", "mxRealloc", "mxFree", "mxAssert", "mxAssertS", "mxIsInf",
        "mxIsFinite", "mxIsNaN", "mxGetEps", "mxGetInf", "mxGetNaN"
    ],
    R2018a => [
        "mxGetDoubles", "mxSetDoubles", "mxGetSingles", "mxSetSingles",
        "mxGetInt8s", "mxSetInt8s", "mxGetUint8s", "mxSetUint8s", "mxGetInt16s",
        "mxSetInt16s", "mxGetUint16s", "mxSetUint16s", "mxGetInt32s",
        "mxSetInt32s", "mxGetUint32s", "mxSetUint32s", "mxGetInt64s",
        "mxSetInt64s", "mxGetUint64s", "mxSetUint64s", "mxGetComplexDoubles",
        "mxSetComplexDoubles", "mxGetComplexSingles", "mxSetComplexSingles",
        "mxGetComplexInt8s", "mxSetComplexInt8s", "mxGetComplexUint8s",
        "mxSetComplexUint8s", "mxGetComplexInt16s", "mxSetComplexInt16s",
        "mxGetComplexUint16s", "mxSetComplexUint16s", "mxGetComplexInt32s",
        "mxSetComplexInt32s", "mxGetComplexUint32s", "mxSetComplexUint32s",
        "mxGetComplexInt64s", "mxSetComplexInt64s", "mxGetComplexUint64s",
        "mxSetComplexUint64s", "mxMakeArrayComplex", "mxMakeArrayReal"
    ],
};

pub const C_MEX_API: &[MexApiSymbol] = api_symbols! {
    R2017b => [
        "mexFunction", "mexFunctionName", "mexAtExit", "mexCallMATLAB",
        "mexCallMATLABWithTrap", "mexEvalString", "mexEvalStringWithTrap",
        "mexGetVariable", "mexGetVariablePtr", "mexPutVariable", "mexGet", "mexSet",
        "mexPrintf", "mexErrMsgTxt", "mexErrMsgIdAndTxt", "mexWarnMsgTxt",
        "mexWarnMsgIdAndTxt", "mexIsLocked", "mexLock", "mexUnlock",
        "mexMakeArrayPersistent", "mexMakeMemoryPersistent", "mexIsGlobal"
    ],
};

pub const FORTRAN_MATRIX_API: &[MexApiSymbol] = api_symbols! {
    R2017b => [
        "mxCreateDoubleMatrix", "mxCreateDoubleScalar", "mxCreateNumericMatrix",
        "mxCreateNumericArray", "mxCreateLogicalMatrix", "mxCreateLogicalArray",
        "mxCreateLogicalScalar", "mxCreateCharArray", "mxCreateString",
        "mxCreateCellMatrix", "mxCreateCellArray", "mxCreateStructMatrix",
        "mxCreateStructArray", "mxCreateSparse", "mxCreateSparseLogicalMatrix",
        "mxDuplicateArray", "mxDestroyArray", "mxGetData", "mxSetData",
        "mxGetPr", "mxSetPr", "mxGetPi", "mxSetPi", "mxGetLogicals",
        "mxGetChars", "mxGetDimensions", "mxGetM", "mxGetN",
        "mxGetNumberOfElements", "mxGetNumberOfDimensions", "mxGetElementSize",
        "mxGetNzmax", "mxGetIr", "mxGetJc", "mxSetIr", "mxSetJc",
        "mxSetM", "mxSetN", "mxSetNzmax", "mxGetScalar", "mxGetClassID",
        "mxIsNumeric", "mxIsDouble", "mxIsSingle", "mxIsInt8", "mxIsUint8",
        "mxIsInt16", "mxIsUint16", "mxIsInt32", "mxIsUint32", "mxIsInt64",
        "mxIsUint64", "mxIsLogical", "mxIsLogicalScalar",
        "mxIsLogicalScalarTrue", "mxIsChar", "mxIsCell", "mxIsStruct",
        "mxIsSparse", "mxIsComplex", "mxIsEmpty", "mxGetCell", "mxSetCell",
        "mxGetNumberOfFields", "mxGetFieldNumber", "mxAddField", "mxRemoveField",
        "mxGetField", "mxSetField", "mxGetFieldByNumber", "mxSetFieldByNumber",
        "mxIsClass", "mxSetClassName", "mxGetProperty", "mxSetProperty",
        "mxGetString", "mxMalloc", "mxCalloc", "mxRealloc", "mxFree",
        "mxCopyPtrToReal8", "mxCopyReal8ToPtr", "mxCopyPtrToReal4",
        "mxCopyReal4ToPtr", "mxCopyPtrToInteger1", "mxCopyInteger1ToPtr",
        "mxCopyPtrToInteger2", "mxCopyInteger2ToPtr", "mxCopyPtrToInteger4",
        "mxCopyInteger4ToPtr", "mxCopyPtrToInteger8", "mxCopyInteger8ToPtr",
        "mxCopyPtrToComplex16", "mxCopyComplex16ToPtr", "mxCopyPtrToComplex8",
        "mxCopyComplex8ToPtr"
    ],
    R2018a => [
        "mxGetDoubles", "mxSetDoubles", "mxGetSingles", "mxSetSingles",
        "mxGetInt8s", "mxSetInt8s", "mxGetUint8s", "mxSetUint8s",
        "mxGetInt16s", "mxSetInt16s", "mxGetUint16s", "mxSetUint16s",
        "mxGetInt32s", "mxSetInt32s", "mxGetUint32s", "mxSetUint32s",
        "mxGetInt64s", "mxSetInt64s", "mxGetUint64s", "mxSetUint64s",
        "mxGetComplexDoubles", "mxSetComplexDoubles", "mxGetComplexSingles",
        "mxSetComplexSingles"
    ],
};

pub const FORTRAN_MEX_API: &[MexApiSymbol] = api_symbols! {
    R2017b => [
        "mexFunction", "mexAtExit", "mexCallMATLAB", "mexEvalString",
        "mexGetVariable", "mexPutVariable", "mexPrintf", "mexErrMsgTxt",
        "mexErrMsgIdAndTxt", "mexWarnMsgTxt", "mexWarnMsgIdAndTxt",
        "mexIsLocked", "mexLock", "mexUnlock", "mexMakeArrayPersistent",
        "mexMakeMemoryPersistent", "mexIsGlobal"
    ],
};

pub const GPU_MATRIX_API: &[MexApiSymbol] = api_symbols! {
    R2017b => [
        "mxInitGPU", "mxIsGPUArray", "mxGPUIsValidGPUData",
        "mxGPUCreateFromMxArray", "mxGPUCopyFromMxArray", "mxGPUCopyGPUArray",
        "mxGPUCreateGPUArray", "mxGPUCreateMxArrayOnGPU",
        "mxGPUCreateMxArrayOnCPU", "mxGPUDestroyGPUArray", "mxGPUGetClassID",
        "mxGPUGetComplexity", "mxGPUGetDimensions", "mxGPUGetNumberOfDimensions",
        "mxGPUGetNumberOfElements", "mxGPUGetData", "mxGPUGetDataReadOnly",
        "mxGPUIsSparse", "mxGPUIsSame", "mxGPUCopyReal", "mxGPUCopyImag",
        "mxGPUCreateComplexGPUArray"
    ],
    R2018a => [
        "mxGPUSetDimensions"
    ],
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn cataloged_symbols_are_unique_and_present_in_the_bundled_interface() {
        let matrix = include_str!("../include/matrix.h");
        let mex = include_str!("../include/mex.h");
        let support_source = concat!(
            include_str!("../src-c/runmat_mex_support.c"),
            include_str!("../src-c/sparse_index_compat.inc"),
            include_str!("../src-c/matrix_api.inc"),
            include_str!("../src-c/data_engine.inc"),
            include_str!("../src-c/mex_api.inc")
        );
        let mut names = BTreeSet::new();
        for symbol in C_MATRIX_API.iter().chain(C_MEX_API) {
            assert!(
                names.insert(symbol.name),
                "duplicate symbol {}",
                symbol.name
            );
            let header = if symbol.name.starts_with("mex") {
                mex
            } else {
                matrix
            };
            assert!(
                header.contains(symbol.name),
                "bundled interface does not declare {}",
                symbol.name
            );
            if !matches!(symbol.name, "mexFunction" | "mxAssert" | "mxAssertS") {
                assert!(
                    support_source.contains(symbol.name),
                    "native compatibility support does not implement {}",
                    symbol.name
                );
            }
        }
    }

    #[test]
    fn api_pins_form_an_additive_contract() {
        assert!(C_MATRIX_API.iter().any(|symbol| {
            symbol.name == "mxGetPr" && symbol.availability == MexApiAvailability::R2017b
        }));
        assert!(C_MATRIX_API.iter().any(|symbol| {
            symbol.name == "mxGetComplexUint64s"
                && symbol.availability == MexApiAvailability::R2018a
        }));
    }

    #[test]
    fn fortran_catalog_is_backed_by_the_bundled_header_and_abi_unit() {
        let header = include_str!("../include/fintrf.h");
        let implementation = include_str!("../src-c/fortran_api.inc").to_ascii_lowercase();
        let mut names = BTreeSet::new();
        for symbol in FORTRAN_MATRIX_API.iter().chain(FORTRAN_MEX_API) {
            assert!(
                names.insert(symbol.name),
                "duplicate Fortran symbol {}",
                symbol.name
            );
            assert!(
                header.contains(symbol.name)
                    || implementation.contains(&symbol.name.to_ascii_lowercase()),
                "bundled Fortran interface does not expose {}",
                symbol.name
            );
            if symbol.name != "mexFunction" {
                let generated_copy_family = symbol
                    .name
                    .strip_prefix("mxCopyPtrTo")
                    .or_else(|| {
                        symbol
                            .name
                            .strip_prefix("mxCopy")
                            .and_then(|name| name.strip_suffix("ToPtr"))
                    })
                    .is_some_and(|element| {
                        implementation.contains(&format!(
                            "runmat_fortran_copy_pair({},",
                            element.to_ascii_lowercase()
                        ))
                    });
                assert!(
                    implementation.contains(&symbol.name.to_ascii_lowercase())
                        || generated_copy_family,
                    "Fortran ABI unit does not implement {}",
                    symbol.name
                );
            }
        }
    }

    #[test]
    fn gpu_catalog_is_backed_by_the_public_header_and_native_support_unit() {
        let header = include_str!("../include/gpu/mxGPUArray.h");
        let implementation = include_str!("../src-c/gpu_api.inc");
        let mut names = BTreeSet::new();
        for symbol in GPU_MATRIX_API {
            assert!(
                names.insert(symbol.name),
                "duplicate GPU symbol {}",
                symbol.name
            );
            assert!(
                header.contains(symbol.name),
                "bundled GPU interface does not declare {}",
                symbol.name
            );
            assert!(
                implementation.contains(symbol.name),
                "native GPU support does not implement {}",
                symbol.name
            );
        }
    }
}
