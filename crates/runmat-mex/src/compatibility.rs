//! Source-compatibility catalog for the C Matrix and C MEX APIs.
//!
//! This catalog is the adapter's public support contract. It intentionally
//! excludes the C++ Data API, engine API, MAT-file API, and undocumented
//! entrypoints; those belong to separate adapters or compatibility tiers.

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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn cataloged_symbols_are_unique_and_present_in_the_bundled_interface() {
        let matrix = include_str!("../include/matrix.h");
        let mex = include_str!("../include/mex.h");
        let shim = concat!(
            include_str!("../native/runmat_mex_shim.c"),
            include_str!("../native/data_engine.inc")
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
                    shim.contains(symbol.name),
                    "compatibility shim does not implement {}",
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
}
