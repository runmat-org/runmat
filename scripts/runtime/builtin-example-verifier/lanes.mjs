// @ts-check

export function usesBrowserLane(harness) {
    return harness === "LegacyBrowser"
        || harness === "Portable"
        || harness === "Browser"
        || harness === "BrowserGraphics"
        || harness === "Wgpu";
}

export function usesNativeLane(harness) {
    return harness === "Portable"
        || harness === "Native"
        || harness === "NativeFilesystem"
        || harness === "NativeForeignRuntime";
}

export function mergeLaneResults(cases, browserResults, nativeResults, unsupportedCases) {
    const browser = new Map(browserResults.map((result) => [result.id, result]));
    const native = new Map(nativeResults.map((result) => [result.id, result]));
    const unsupported = new Set(unsupportedCases.map((testCase) => testCase.id));
    return cases.map((testCase) => {
        if (unsupported.has(testCase.id)) {
            return missingResult(testCase.id, `no verifier adapter is available for the ${testCase.harness} harness`);
        }
        const browserResult = browser.get(testCase.id);
        const nativeResult = native.get(testCase.id);
        if (!browserResult && !nativeResult) {
            return missingResult(testCase.id, `the ${testCase.harness} harness produced no lane result`);
        }
        if (!browserResult) return nativeResult;
        if (!nativeResult) return browserResult;
        const expectedIdentifier = expectedErrorIdentifier(testCase.verification);
        if (expectedIdentifier) {
            if (browserResult.errorIdentifier === expectedIdentifier && nativeResult.errorIdentifier === expectedIdentifier) {
                return browserResult;
            }
            return mismatch(testCase.id, browserResult, nativeResult, "expected error identifiers differ across portable lanes");
        }
        if (browserResult.errorText || nativeResult.errorText) {
            return mismatch(testCase.id, browserResult, nativeResult, "portable example failed in at least one execution lane");
        }
        return browserResult;
    });
}

function expectedErrorIdentifier(verification) {
    if (!verification || typeof verification !== "object" || !("ExpectedError" in verification)) return "";
    const expected = verification.ExpectedError;
    return expected && typeof expected === "object" && typeof expected.identifier === "string"
        ? expected.identifier
        : "";
}

function missingResult(id, errorText) {
    return { id, stdoutText: "", valueText: "", errorText, errorIdentifier: "" };
}

function mismatch(id, browser, native, reason) {
    const browserError = browser.errorText || "none";
    const nativeError = native.errorText || "none";
    return {
        id,
        stdoutText: browser.stdoutText,
        valueText: browser.valueText,
        errorText: `${reason}; browser=${browserError}; native=${nativeError}`,
        errorIdentifier: ""
    };
}
