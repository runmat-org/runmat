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
        || harness === "NativeLoopbackNetwork"
        || harness === "NativeForeignRuntime"
        || harness === "InteractiveHost";
}

export const EXECUTION_LANES = Object.freeze([
    "browser-host",
    "browser-graphics",
    "browser-wgpu",
    "native-host",
    "native-filesystem",
    "native-loopback-network",
    "native-foreign-runtime",
    "interactive-host",
    "desktop-host"
]);

const HARNESS_LANES = Object.freeze({
    LegacyBrowser: ["browser-host"],
    Portable: ["native-host", "browser-host"],
    Native: ["native-host"],
    Browser: ["browser-host"],
    BrowserGraphics: ["browser-graphics"],
    NativeFilesystem: ["native-filesystem"],
    NativeLoopbackNetwork: ["native-loopback-network"],
    Wgpu: ["browser-wgpu"],
    NativeForeignRuntime: ["native-foreign-runtime"],
    InteractiveHost: ["interactive-host"]
});

export function requiredExecutionLanes(harness, hostRequirement = "Any") {
    if (harness === "InteractiveHost" && hostRequirement === "DesktopHostOnly") return ["desktop-host"];
    const lanes = HARNESS_LANES[harness];
    if (!lanes) throw new Error(`Unknown builtin example harness: ${harness}`);
    return [...lanes];
}

export function isExecutionLane(lane) {
    return EXECUTION_LANES.includes(lane);
}

export function laneProduct(lane) {
    if (!isExecutionLane(lane)) throw new Error(`Unknown builtin example execution lane: ${lane}`);
    if (lane.startsWith("browser-")) return "browser-wasm";
    if (lane === "desktop-host") return "desktop-native";
    return "native-cli";
}

export function laneAdapterAvailability(lane) {
    if (!isExecutionLane(lane)) throw new Error(`Unknown builtin example execution lane: ${lane}`);
    return { available: true, reason: "" };
}

export function laneEnablesGpu(lane) {
    if (!isExecutionLane(lane)) throw new Error(`Unknown builtin example execution lane: ${lane}`);
    return lane === "browser-wgpu";
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
