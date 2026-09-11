// @ts-check

export const NATIVE_EMBEDDED_AOT_PROBE = "native-embedded-aot-compile-and-execute-v1";

export function requiredArtifactRoles(product, profile = defaultArtifactProfile(product)) {
    if (product === "native-cli" && profile === "embedded-aot") return ["runmat-binary"];
    if (product === "browser-wasm" && profile === "web") return ["wasm-js", "wasm-binary"];
    throw new Error(`Unknown artifact profile ${profile} for ${product}`);
}

export function requiredProductProbes(product, profile = defaultArtifactProfile(product)) {
    if (product === "native-cli" && profile === "embedded-aot") return [NATIVE_EMBEDDED_AOT_PROBE];
    if (product === "browser-wasm" && profile === "web") return [];
    throw new Error(`Unknown artifact profile ${profile} for ${product}`);
}

export function defaultArtifactProfile(product) {
    if (product === "native-cli") return "embedded-aot";
    if (product === "browser-wasm") return "web";
    throw new Error(`Unknown builtin example product: ${product}`);
}
