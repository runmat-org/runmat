import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

// Source changes are checked by Cargo. Compiler, dependency and build-policy
// changes select a separate checkpoint family before restoring workspace output.
export function nativeBuildCacheKey({ compiler, target, inputs, platform, architecture, environment = {} }) {
  if (!compiler || !target || !platform || !architecture || !/^[a-f0-9]{64}$/.test(inputs)) {
    throw new Error("Native cache identity requires compiler, target, platform and input digest");
  }
  const digest = createHash("sha256").update(JSON.stringify({
    compiler, target, inputs, platform, architecture,
    environment: Object.entries(environment).sort(([left], [right]) => left.localeCompare(right)),
  })).digest("hex");
  return `native-checkpoint-v1-${digest}`;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const compiler = execFileSync("rustc", ["-vV"], { encoding: "utf8" });
  const keys = ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_INCREMENTAL",
    "CARGO_PROFILE_DEV_DEBUG", "CARGO_PROFILE_DEV_SPLIT_DEBUGINFO", "CARGO_PROFILE_RELEASE_DEBUG",
    "CARGO_PROFILE_RELEASE_STRIP", "MACOSX_DEPLOYMENT_TARGET", "CMAKE_OSX_DEPLOYMENT_TARGET",
    "CC", "CXX", "CFLAGS", "CXXFLAGS"];
  console.log(`key=${nativeBuildCacheKey({ compiler,
    target: process.env.RUNMAT_CACHE_TARGET, inputs: process.env.RUNMAT_CACHE_INPUTS,
    platform: process.platform, architecture: process.arch,
    environment: Object.fromEntries(keys.map(key => [key, process.env[key] || ""])),
  })}`);
}
