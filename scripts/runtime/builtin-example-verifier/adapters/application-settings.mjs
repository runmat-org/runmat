import { mkdirSync } from "node:fs";
import { join } from "node:path";

/** Select only a fresh child-process application-settings authority. */
export function isolateApplicationSettings(caseDirectory, fixture, platform = process.platform) {
    if (!fixture || typeof fixture !== "object" || !("ApplicationSettings" in fixture)) {
        return { available: true, environment: {} };
    }
    if (platform === "win32") {
        // dirs::home_dir() on Windows is not governed solely by HOME. Until
        // the host exposes an explicit settings path, fail closed.
        return { available: false, reason: "the native Windows example host has no verified isolated application-settings location" };
    }
    const isolatedHome = join(caseDirectory, "application-home");
    mkdirSync(isolatedHome);
    return { available: true, environment: { HOME: isolatedHome } };
}
