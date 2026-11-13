import { describe, expect, it, vi } from "vitest";
import { listNativeProjectEntrypoints, prepareNativeNamedClusterProgram, runNativeProjectEntrypoint } from "./project-entrypoints.js";

describe("named project entrypoint owner delegation", () => {
  it("keeps project/name, exact byte arguments, existing session and bundle owners", async () => {
    const selection = { projectPath: "/workspace/runmat.toml", name: "server" };
    const options = { argumentsJson: new TextEncoder().encode("[]"), requestedOutputs: 2 };
    const response = { kind: "typedFunction" as const, result: new Uint8Array([91, 93]) };
    const native = {
      listProjectEntrypoints: vi.fn(async () => []),
      runProjectEntrypoint: vi.fn(async () => response),
      prepareNamedClusterProgram: vi.fn(async () => ({ bundleArchive: new Uint8Array([1]), descriptor: new Uint8Array([2]) })),
    };
    expect(await listNativeProjectEntrypoints(native, selection.projectPath)).toEqual([]);
    expect(await runNativeProjectEntrypoint(native, selection, options)).toBe(response);
    expect(native.runProjectEntrypoint).toHaveBeenCalledWith(selection.projectPath, selection.name, options);
    const prepared = await prepareNativeNamedClusterProgram(native, selection);
    expect(native.prepareNamedClusterProgram).toHaveBeenCalledWith(selection.projectPath, selection.name, undefined);
    expect(prepared.bundleArchive).toEqual(new Uint8Array([1]));
    expect(() => runNativeProjectEntrypoint(native, selection, { requestedOutputs: -1 })).toThrow(RangeError);
    expect(native.runProjectEntrypoint).toHaveBeenCalledTimes(1);
    expect(() => listNativeProjectEntrypoints({}, selection.projectPath)).toThrow("Reload");
    expect(() => runNativeProjectEntrypoint({}, selection)).toThrow("Reload");
    expect(() => prepareNativeNamedClusterProgram({}, selection)).toThrow("Reload");
    const inputManifest = { snapshotId: "captured", entries: [] };
    await listNativeProjectEntrypoints(native, selection.projectPath, { userSettingsJson: "{}", inputManifest });
    expect(native.listProjectEntrypoints).toHaveBeenLastCalledWith(selection.projectPath, { userSettingsJson: "{}", inputManifest });
    await prepareNativeNamedClusterProgram(native, { ...selection, inputManifest });
    expect(native.prepareNamedClusterProgram).toHaveBeenLastCalledWith(selection.projectPath, selection.name, inputManifest);
    await runNativeProjectEntrypoint(native, { ...selection, inputManifest }, options);
    expect(native.runProjectEntrypoint).toHaveBeenLastCalledWith(selection.projectPath, selection.name, { ...options, inputManifest });
    expect(() => runNativeProjectEntrypoint(native, { ...selection, inputManifest }, { inputManifest })).toThrow(TypeError);
  });
});
