import { describe, expect, it, vi } from "vitest";
import { MemoryVolume } from "./memory.js";

describe("committed mutation observers", () => {
  it("notifies synchronously and reports errors separately from successful replacement", async () => {
    const volume = new MemoryVolume({ initialFiles: { "/source": "new", "/target": "old" } });
    const failure = new Error("observer failed");
    const errors: unknown[] = [];
    let calls = 0;
    const fs = volume.createProvider(() => {
      calls += 1;
      expect(new TextDecoder().decode(volume.readFile("/target"))).toBe("new");
      throw failure;
    }, error => errors.push(error));
    expect(() => fs.renameReplace!("/source", "/target")).not.toThrow();
    expect(calls).toBe(1);
    expect(errors).toEqual([failure]);
    expect(new TextDecoder().decode(await fs.readFile("/target"))).toBe("new");
    expect(() => fs.metadata("/source")).toThrow();
  });

  it("reports observer and sink failures without changing committed filesystem results", () => {
    const observerFailure = new Error("observer failed");
    const sinkFailure = new Error("sink failed");
    const reporting = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const volume = new MemoryVolume({ initialFiles: { "/source": "new", "/target": "old" } });
      const defaultReporting = volume.createProvider(() => { throw observerFailure; });
      expect(() => defaultReporting.renameReplace!("/source", "/target")).not.toThrow();
      expect(reporting).toHaveBeenLastCalledWith(
        "RunMat filesystem observer failed after commit", { error: observerFailure, sinkError: undefined }
      );
      const customReporting = volume.createProvider(
        () => { throw observerFailure; }, () => { throw sinkFailure; }
      );
      expect(() => customReporting.writeFile("/written", new Uint8Array([7]))).not.toThrow();
      expect(reporting).toHaveBeenLastCalledWith(
        "RunMat filesystem observer failed after commit", { error: observerFailure, sinkError: sinkFailure }
      );
      reporting.mockImplementation(() => { throw new Error("host logger failed"); });
      expect(() => customReporting.removeFile("/written")).not.toThrow();
      expect(() => volume.metadata("/written")).toThrow();
    } finally {
      reporting.mockRestore();
    }
  });
});
