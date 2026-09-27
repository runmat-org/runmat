import { describe, expect, it } from "vitest";
import { createInMemoryFsProvider } from "./memory.js";
import { BrowserPackageMountFilesystem } from "../package-cache/mount.js";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

describe("provider-owned rename replacement", () => {
  it("delegates replacement through mounts without falling back to rename", async () => {
    const workspace = createInMemoryFsProvider({ initialFiles: { "/source": "new", "/target": "old" } });
    const mount = new BrowserPackageMountFilesystem(workspace);
    await mount.renameReplace("/source", "/target");
    expect(decoder.decode(await mount.readFile("/target"))).toBe("new");
    await workspace.writeFile("/source", encoder.encode("next"));
    workspace.renameReplace = undefined;
    workspace.rename = () => { throw new Error("ordinary rename must not run"); };
    expect(() => mount.renameReplace("/source", "/target")).toThrow("renameReplace");
    expect(decoder.decode(await workspace.readFile("/source"))).toBe("next");
    expect(decoder.decode(await workspace.readFile("/target"))).toBe("new");
  });

  it("rejects either immutable mount endpoint before touching the workspace", async () => {
    const workspace = createInMemoryFsProvider({ initialFiles: { "/source": "keep" } });
    const mount = new BrowserPackageMountFilesystem(workspace);
    const digest = `sha256:${"a".repeat(64)}`;
    const root = mount.registerVolatile({ source: { tree_digest: digest }, tree: { digest, entries: [] } });
    expect(() => mount.renameReplace("/source", root)).toThrow("read-only");
    expect(() => mount.renameReplace(root, "/source")).toThrow("read-only");
    expect(decoder.decode(await workspace.readFile("/source"))).toBe("keep");
  });
  it("replaces files, preserves self, and leaves failed replacements unchanged", async () => {
    const fs = createInMemoryFsProvider({
      initialFiles: { "/source": "new", "/target": "old" },
    });
    await fs.renameReplace!("/source", "/target");
    expect(decoder.decode(await fs.readFile("/target"))).toBe("new");
    expect(() => fs.metadata("/source")).toThrow();
    await fs.renameReplace!("/target", "/target");
    expect(decoder.decode(await fs.readFile("/target"))).toBe("new");
  });

  it("preflights incompatible, nonempty, readonly, missing-parent and descendant targets", async () => {
    const fs = createInMemoryFsProvider({ initialFiles: { "/file": "original", "/occupied/keep": "keep" } });
    await fs.createDir!("/directory");
    await fs.createDir!("/directory/child");
    const failures: Array<[string, string]> = [
      ["/file", "/occupied"],
      ["/directory", "/file"],
      ["/directory", "/occupied"],
      ["/directory", "/directory/child/target"],
      ["/file", "/missing/target"],
      ["/missing", "/file"],
    ];
    for (const [from, to] of failures) {
      expect(() => fs.renameReplace!(from, to)).toThrow();
      expect(decoder.decode(await fs.readFile("/file"))).toBe("original");
      expect(decoder.decode(await fs.readFile("/occupied/keep"))).toBe("keep");
      expect((await fs.metadata("/directory/child")).fileType).toBe("dir");
    }
    expect(() => fs.metadata("/missing")).toThrow();
    await fs.writeFile("/readonly", encoder.encode("readonly"));
    await fs.setReadonly!("/readonly", true);
    expect(() => fs.renameReplace!("/file", "/readonly")).toThrow();
    expect(decoder.decode(await fs.readFile("/readonly"))).toBe("readonly");
    expect(decoder.decode(await fs.readFile("/file"))).toBe("original");
  });

  it("moves a directory onto an empty compatible directory without merging", async () => {
    const fs = createInMemoryFsProvider({ initialFiles: { "/source/child": "value" } });
    await fs.createDir!("/empty");
    await fs.renameReplace!("/source", "/empty");
    expect(decoder.decode(await fs.readFile("/empty/child"))).toBe("value");
    expect(() => fs.metadata("/source")).toThrow();
    expect((await fs.readDir("/")).map(entry => entry.fileName)).toEqual(["empty"]);
    expect(() => fs.renameReplace!("/", "/empty")).toThrow();
  });
});
