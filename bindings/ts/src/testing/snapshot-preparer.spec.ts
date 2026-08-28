import { describe, expect, it, vi } from "vitest";

import { createInMemoryFsProvider } from "../fs/index.js";
import { BrowserTestSnapshotPreparer } from "./snapshot-preparer.js";

describe("BrowserTestSnapshotPreparer", () => {
  it("captures deterministic project sources and delegates semantics to Rust", async () => {
    const filesystem = createInMemoryFsProvider();
    await filesystem.createDirAll?.("/workspace/tests");
    await filesystem.createDirAll?.("/workspace/src");
    await filesystem.writeFile(
      "/workspace/runmat.toml",
      new TextEncoder().encode("[package]\nname = \"sample\"\n")
    );
    await filesystem.writeFile(
      "/workspace/tests/b_test.m",
      new TextEncoder().encode("%% b\n")
    );
    await filesystem.writeFile(
      "/workspace/tests/a_test.m",
      new TextEncoder().encode("%% a\n")
    );
    await filesystem.writeFile(
      "/workspace/tests/fixtures.csv",
      new TextEncoder().encode("x,y\n1,2\n")
    );
    await filesystem.writeFile(
      "/workspace/src/helper.m",
      new TextEncoder().encode("function helper\nend\n")
    );
    const freezeTestSnapshot = vi.fn(async (input) => ({
      program_revision: { graph_digest: input.graphDigest },
      sources: input.savedSources
    }));
    const prepareTests = vi.fn(async (snapshot, selector) => ({
      snapshot,
      discovery: { suites: [] },
      plan: { run_id: "run-1", selector }
    }));
    const preparer = new BrowserTestSnapshotPreparer(
      {
        projectTestLayout: vi.fn(async () => ({
          sourceRoots: ["src"],
          testRoots: ["tests"],
          testPaths: [],
          testConfigDigest: "sha256:config"
        })),
        freezeTestSnapshot
      },
      { prepareTests },
      filesystem
    );

    const prepared = await preparer.prepare({
      manifestPath: "/workspace/runmat.toml",
      projectRevision: {
        graph_digest: "sha256:graph",
        source_revision: "sha256:sources"
      }
    });

    expect(freezeTestSnapshot).toHaveBeenCalledWith(
      expect.objectContaining({
        graphDigest: "sha256:graph",
        baseSourceDigest: "sha256:sources",
        savedSources: [
          expect.objectContaining({ relative_path: "src/helper.m" }),
          expect.objectContaining({ relative_path: "tests/a_test.m" }),
          expect.objectContaining({ relative_path: "tests/b_test.m" })
        ]
      })
    );
    expect(prepareTests).toHaveBeenCalledWith(
      expect.any(Object),
      expect.objectContaining({ source_prefixes: ["tests/"] })
    );
    expect(prepared.plan).toMatchObject({ run_id: "run-1" });
    expect(prepared.filesystemSnapshot?.map(({ path }) => path)).toEqual([
      "/workspace/runmat.toml",
      "/workspace/src/helper.m",
      "/workspace/tests/a_test.m",
      "/workspace/tests/b_test.m",
      "/workspace/tests/fixtures.csv"
    ]);
  });
});
