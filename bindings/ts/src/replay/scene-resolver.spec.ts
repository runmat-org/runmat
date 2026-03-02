import { describe, expect, it } from "vitest";

import { hydrateFigureSceneDataRefs } from "./scene-resolver.js";

describe("hydrateFigureSceneDataRefs", () => {
  it("hydrates runmat-data-array refs by artifact id", async () => {
    const scene = {
      schemaVersion: 1,
      kind: "figure-scene",
      figure: {
        schemaVersion: 1,
        layout: { axesRows: 1, axesCols: 1, axesIndices: [0] },
        metadata: {
          gridEnabled: true,
          legendEnabled: true,
          colorbarEnabled: false,
          axisEqual: false,
          backgroundRgba: [1, 1, 1, 1],
          legendEntries: [],
        },
        plots: [
          {
            kind: "surface",
            x: [0, 1],
            y: [0, 1],
            z: {
              refKind: "runmat-data-array-v1",
              dtype: "f64",
              shape: [2, 2],
              chunks: [
                {
                  artifactId:
                    "sha256:7af8faff9e5fe6ba87fec8e4ce6d79dca7f29bbee9f9809a36119346b411ee36",
                },
              ],
            },
            colormap: "Parula",
            shadingMode: "Smooth",
            wireframe: false,
            alpha: 1,
            flattenZ: false,
            colorLimits: null,
            axesIndex: 0,
            label: "surface",
            visible: true,
          },
        ],
      },
    };
    const bytes = new TextEncoder().encode(JSON.stringify(scene));

    const hydrated = await hydrateFigureSceneDataRefs(bytes, async (path) => {
      if (path.endsWith(
        "7af8faff9e5fe6ba87fec8e4ce6d79dca7f29bbee9f9809a36119346b411ee36.f64.chunk.json"
      )) {
        return new TextEncoder().encode(
          JSON.stringify({ dtype: "f64", shape: [4], values: [0, 1, 1, 2] })
        );
      }
      throw new Error(`not found: ${path}`);
    });

    const parsed = JSON.parse(new TextDecoder().decode(hydrated)) as {
      figure?: { plots?: Array<{ z?: unknown }> };
    };
    expect(parsed.figure?.plots?.[0]?.z).toEqual([
      [0, 1],
      [1, 2],
    ]);
  });
});
