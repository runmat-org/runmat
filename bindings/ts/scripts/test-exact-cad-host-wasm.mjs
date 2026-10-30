import assert from "node:assert/strict";
import "fake-indexeddb/auto";
import { readFile, writeFile } from "node:fs/promises";
import createOcct from "../dist/occt/runmat_geometry_wasm_host.js";
import { initRunMat, createInMemoryFsProvider, createIndexedDbPackageCache, createBrowserProjectSession } from "../dist/index.js";
import { assertCadImportParity } from "./exact-cad-parity.mjs";

// Exercise the compiled C++/Rust adapter and then real browser-runtime checking.
// JSON payloads stay strings across JS so u64 identities cannot be rounded.
const occt = await createOcct({ wasmBinary: await readFile(new URL("../dist/occt/runmat_geometry_wasm_host.wasm", import.meta.url)) });
assert.ok(occt.HEAPU8 instanceof Uint8Array, "CAD module exports its growable heap view");
const invoke = occt.cwrap("runmat_occt_exact_import", "number", ["number"]);
const release = occt.cwrap("runmat_occt_exact_free", null, ["number"]);
function request(mode, path, bytes, options) {
  const packet = new TextEncoder().encode(JSON.stringify({ mode, path, bytes_base64: Buffer.from(bytes).toString("base64"), options }));
  const input = occt._malloc(packet.length + 1);
  assert.ok(input);
  let output = 0;
  try {
    occt.HEAPU8.set(packet, input); occt.HEAPU8[input + packet.length] = 0;
    output = invoke(input); assert.ok(output);
    const result = JSON.parse(occt.UTF8ToString(output));
    if (result.error) throw Object.assign(new Error(result.error.message), { occtImportError: result.error.cause });
    assert.equal(typeof result.imported, "string");
    return result.imported;
  } finally { if (output) release(output); occt._free(input); }
}
globalThis.__runmatOcctImportExactCad = (path, bytes, options) => request("exact", path, bytes, options);
globalThis.__runmatOcctImportGeometry = (path, bytes, options) => request("preview", path, bytes, options);
const filesystem = createInMemoryFsProvider();
const encoder = new TextEncoder();
await filesystem.writeFile("/runmat.toml", encoder.encode('[package]\nname = "exact-cad-browser-qualification"\n[sources]\nroots = ["."]\n'));
const wasmModule = await readFile(new URL("../dist/pkg-web/runmat_wasm_web_bg.wasm", import.meta.url));
const session = await initRunMat({ enableGpu: false, enableJit: false, telemetryConsent: false, fsProvider: filesystem, wasmModule });
const cache = await createIndexedDbPackageCache({ dbName: `exact-cad-${crypto.randomUUID()}` });
const project = await createBrowserProjectSession({ session, filesystem, packageCache: cache.provider, gitGateway: { baseUrl: "http://127.0.0.1:9" } });
try {
  await project.resolve({ manifestPath: "/runmat.toml", lockPath: null, options: {
    target: "wasm32-unknown-unknown", default_server_origin: "http://127.0.0.1:9", default_registry_index: "http://127.0.0.1:9",
    groups: ["runtime"], root_features: [], host_capabilities: ["browser-filesystem", "worker"], source_intent: "execute", source_policy: { offline: true },
  } });
  for (const file of ["box.brep", "meshing_sphere.brep", "meshing_torus.brep", "two_box_assembly.step", ...(process.argv[2] ? [process.argv[2]] : [])]) {
    const bytes = await readFile(file.startsWith("/") ? file : new URL(`../../../crates/runmat-geometry/io/tests/fixtures/${file}`, import.meta.url));
    const name = file.split("/").at(-1);
    if (file === process.argv[2]) {
      const native = process.env.RUNMAT_NATIVE_EXACT_RESPONSE
        ? JSON.parse(JSON.parse(await readFile(process.env.RUNMAT_NATIVE_EXACT_RESPONSE, "utf8")).imported) : null;
      if (native) {
        const wire = JSON.parse(await readFile(process.env.RUNMAT_EXACT_REQUEST, "utf8"));
        const browser = JSON.parse(request("exact", wire.path, bytes, wire.options));
        if (process.env.RUNMAT_BROWSER_EXACT_RESPONSE) await writeFile(process.env.RUNMAT_BROWSER_EXACT_RESPONSE, JSON.stringify(browser));
        assertCadImportParity(browser,native);
        const wrongFace = structuredClone(browser);
        wrongFace.source_face_ids["0"] = wrongFace.source_face_ids["1"];
        assert.throws(() => assertCadImportParity(wrongFace,native), "a different face must never match");
        const wrongGeometry = structuredClone(browser);
        wrongGeometry.topology.vertices[0].point_m[0] += 0.001;
        assert.throws(() => assertCadImportParity(wrongGeometry,native), "physical changes must fail parity");
        const corruptArtifact = structuredClone(browser);
        corruptArtifact.representation = Buffer.from("corrupt").toString("base64");
        assert.throws(() => assertCadImportParity(corruptArtifact,native), "exact hashes cannot be normalized away");
        console.log(`Browser/desktop source-face, topology, and evaluator parity: ${name}`);
      }
      for (const [scale, expected] of [[1, [0.04, 0.05, 0.035]], [0.001, [40, 50, 35]]]) {
        // The canonical LengthUnit wire is a dimensioned SI conversion, not
        // an enum name. Keep the same meter/millimeter geometry assertions.
        const units = { dimension: { length: 1, mass: 0, time: 0, current: 0,
          temperature: 0, amount: 0, luminous_intensity: 0, information: 0 }, scale, offset: 0 };
        const preview = JSON.parse(request("preview", name, bytes, JSON.stringify({
          max_triangles: 16000000, budget_policy: "Strict", units, relative_deflection: false,
          tessellation_profile: { profile_id: "default-v1", chord_tolerance: null, angle_tolerance_deg: null, healing_mode: "safe" },
        })));
        const vertices = preview.asset.surface_meshes.flatMap(mesh => mesh.vertices);
        for (let axis = 0; axis < 3; axis++) {
          const values = vertices.map(vertex => vertex[axis]);
          const extent = Math.max(...values) - Math.min(...values);
          assert.ok(Math.abs(extent - expected[axis]) < expected[axis] * 1e-6, `${units} axis ${axis}: ${extent}`);
        }
        console.log(`Bracket preview scale verified at ${scale} meters per unit`);
      }
    }
    await filesystem.writeFile(`/${name}`, bytes);
    await filesystem.writeFile("/study.fea", encoder.encode(`geometry part {\n file = "./${name}"\n units = mm\n}\n`));
    const checked = await session.checkFeaWorkingSource({ root: "study.fea", overlays: [] });
    assert.equal(checked.status, "valid", JSON.stringify(checked.diagnostics));
    assert.equal(checked.program.geometries.length, 1);
    assert.equal(checked.program.geometries[0].source.capabilities.topology, true);
    assert.equal(checked.program.geometries[0].source.capabilities.persistent_ids, true);
    const definition = await session.browseFeaSource({ revision: checked.revision,
      module_path: null, filter: { type: "declaration", kind: "geometry" },
      after: null, limit: 1, detail: "definition" });
    assert.equal(definition.type, "page", JSON.stringify(definition));
    assert.equal(definition.page.details.length, 1);
    console.log(`Actual WASM check: ${name} has validated exact topology and persistent IDs`);
  }
  await filesystem.writeFile("/broken.step", encoder.encode("not a STEP model"));
  await filesystem.writeFile("/study.fea", encoder.encode('geometry part {\n file = "./broken.step"\n units = mm\n}\n'));
  const invalid = await session.checkFeaWorkingSource({ root: "study.fea", overlays: [] });
  assert.equal(invalid.status, "invalid");
  assert.ok(!JSON.stringify(invalid.diagnostics).includes("native OCCT host"));
  // A real IGES wireframe is readable CAD, but is correctly rejected as a solid
  // simulation input by the same typed admission rule used on desktop.
  await filesystem.writeFile("/wireframe.iges", await readFile(new URL("../../../crates/runmat-geometry/io/tests/fixtures/ansys_sample.iges", import.meta.url)));
  await filesystem.writeFile("/study.fea", encoder.encode('geometry part {\n file = "./wireframe.iges"\n units = mm\n}\n'));
  const wireframe = await session.checkFeaWorkingSource({ root: "study.fea", overlays: [] });
  assert.equal(wireframe.status, "invalid");
  assert.ok(!JSON.stringify(wireframe.diagnostics).includes("native OCCT host"));
  // Recover in the same worker after the kernel rejects invalid input.
  await filesystem.writeFile("/study.fea", encoder.encode('geometry part {\n file = "./box.brep"\n units = mm\n}\n'));
  assert.equal((await session.checkFeaWorkingSource({ root: "study.fea", overlays: [] })).status, "valid");
  console.log("Invalid CAD diagnostics and recovery in the same browser host passed");
} finally { await project.dispose(); await session.dispose(); cache.close(); }
