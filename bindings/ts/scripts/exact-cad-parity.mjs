import assert from "node:assert/strict";
import { createHash } from "node:crypto";

// Exact BREP bytes can differ in derived p-curve roundoff between kernels.
// Keep their content hashes exact. Compare source-fenced entity addresses and
// incidence instead of pretending different representations have the same hash.
const identity = value => JSON.stringify([value.kind, value.source_topology_id, value.assembly_path]);
function portableSnapshot(imported) {
  const aliases = new Map(), shapeNames = new Map(), evaluators = new Map();
  for (const [slot, id] of imported.kernel_entity_ids) {
    const alias = `kernel:${JSON.stringify(slot)}`;
    aliases.set(identity(id), alias);
    shapeNames.set(JSON.stringify([id.source_topology_id, id.assembly_path]), alias);
  }
  const digest = [...createHash("sha256").update(Buffer.from(imported.representation, "base64")).digest()];
  assert.deepEqual(imported.model.artifact.digest, digest, "exact artifact hash matches its own bytes");
  for (const [role, rows] of Object.entries(imported.evaluators)) {
    if (!Array.isArray(rows)) continue;
    for (const row of rows) {
      const reference = row.implementation.reference;
      if (reference) {
        assert.deepEqual(reference.representation_digest, digest, "evaluator binds its own exact artifact");
        evaluators.set(row.id, `${role}:${reference.entity_token}`);
      } else if (role === "mass_properties") {
        evaluators.set(row.id, `mass:${JSON.stringify(imported.kernel_body_shapes[row.id])}`);
      }
    }
  }
  const pcurves = new Map(imported.evaluators.pcurves.map(row => [row.id, row.implementation.reference.entity_token]));
  for (const coedge of imported.topology.coedges) aliases.set(identity(coedge.id), `coedge:${pcurves.get(coedge.pcurve_evaluator_id)}`);
  const coedges = new Map(imported.topology.coedges.map(row => [identity(row.id), row]));
  for (const wire of imported.topology.wires) {
    const coedge = coedges.get(identity(wire.coedge_ids[0]));
    const token = pcurves.get(coedge.pcurve_evaluator_id);
    const wireSlot = /:wire:(\d+):/.exec(token)?.[1];
    assert.ok(wireSlot, "wire has a concrete kernel address");
    aliases.set(identity(wire.id), `wire:${wireSlot}`);
  }
  const portableId = id => ({ ...id, source_topology_id: aliases.get(identity(id))
    ?? shapeNames.get(JSON.stringify([id.source_topology_id, id.assembly_path])) ?? id.source_topology_id });
  for (const shell of imported.topology.shells) aliases.set(identity(shell.id), `shell:${JSON.stringify(shell.face_uses.map(use => ({...use,entity_id:portableId(use.entity_id)})).sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b))))}`);
  for (const region of imported.topology.regions) aliases.set(identity(region.id), `region:${JSON.stringify(portableId(region.solid_id))}`);
  function normalize(value, key = "") {
    if (Array.isArray(value)) {
      const rows = value.map(item => normalize(item));
      if (key === "coedge_ids" && rows.length) {
        // A closed loop has no privileged start edge. Preserve its direction
        // and adjacency while choosing a start from portable kernel addresses.
        const first = rows.reduce((best,item,index)=>JSON.stringify(item)<JSON.stringify(rows[best]) ? index : best,0);
        return [...rows.slice(first),...rows.slice(0,first)];
      }
      if (rows.every(item => item && typeof item === "object" && "id" in item)) rows.sort((a,b)=>JSON.stringify(a.id).localeCompare(JSON.stringify(b.id)));
      else if (["face_uses","body_ids","child_instance_ids","lump_ids","solid_ids"].includes(key)) rows.sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b)));
      return rows;
    }
    if (value && typeof value === "object") {
      if ("source_topology_id" in value && "assembly_path" in value) return portableId(value);
      // These attest platform-local exact data; each closure was validated by
      // build_closure and representation references are checked above.
      return Object.fromEntries(Object.entries(value).filter(([key]) => !["definition_digest","representation_digest","validation_digest"].includes(key)).map(([key,item])=>[key,normalize(item,key)]));
    }
    return typeof value === "string" ? evaluators.get(value) ?? value : value;
  }
  return {sourceFaces:Object.fromEntries(Object.entries(imported.source_face_ids).map(([ordinal,id])=>[ordinal,portableId(id)])),topology:normalize(imported.topology),evaluators:normalize(imported.evaluators)};
}
function close(actual, expected, path = "", scale = 0) {
  if (typeof expected === "number") {
    assert.equal(typeof actual,"number",path);
    if (Number.isInteger(actual) && Number.isInteger(expected)) assert.equal(actual,expected,path);
    else {
      const coordinateFloor = /point_m|centroid_m/.test(path) ? 1e-12 : 1e-30;
      assert.ok(Math.abs(actual-expected) <= Math.max(coordinateFloor,Math.abs(expected)*1e-12,scale*1e-12),`${path}: ${actual} versus ${expected}`);
    }
  } else if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual),path);assert.equal(actual.length,expected.length,path);
    const localScale=expected.every(value=>typeof value==="number") ? Math.max(0,...expected.map(Math.abs)) : scale;
    expected.forEach((value,index)=>close(actual[index],value,`${path}[${index}]`,localScale));
  } else if (expected && typeof expected === "object") {
    assert.deepEqual(Object.keys(actual).sort(),Object.keys(expected).sort(),path);
    for (const key of Object.keys(expected)) close(actual[key],expected[key],`${path}.${key}`,scale);
  } else assert.equal(actual,expected,path);
}
export function assertCadImportParity(browser,native) {
  assert.deepEqual(browser.source_digest,native.source_digest,"same source revision");
  for (const field of ["source_format", "source_unit_evidence", "kernel_units"]) {
    assert.ok(Object.hasOwn(browser, field) && Object.hasOwn(native, field), `${field} must be present in both current contracts`);
    assert.deepEqual(browser[field], native[field], field);
  }
  assert.equal(browser.kernel_version,native.kernel_version);
  close(portableSnapshot(browser),portableSnapshot(native));
}
