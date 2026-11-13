#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

import {
  canonicalEvidencePath, publishEvidenceBytes,
} from "./builtin-migration/atomic-evidence-publication.mjs";
import {
  renderModuleCompositionCompanion, renderModuleCompositionProduct,
} from "./builtin-migration/module-composition/generate.mjs";
import { parseCompositionProduct } from "./builtin-migration/module-composition/schema.mjs";

const { output, section } = outputArguments(process.argv.slice(2));
const product = parseCompositionProduct(JSON.parse(fs.readFileSync(0, "utf8")));
if (lstatOrNull(output) !== null) throw new Error("composition output already exists");
if (product.state === "absent") {
  // A successful absent product intentionally leaves the create-new output unmaterialized.
} else {
  const source = section === null ? renderModuleCompositionProduct(product)
    : renderModuleCompositionCompanion(product, section);
  publishEvidenceBytes(output, source);
}

function outputArguments(argumentsList) {
  if (![2, 4].includes(argumentsList.length) || argumentsList[0] !== "--output"
    || (argumentsList.length === 4 && argumentsList[2] !== "--section")) {
    throw new Error("usage: generate-builtin-module-composition.mjs --output PATH [--section reexports|declarations]");
  }
  const result = path.resolve(argumentsList[1]);
  if (result === path.parse(result).root) throw new Error("composition output must be a file path");
  const section = argumentsList.length === 4 ? argumentsList[3] : null;
  if (section !== null && !["reexports", "declarations"].includes(section)) {
    throw new Error("composition companion section is invalid");
  }
  return { output: canonicalEvidencePath(result), section };
}

function lstatOrNull(target) {
  try {
    return fs.lstatSync(target);
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}
