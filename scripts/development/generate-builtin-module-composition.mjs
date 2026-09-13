#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

import { renderModuleCompositionProduct } from "./builtin-migration/module-composition/generate.mjs";
import { parseCompositionProduct } from "./builtin-migration/module-composition/schema.mjs";

const output = outputArgument(process.argv.slice(2));
const product = parseCompositionProduct(JSON.parse(fs.readFileSync(0, "utf8")));
if (lstatOrNull(output) !== null) throw new Error("composition output already exists");
if (product.state === "absent") {
  // A successful absent product intentionally leaves the create-new output unmaterialized.
} else {
  const source = renderModuleCompositionProduct(product);
  const temporary = `${output}.runmat-${process.pid}.tmp`;
  try {
    fs.writeFileSync(temporary, source, { encoding: "utf8", flag: "wx" });
    fs.linkSync(temporary, output);
    fs.unlinkSync(temporary);
  } finally {
    fs.rmSync(temporary, { force: true });
  }
}

function outputArgument(argumentsList) {
  if (argumentsList.length !== 2 || argumentsList[0] !== "--output") {
    throw new Error("usage: generate-builtin-module-composition.mjs --output PATH");
  }
  const result = path.resolve(argumentsList[1]);
  if (result === path.parse(result).root) throw new Error("composition output must be a file path");
  return result;
}

function lstatOrNull(target) {
  try {
    return fs.lstatSync(target);
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}
