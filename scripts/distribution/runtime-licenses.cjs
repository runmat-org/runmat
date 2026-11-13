// Exact first-party license assets, separate from third-party publisher notices.
const { lstatSync, readFileSync, writeFileSync } = require("node:fs");
const { join, resolve } = require("node:path");

const LICENSE_FILE_NAMES = Object.freeze(["LICENSE.md", "FREE-USE.md"]);
const sourceRoot = resolve(__dirname, "../..");

function regularFile(filename) {
  const stat = lstatSync(filename);
  if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`Runtime license must be a regular file: ${filename}`);
}

function verifyRuntimeLicenses(destination) {
  const root = resolve(destination);
  const stat = lstatSync(root);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error(`Runtime license destination must be a real directory: ${root}`);
  for (const name of LICENSE_FILE_NAMES) {
    const source = join(sourceRoot, name), target = join(root, name);
    regularFile(source);
    regularFile(target);
    if (!readFileSync(source).equals(readFileSync(target))) throw new Error(`Runtime license bytes differ: ${target}`);
  }
}

function stageRuntimeLicenses(destination) {
  const root = resolve(destination);
  const stat = lstatSync(root);
  if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error(`Runtime license destination must be a real directory: ${root}`);
  for (const name of LICENSE_FILE_NAMES) {
    const source = join(sourceRoot, name), target = join(root, name);
    regularFile(source);
    try { writeFileSync(target, readFileSync(source), { flag: "wx" }); }
    catch (error) {
      if (error.code !== "EEXIST") throw error;
      regularFile(target);
      if (!readFileSync(source).equals(readFileSync(target))) throw new Error(`Runtime license bytes differ: ${target}`);
    }
  }
  verifyRuntimeLicenses(root);
}

module.exports = { LICENSE_FILE_NAMES, stageRuntimeLicenses, verifyRuntimeLicenses };

if (require.main === module) {
  const [action, destination, ...extra] = process.argv.slice(2);
  if (extra.length || !destination || !["stage", "verify"].includes(action)) throw new Error("Usage: runtime-licenses.cjs <stage|verify> <directory>");
  (action === "stage" ? stageRuntimeLicenses : verifyRuntimeLicenses)(destination);
}
