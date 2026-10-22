import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { inputFile, safePath } from "./manifest.mjs";
import { readDocumentationExport } from "../development/builtin-migration/documentation-export.mjs";
export const readJson = (file) => JSON.parse(readFileSync(file, "utf8"));
export function copyInput(source, output, sourcePath, outputPath = sourcePath) {
    const target = join(output, outputPath);
    mkdirSync(dirname(target), { recursive: true });
    copyFileSync(inputFile(source, sourcePath), target);
}
export function collectDocuments(source, output, validate, builtinDocumentationPath, productionDocumentation) {
    const catalog = readJson(join(source, "docs/catalog.json"));
    validate("document", catalog);
    const ids = new Set();
    for (const document of catalog.documents) {
        if (ids.has(document.id))
            throw new Error(`Duplicate document: ${document.id}`);
        ids.add(document.id);
        copyInput(source, output, document.path, "documents/" + document.path);
    }
    writeFileSync(join(output, "documents.json"), JSON.stringify(catalog, null, 2) + "\n");
    const exported = productionDocumentation ?? readDocumentationExport({ repository: source, exportPath: builtinDocumentationPath });
    const names = new Set();
    const documents = exported.builtins.map(document => {
        const name = document.module_stem;
        if (typeof name !== "string" || name.includes("/") || name.includes("\\") || /[\x00-\x1f]/u.test(name))
            throw new Error("Invalid exported builtin document filename");
        safePath(name);
        if (names.has(name.toLowerCase())) throw new Error(`Duplicate builtin document filename: ${name}`);
        names.add(name.toLowerCase());
        return { name, document };
    }).sort((left, right) => left.name < right.name ? -1 : left.name > right.name ? 1 : 0);
    mkdirSync(join(output, "builtins/reference"), { recursive: true });
    for (const { name, document } of documents) {
        writeFileSync(join(output, "builtins/reference", name + ".json"), JSON.stringify(document, null, 2) + "\n");
    }
}
