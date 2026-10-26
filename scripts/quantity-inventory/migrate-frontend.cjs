#!/usr/bin/env node
// Exact TypeScript object-literal edits; strings containing source code and
// labels outside physical unit fields are deliberately untouched.
const fs=require('node:fs'),path=require('node:path');
const ts=require('../../bindings/ts/node_modules/typescript');
const unitNames=new Set(['meter','millimeter','inch','centimeter','micrometer','kilometer','foot']);
function files(root,out=[]){for(const e of fs.readdirSync(root,{withFileTypes:true})){if(e.isSymbolicLink())continue;const p=path.join(root,e.name);if(e.isDirectory()){if(!['node_modules','target','.git','dist'].includes(e.name))files(p,out);}else if(/\.tsx?$/.test(e.name))out.push(p);}return out;}
let count=0,total=0;
for(const file of process.argv.slice(2).flatMap(root=>files(root))){
 const source=fs.readFileSync(file,'utf8'),ast=ts.createSourceFile(file,source,ts.ScriptTarget.Latest,true,file.endsWith('.tsx')?ts.ScriptKind.TSX:ts.ScriptKind.TS);
 const edits=[],needed=new Set();
 const add=(start,end,text)=>edits.push({start,end,text});
 const propName=node=>node.name&&(ts.isIdentifier(node.name)||ts.isStringLiteral(node.name))?node.name.text:null;
 const convert=value=>{if(ts.isStringLiteral(value)&&unitNames.has(value.text)){needed.add('physicalLengthUnit');return `physicalLengthUnit(${JSON.stringify(value.text)})`;}return value.getText(ast);};
 const walk=node=>{
  if(ts.isObjectLiteralExpression(node)){
   const props=node.properties.filter(ts.isPropertyAssignment);
   const kind=props.find(p=>propName(p)==='kind');
   const imported=kind&&ts.isStringLiteral(kind.initializer)&&kind.initializer.text==='imported'&&props.some(p=>propName(p)==='logical_path');
   if(imported){
    const units=props.find(p=>propName(p)==='units');
    const fallback=props.find(p=>propName(p)==='unit_fallback');
    const unit=units||fallback;
    if(units){add(units.name.getStart(ast),units.name.getEnd(),'unit_fallback');const value=convert(units.initializer);if(value!==units.initializer.getText(ast))add(units.initializer.getStart(ast),units.initializer.getEnd(),value);}
    if(!unit)add(node.getStart(ast)+1,node.getStart(ast)+1,'unit_fallback: null,');
    if(!props.some(p=>propName(p)==='unit_evidence')){
     const expr=unit?convert(unit.initializer):'null';
     const evidence=expr==='null'?'{status:"missing",reason:"fixture has no source-unit evidence"}':`explicitSourceLengthUnit(${expr}, "explicit fixture geometry units")`;
     if(expr!=='null')needed.add('explicitSourceLengthUnit');
     add(node.getStart(ast)+1,node.getStart(ast)+1,`unit_evidence: ${evidence},`);
    }
   }
   for(const p of props){if(imported&&propName(p)==='units')continue;if(['units','source_units','resolved_units','geometryUnits'].includes(propName(p))){const value=convert(p.initializer);if(value!==p.initializer.getText(ast))add(p.initializer.getStart(ast),p.initializer.getEnd(),value);}}
  }
  ts.forEachChild(node,walk);
 };
 walk(ast);if(!edits.length)continue;
 edits.sort((a,b)=>b.start-a.start||b.end-a.end);
 let text=source,upper=source.length;
 for(const e of edits){if(e.end>upper)throw Error(`overlapping frontend edits ${file}:${e.start}`);text=text.slice(0,e.start)+e.text+text.slice(e.end);upper=e.start;}
 const existing=source.match(/import\s*\{([^}]+)\}\s*from\s*["']runmat\/fea-contracts["']/g)||[];
 const missing=[...needed].filter(name=>!existing.some(line=>new RegExp(`\\b${name}\\b`).test(line)));
 if(missing.length)text=`import { ${missing.join(', ')} } from "runmat/fea-contracts";\n`+text;
 fs.writeFileSync(file,text);count++;total+=edits.length;
}
console.log(`Migrated ${total} physical object-literal fields across ${count} frontend files`);
