import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const skillsDir = path.join(root, '.agents', 'skills');
const out = path.join(root, 'bootstrap', 'SKILLS-MANIFEST.json');
const source = {
  'scientific-critical-thinking':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/scientific-critical-thinking','MIT'],
  'experimental-design':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/experimental-design','MIT'],
  'exploratory-data-analysis':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/exploratory-data-analysis','MIT'],
  'hypothesis-generation':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/hypothesis-generation','MIT'],
  'literature-review':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/literature-review','MIT'],
  'networkx':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/networkx','MIT'],
  'peer-review':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/peer-review','MIT'],
  'pennylane':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/pennylane','MIT'],
  'qiskit':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/qiskit','MIT'],
  'scientific-writing':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/scientific-writing','MIT'],
  'scikit-learn':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/scikit-learn','MIT'],
  'statistical-analysis':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/statistical-analysis','MIT'],
  'statsmodels':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/statsmodels','MIT'],
  'sympy':['K-Dense-AI/scientific-agent-skills','5f2bc0341354fd5590aef76a533ada1e1e5b4e62','skills/sympy','MIT'],
  'scientific-problem-selection':['anthropics/knowledge-work-plugins','c75fda00ff0e41c8ee331ae312c8a07f1967653b','bio-research/skills/scientific-problem-selection','Apache-2.0'],
  'architecture-patterns':['wshobson/agents','0874a27b17a22e2b7c2e2bcd725768e0803cad06','plugins/backend-development/skills/architecture-patterns','MIT'],
  'code-review-excellence':['wshobson/agents','0874a27b17a22e2b7c2e2bcd725768e0803cad06','plugins/developer-essentials/skills/code-review-excellence','MIT'],
  'debugging-strategies':['wshobson/agents','0874a27b17a22e2b7c2e2bcd725768e0803cad06','plugins/developer-essentials/skills/debugging-strategies','MIT'],
  'architecture-decision-records':['wshobson/agents','0874a27b17a22e2b7c2e2bcd725768e0803cad06','plugins/documentation-generation/skills/architecture-decision-records','MIT'],
  'python-design-patterns':['wshobson/agents','0874a27b17a22e2b7c2e2bcd725768e0803cad06','plugins/python-development/skills/python-design-patterns','MIT'],
  'python-packaging':['wshobson/agents','0874a27b17a22e2b7c2e2bcd725768e0803cad06','plugins/python-development/skills/python-packaging','MIT'],
  'python-testing-patterns':['wshobson/agents','0874a27b17a22e2b7c2e2bcd725768e0803cad06','plugins/python-development/skills/python-testing-patterns','MIT'],
  'python-type-safety':['wshobson/agents','0874a27b17a22e2b7c2e2bcd725768e0803cad06','plugins/python-development/skills/python-type-safety','MIT'],
  'stride-analysis-patterns':['wshobson/agents','0874a27b17a22e2b7c2e2bcd725768e0803cad06','plugins/security-scanning/skills/stride-analysis-patterns','MIT'],
  'no-ai-slop':['petergyang/no-ai-slop','000650b156983f5159695b441477f4e63b25dc85','skills/no-ai-slop','MIT'],
};
const expected = new Set(Object.keys(source));
const actual = new Set(fs.readdirSync(skillsDir,{withFileTypes:true}).filter(d=>d.isDirectory()).map(d=>d.name));
const errors=[];
if(expected.size!==25) errors.push(`source table should contain exactly25, got ${expected.size}`);
for(const n of actual) if(!expected.has(n)) errors.push(`unexpected skill directory: ${n}`);
for(const n of expected) if(!actual.has(n)) errors.push(`missing skill directory: ${n}`);
const allFiles = (dir) => fs.readdirSync(dir,{withFileTypes:true}).flatMap(d=>d.isDirectory()?allFiles(path.join(dir,d.name)):[path.join(dir,d.name)]);
const stagingPath = (name, repo, upstream) => {
  if(repo==='K-Dense-AI/scientific-agent-skills') return path.join(root,'bootstrap','skill-sources','selected-kdense','skills',name);
  if(repo==='wshobson/agents') return path.join(root,'bootstrap','skill-sources','selected-wshobson',upstream);
  if(repo==='petergyang/no-ai-slop') return path.join(root,'bootstrap','skill-sources','selected-no-ai-slop','skills',name);
  if(repo==='anthropics/knowledge-work-plugins') return path.join(root,'_archive','pre-debugleaf-20261028-220317','preparation','temporary-skills',name);
  throw new Error(`unknown source repository for ${name}`);
};
const entries=[];
const pathRe=/(?:\.\.?\/)?(?:references|scripts|assets)\/[A-Za-z0-9_.${}/-]+\.[A-Za-z0-9]+/g;
const linkRe=/\[[^\]]*\]\(([^)]+)\)/g;
const suspicious=/(\bsubprocess\b|\bos\.system\b|\beval\s*\(|\bexec\s*\(|requests\.(?:get|post|put|delete)|urllib\.request|httpx\.(?:get|post)|socket\.)/g;
for(const name of [...expected].sort()) {
  const dir=path.join(skillsDir,name), skill=path.join(dir,'SKILL.md');
  if(!fs.existsSync(skill)){errors.push(`${name}: missing SKILL.md`);continue;}
  const content=fs.readFileSync(skill,'utf8');
  const front=content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  const nameMatch=front?.[1].match(/^name:\s*["']?([^\s"']+)/m);
  const descMatch=front?.[1].match(/^description:\s*(.+)$/m);
  if(!front) errors.push(`${name}: invalid/missing YAML frontmatter`);
  if(nameMatch?.[1]!==name) errors.push(`${name}: frontmatter name missing or mismatched`);
  if(!descMatch?.[1]?.trim()) errors.push(`${name}: frontmatter description missing`);
  const files=[], missingRefs=new Set(), signals=[], referenceFindings=[];
  for(const f of allFiles(dir).sort()){
    const rel=path.relative(dir,f).split(path.sep).join('/'), buf=fs.readFileSync(f);
    files.push({path:rel,bytes:buf.length,sha256:crypto.createHash('sha256').update(buf).digest('hex')});
    if(/\.(md|txt)$/i.test(f)){
      const md=buf.toString('utf8');
      const refs=new Set(md.match(pathRe)||[]);
      for(const m of md.matchAll(linkRe)){
        const target=m[1].split('#')[0];
        if(!/^(?:https?:|mailto:)/i.test(target) && (target.includes('/') || /\.(md|py|json|csv|ya?ml)$/i.test(target))) refs.add(target);
      }
      for(let ref of refs){
        ref=ref.replace(/[.>,;:'"`]+$/g,'').split('#')[0];
        if(!ref || /[{}$]/.test(ref)) continue;
        const candidates=[path.resolve(path.dirname(f),ref),path.resolve(dir,ref)];
        const inSkill=candidates.filter(p=>(p.startsWith(dir+path.sep)||p===dir));
        if(inSkill.length && !inSkill.some(p=>fs.existsSync(p))) missingRefs.add(ref);
      }
    }
    if(/\.(py|sh|ps1|js|mjs)$/i.test(f)){
      const hits=[...new Set(buf.toString('utf8').match(suspicious)||[])];
      if(hits.length) signals.push({path:rel,signals:hits});
    }
  }
  if(name==='architecture-decision-records') {
    const examples=[...missingRefs].filter(x=>/^000[123]-|^0020-/.test(x));
    if(examples.length) referenceFindings.push({classification:'illustrative-only',paths:examples,detail:'Sample ADR tree/index names shown in documentation; the skill creates these in the target repository and does not require bundled example files.'});
    for(const x of examples) missingRefs.delete(x);
  }
  if(name==='scientific-critical-thinking') {
    const schematics=[...missingRefs].filter(x=>x==='scripts/generate_schematic.py');
    if(schematics.length) referenceFindings.push({classification:'optional-cross-skill-unavailable',paths:['skills/scientific-schematics/scripts/generate_schematic.py'],detail:'Optional separate scientific-schematics skill; not in the 25-skill set. Do not invoke. Figure generation requires explicit authorization and outbound API credentials.'});
    for(const x of schematics) missingRefs.delete(x);
  }
  if(name==='hypothesis-generation'||name==='peer-review') {
    const removed=[...missingRefs].filter(x=>x==='scripts/generate_schematic.py'||x==='scripts/generate_schematic_ai.py');
    if(removed.length) referenceFindings.push({classification:'documented-deleted-optional-scripts',paths:removed,detail:'Security-validation reference documents these schematic scripts as intentionally removed; no skill instruction requires them.'});
    for(const x of removed) missingRefs.delete(x);
  }
  if(missingRefs.size) errors.push(`${name}: missing refs ${[...missingRefs].sort().join(', ')}`);
  const [repo,commit,upstream,license]=source[name];
  const sourceDir=stagingPath(name,repo,upstream);
  const sourceFiles=fs.existsSync(sourceDir)?allFiles(sourceDir):[];
  const sourceMap=new Map(sourceFiles.filter(f=>!/^license\.(md|txt)$/i.test(path.basename(f))&&path.basename(f).toLowerCase()!=='license').map(f=>[path.relative(sourceDir,f).split(path.sep).join('/'),crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex')]));
  const installedMap=new Map(files.filter(f=>!/^license\.(md|txt)$/i.test(f.path)&&f.path.toLowerCase()!=='license').map(f=>[f.path,f.sha256]));
  const missingFromInstall=[...sourceMap.keys()].filter(f=>!installedMap.has(f)).sort();
  const extraInInstall=[...installedMap.keys()].filter(f=>!sourceMap.has(f)).sort();
  const hashMismatches=[...sourceMap.keys()].filter(f=>installedMap.has(f)&&sourceMap.get(f)!==installedMap.get(f)).sort();
  const licenseSource=repo==='K-Dense-AI/scientific-agent-skills'?path.join(root,'bootstrap','skill-sources','selected-kdense','LICENSE.md'):repo==='wshobson/agents'?path.join(root,'bootstrap','skill-sources','selected-wshobson','LICENSE'):repo==='petergyang/no-ai-slop'?path.join(root,'bootstrap','skill-sources','selected-no-ai-slop','LICENSE'):path.join(sourceDir,'LICENSE.txt');
  const installedLicense=files.find(f=>/^license(\.(md|txt))?$/i.test(f.path));
  const licenseMatched=Boolean(installedLicense&&fs.existsSync(licenseSource)&&installedLicense.sha256===crypto.createHash('sha256').update(fs.readFileSync(licenseSource)).digest('hex'));
  if(!sourceMap.size) errors.push(`${name}: pinned staged source unavailable for comparison`);
  if(missingFromInstall.length||extraInInstall.length||hashMismatches.length) errors.push(`${name}: source copy differs; missing=${missingFromInstall.join(',')}; extra=${extraInInstall.join(',')}; hash=${hashMismatches.join(',')}`);
  if(!licenseMatched) errors.push(`${name}: installed license missing or differs from pinned source license`);
  let reviewFindings=['Static files and references checked; bundled scripts were not executed and runtime behavior is not verified.'];
  if(name==='experimental-design') reviewFindings.push('Bundled randomization/DOE helpers operate on local inputs and seeded designs; listed package-install commands are instructions, not executed setup.');
  if(name==='exploratory-data-analysis') reviewFindings.push('Bundled analyzers read caller-supplied local files and may write requested reports; constrain input paths and use an explicit output directory.');
  if(name==='hypothesis-generation'||name==='peer-review') reviewFindings.push('Bundled validation/scaffold tools are documented as local deterministic CLIs; source validation notes explicitly remove network, credential, subprocess, and dynamic-code behavior.');
  if(name==='scientific-writing'||name==='statistical-analysis') reviewFindings.push('Bundled helpers are local report/assumption checks; outputs support review and do not establish scientific truth or replace judgment.');
  if(name==='literature-review') reviewFindings.push('Optional citation lookup makes outbound DOI/database requests. Optional schematic scripts call Pandoc/LaTeX subprocesses and can read OPENROUTER_API_KEY from the environment or .env before sending prompts to OpenRouter; do not invoke without explicit authorization and a secret-handling plan.');
  if(name==='qiskit') reviewFindings.push('Runtime inspection may use IBM Quantum saved credentials for read-only account/backend inspection; no job submission was found by independent static review.');
  if(name==='pennylane') reviewFindings.push('No bundled scripts; local simulator guidance is separate from optional provider integrations, which require explicit account/network setup.');
  if(name==='scikit-learn') reviewFindings.push('One demo writes two fixed PNG names in the current working directory and may overwrite existing files; use a scratch directory. Split strategy must match grouped/temporal data.');
  if(name==='statsmodels') reviewFindings.push('Documentation/examples only; grouped or chronological validation may be required by the route/time sampling design.');
  if(name==='sympy') reviewFindings.push('Parsing/evaluation APIs can execute input expressions; accept only trusted mathematical expressions.');
  if(name==='networkx') reviewFindings.push('Persistence docs warn that Python pickle can execute code; use trusted graph artifacts and constrain caller-selected file paths.');
  if(name==='scikit-learn') reviewFindings.push('Persistence docs warn that pickle/joblib artifacts can execute code; load only trusted artifacts.');
  if(name==='architecture-patterns') reviewFindings.push('Documentation-only; use boundary guidance without adopting service/microservice examples absent a project need.');
  if(name==='stride-analysis-patterns') reviewFindings.push('Documentation-only threat taxonomy; identifying threats does not establish that mitigations exist or work.');
  if(['architecture-decision-records','code-review-excellence','debugging-strategies','python-design-patterns','python-packaging','python-testing-patterns','python-type-safety'].includes(name)) reviewFindings.push('Documentation/reference skill with no bundled executable scripts; package commands shown in examples are not run automatically.');
  if(name==='no-ai-slop') reviewFindings.push('Includes an editorial eval rubric and OpenAI skill metadata; no runtime script or hook.');
  entries.push({name,source:{repository:repo,url:`https://github.com/${repo}`,commit,path:upstream,license},purpose:descMatch?.[1]?.trim().replace(/^['"]|['"]$/g,''),installed_path:`.agents/skills/${name}`,compatibility:'Agent Skills SKILL.md format; project-local Codex discovery path. Native activation not tested.',license_files:files.filter(f=>/^license/i.test(f.path)).map(f=>f.path),license_sha256_matches_pinned:licenseMatched,source_comparison:{staged_source_path:path.relative(root,sourceDir).split(path.sep).join('/'),source_file_count:sourceMap.size,matched:sourceMap.size===installedMap.size&&!missingFromInstall.length&&!extraInInstall.length&&!hashMismatches.length,missing_from_install:missingFromInstall,extra_in_install:extraInInstall,hash_mismatches:hashMismatches},file_count:files.length,files,missing_local_references:[...missingRefs].sort(),reference_findings:referenceFindings,static_code_signals:signals,review_findings:reviewFindings,review_status:'STATIC_REVIEW_COMPLETE_WITH_LIMITS',activation_status:'NOT_TESTED'});
}
const result={scope:'Project-local skill bootstrap; no runtime dependencies installed; downloaded scripts were not executed.',total_skills:entries.length,structural_errors:errors,total_files:entries.reduce((s,e)=>s+e.file_count,0),skills:entries};
fs.writeFileSync(out,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({skills:entries.length,errors,totalFiles:result.total_files,manifest:path.relative(root,out)},null,2));
process.exitCode=errors.length?1:0;




