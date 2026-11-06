// Execute the served scripts against a small DOM fixture. No GPU or browser needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const testBase = process.env.LAB_TEST_URL || 'http://127.0.0.1:8765';
class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this.style = {}; this.dataset = {};
    this.disabled = false; this.checked = false; this._value = ''; this.textContent = '';
    const classes = new Set();
    this.classList = {toggle(k, on) {if (on) classes.add(k); else classes.delete(k);}, contains: k => classes.has(k)};
  }
  append(...items) {this.children.push(...items);}
  replaceChildren(...items) {this.children = items; this._value = '';}
  add(item) {this.append(item);}
  get options() {return this.children;}
  get value() {return this._value || (this.tagName === 'select' ? this.options[0]?.value || '' : '');}
  set value(x) {this._value = String(x);}
  setAttribute() {}
  closest() {return new Element('label');}
}
async function testStartup(live = false, failed = false) {
  const html = fs.readFileSync(path.join(root, 'ui/index.html'), 'utf8'), elements = {};
  for (const match of html.matchAll(/<(\w+)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const element = elements[match[2]] = new Element(match[1]);
    element.value = /\bvalue="([^"]*)"/.exec(match[0])?.[1] || '';
  }
  Object.assign(elements.mode, {_value: 'joint'});
  const defaultReply = {condition:'baseline',phase:'probe',prompt_id:'cookie',prompt:'Where is my cookie?',text:'Default reply',output_tokens:3,seconds:1,terminated_with_eos:true};
  const edited = {...defaultReply,condition:'joint-profile',profile_id:'villain-x4',persona:'villain',candidate:'joint',dose:4,text:'Edited reply'};
  const fixture = {
    '/api/state': {runs:[{id:'failed-PERF',kind:'performance',outputs:0,model_key:'lfm3'},{id:'CHAR-002',kind:'characters',outputs:2,model_key:'lfm3'},{id:'QWEN-LIVE',kind:'live',outputs:2,model_key:'qwen08'}],jobs:[],extra_designs:{}},
    '/api/models':{lfm3:{label:'LFM3-1.7B',model_id:'LiquidAI/LFM3-1.7B'},qwen08:{label:'Qwen3.5-0.8B',model_id:'Qwen/Qwen3.5-0.8B'}},
    '/api/run?id=QWEN-LIVE':{metadata:{experiment:'QWN-002',model:{model_id:'Qwen/Qwen3.5-0.8B'}},outputs:[
      {...defaultReply,measurements:{exact_math_match:true}},
      {...defaultReply,condition:'published-checkpoint',text:'Wrong final boxed answer',measurements:{exact_math_match:false}}
    ],rankings:{},recipe_refs:{}},
    '/api/performance':{runs:[],identity:null},
    '/api/personas': {recipes:[],runs:{},run_refs:{}},
    '/api/run?id=CHAR-002': {metadata:{experiment:'CHAR-002'},outputs:[defaultReply,edited],rankings:{},recipe_refs:{}}
  };
  const document = {getElementById:id=>elements[id] || null,createElement:tag=>new Element(tag),querySelectorAll:()=>[],addEventListener(){},visibilityState:'visible'};
  const context = vm.createContext({document,window:{addEventListener(){}},Option:class extends Element {constructor(text,value){super('option');this.textContent=text;this.value=value;}},console,setInterval(){},fetch:async url=>{
    if (live) return fetch(testBase+url);
    if (failed && url === '/api/state') throw Error('Test server unavailable');
    assert.ok(url in fixture, 'Unexpected endpoint '+url);
    return {ok:true,json:async()=>fixture[url]};
  }});
  const scripts = [...html.matchAll(/<script\b[^>]*\bsrc="\/([^"]+)"/g)].map(x=>x[1]);
  for (const script of scripts) vm.runInContext(fs.readFileSync(path.join(root,'ui',script),'utf8'),context,{filename:script});
  // Wait for startup requests and their render continuations.
  for(let i=0;i<100;i++) {
    await new Promise(resolve=>setTimeout(resolve,live?50:1));
    if(elements.activityText.textContent==='Connection error' || elements.comparisonColumns.children.some(x=>x.tagName==='article')) break;
  }
  if (failed) {
    assert.equal(elements.activityText.textContent,'Connection error');
    assert.match(elements.message.textContent,/unavailable/);
  } else {
    for(const id of ['promptFilter','cohortFilter','compareA','compareB']) assert.ok(elements[id].options.length, id+' must have options');
    assert.equal(elements.promptFilter.disabled,false);
    assert.equal(elements.compareA.disabled,false);
    assert.ok(elements.comparisonColumns.children.some(x=>x.tagName==='article'),'Default and edited replies render');
    assert.ok(['GPU lab idle','Experiment running','Review only · inference disabled'].includes(elements.activityText.textContent));
    if(!live) assert.equal(vm.runInContext('selected',context),'CHAR-002','Skip empty/performance runs on startup');
    console.log(live?'Live server startup passed:':'Fixture startup passed:',Object.fromEntries(['promptFilter','cohortFilter','compareA','compareB'].map(id=>[id,elements[id].options.length])));
    elements.modelSelect.value='qwen08';await elements.modelSelect.onchange();
    assert.ok(vm.runInContext('modelRuns().every(x=>x.model_key==="qwen08")',context));
    assert.ok(elements.runs.children.length,'Qwen saved runs available');
    assert.equal(elements.launchButton.disabled,true,'LFM batch designs disabled under Qwen');
    assert.equal(vm.runInContext('current.metadata.model.model_id',context),'Qwen/Qwen3.5-0.8B');
    if(!live) {
      const descendants=node=>[node,...node.children.flatMap(descendants)];
      const texts=descendants(elements.comparisonColumns).map(node=>node.textContent);
      assert.ok(texts.includes('Math answer matched'),'Qwen math grading must be visible');
      assert.ok(texts.includes('Math answer mismatch'),'Qwen math regressions must be visible');
    }
    await vm.runInContext('refreshPerformance()',context);
    assert.match(elements.identityStatus.textContent,/No long-context performance measurements/);
    assert.equal(elements.performanceData.children.length,0,'LFM performance must not appear as Qwen performance');
    elements.modelSelect.value='lfm3';await elements.modelSelect.onchange();
    assert.ok(vm.runInContext('modelRuns().every(x=>x.model_key==="lfm3")',context));
    console.log('Model switching and performance separation pass.');
  }
}
(async()=>{await testStartup();await testStartup(false,true);if(process.argv.includes('--live'))await testStartup(true);})().catch(e=>{console.error(e);process.exitCode=1;});
