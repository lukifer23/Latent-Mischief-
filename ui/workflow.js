"use strict";
let personaData={recipes:[],runs:{},run_refs:{}}, personaSelection="", perfData={runs:[]};
let modelsData={},selectedModelKey='lfm3',liveBusy=false,liveJob='',livePrompt='',liveBranches={},liveHistory={baseline:[],edited:[]};
const modelRuns=()=> (state.runs||[]).filter(x=>x.model_key===selectedModelKey);
const selectedModelId=()=>modelsData[selectedModelKey]?.model_id;
const recipeEntry=id=>personaData.recipes.find(x=>x.id===id);
const outputKey=x=>x.profile_id || [x.condition,x.persona,x.candidate,x.layer,x.gain,x.offset,x.dose,x.control].filter(v=>v!==undefined&&v!==null).join(":");
const outputName=(x,index)=>{
  const entry=recipeEntry(current?.recipe_refs?.[String(index)]);
  return entry?.observed_name ? entry.observed_name+" · "+conditionLabel(x) : conditionLabel(x);
};
function showTab(name) {
  $('runContext').classList.toggle('hidden',!['review','atlas'].includes(name));
  document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('selected',b.dataset.tab===name));
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.toggle('hidden',p.id!==name));
  if(name==='library') renderLibrary();
  if(name==='performance') refreshPerformance().catch(e=>message(e.message));
  if(name==='logs') refreshLog();
}
async function refreshPersonas() { personaData=await api('/api/personas'); renderLibrary(); fillLiveRecipes(); if(current) renderOutputs(); }
function readable(text) { return $('rawOutput').checked?text:text.replace(/<\|im_end\|>|<\|endoftext\|>/g,'').trim(); }
function renderOutputs() {
  if(!current) return;
  const prompt=$('promptFilter').value;
  const baseline=current.outputs.find(x=>(x.prompt_id||x.id)===prompt && x.condition==='baseline');
  const choices=current.outputs.map((x,index)=>({...x,_index:index})).filter(x=>(!x.phase||x.phase==='probe') &&
    (x.prompt_id||x.id)===prompt && x.condition!=='baseline' &&
    ($('personaFilter').value==='all'||x.persona===$('personaFilter').value||x.role==='sparse-random'||x.condition==='random-direction') &&
    ($('conditionFilter').value==='all'||x.condition===$('conditionFilter').value) &&
    ($('layerFilter').value==='all'||String(x.layer)===$('layerFilter').value) &&
    ($('doseFilter').value==='all'||String(x.target_fraction??x.dose)===$('doseFilter').value) &&
    ($('roleFilter').value==='all'||x.role===$('roleFilter').value));
  for(const id of ['compareA','compareB']) {
    const select=$(id),old=select.value; select.replaceChildren();
    if(id==='compareB') select.add(new Option('None — compare one edit','none'));
    for(const x of choices) select.add(new Option(outputName(x,x._index),outputKey(x)));
    select.disabled=!choices.length;
    if(!choices.length&&id==='compareA')select.add(new Option('No edited replies for this selection',''));
    if([...select.options].some(x=>x.value===old)) select.value=old;
    else {
      const preferred=id==='compareA'?(choices.find(x=>recipeEntry(current.recipe_refs?.[String(x._index)])?.observed_name)||choices.find(x=>x.persona==='villain'&&x.condition!=='prompt-persona')||choices[0]):choices.find(x=>x.condition==='prompt-persona'&&x.persona==='villain');
      if(preferred) select.value=outputKey(preferred);
    }
  }
  $('comparisonColumns').replaceChildren();
  if(!baseline) {
    $('comparisonColumns').append(el('div','empty-state',current.calibration?'This is a calibration screen. Open Effect Atlas to inspect its measurements.':'Select a prompt with a recorded default response.'));
    return;
  }
  $('promptText').textContent=baseline.prompt||prompt;
  const picked=[{...baseline,_index:current.outputs.indexOf(baseline),_reference:true},
    ...['compareA','compareB'].map(id=>choices.find(x=>outputKey(x)===$(id).value)).filter(Boolean)];
  for(const output of picked) {
    const card=el('article','response-card'),head=el('div','response-head');
    const id=current.recipe_refs?.[String(output._index)],entry=recipeEntry(id);
    head.append(el('small','',output._reference?'DEFAULT · NO EDIT':entry?.name_status==='provisional'?'PROVISIONAL OBSERVATION':'EDITED CONDITION'),el('h3','',output._reference?'Default':outputName(output,output._index)));
    const metrics=el('div','tags');
    const checks=Object.entries(output.measurements||{}).filter(([k])=>['exact_json_match','exact_text_match','exact_math_match'].includes(k));
    const checkText=checks.length?(checks[0][0]==='exact_math_match'?(checks[0][1]?'Math answer matched':'Math answer mismatch'):(checks[0][1]?'Exact check passed':'Exact check failed')):'No exact check';
    for(const text of [`${output.output_tokens} tokens`,`${output.output_tokens-baseline.output_tokens>=0?'+':''}${output.output_tokens-baseline.output_tokens} vs default`,
      output.terminated_with_eos?'Finished (EOS)':'Output cap / no EOS',checkText,`${output.seconds.toFixed(2)}s`]) metrics.append(el('span','tag',text));
    head.append(metrics); card.append(head);
    const body=el('div','response-body');
    if($('showDiff').checked&&!output._reference) {
      for(const part of LabCompare.wordDiff(readable(baseline.text),readable(output.text))) body.append(el('span','diff-'+part.kind,part.text));
    } else body.textContent=readable(output.text);
    body.classList.toggle('raw', $('rawOutput').checked); card.append(body);
    const actions=el('div','response-actions');
    if(id) {const open=el('button','secondary','Name & methodology');open.onclick=()=>{personaSelection=id;showTab('library');};actions.append(open);}
    if(!output._reference) {
      const details=el('details','review-note'),label=el('select'),note=el('textarea'),save=el('button','secondary','Save review');
      label.setAttribute('aria-label','Review label');note.setAttribute('aria-label','Review notes');note.rows=2;
      ['unreviewed','interesting','no-effect','damaged','needs-review'].forEach(x=>label.add(new Option(x,x)));
      const key=selected+':'+output._index,annotation=drafts[key]||current.annotations?.[String(output._index)];label.value=annotation?.label||'unreviewed';note.value=annotation?.note||'';
      label.onchange=note.oninput=()=>drafts[key]={label:label.value,note:note.value};
      save.onclick=async()=>{try{await api('/api/annotations',{run:selected,index:output._index,label:label.value,note:note.value});current.annotations??={};current.annotations[output._index]={label:label.value,note:note.value};delete drafts[key];message('Review saved locally.');}catch(e){message(e.message);}};
      details.append(el('summary','','Review this reply'),label,note,save);actions.append(details);
    }
    card.append(actions);$('comparisonColumns').append(card);
  }
}
function renderLibrary() {
  $('personaList').replaceChildren();
  const query=$('personaSearch').value.toLowerCase();
  const entries=personaData.recipes.filter(x=>(!selectedModelId()||x.recipe.model.model_id===selectedModelId())&&[x.observed_name,x.codename,x.method_label,x.id,x.recipe.family].join(' ').toLowerCase().includes(query))
    .sort((a,b)=>Number(!!b.observed_name)-Number(!!a.observed_name)||(a.observed_name||a.codename).localeCompare(b.observed_name||b.codename));
  $('libraryCount').textContent=`${entries.length} matching recipes · names stay separate from evidence`;
  if(!entries.some(x=>x.id===personaSelection))personaSelection=entries[0]?.id||'';
  for(const entry of entries) {const b=el('button','library-item'+(entry.id===personaSelection?' selected':''));b.append(el('strong','',entry.observed_name||entry.codename),el('small','',entry.method_label),el('small','',entry.name_status+' · '+entry.id));b.onclick=()=>{personaSelection=entry.id;renderLibrary();};$('personaList').append(b);}
  const entry=recipeEntry(personaSelection);$('personaDetail').replaceChildren();if(!entry)return;
  const panel=$('personaDetail');panel.append(el('small','eyebrow',entry.id),el('h2','',entry.observed_name||entry.codename),el('p','muted',entry.method_label));
  const name=el('input'),status=el('select'),note=el('textarea'),save=el('button','','Save local persona name');
  name.value=entry.observed_name;name.maxLength=80;name.placeholder=entry.codename;name.setAttribute('aria-label','Observed persona nickname');
  ['unreviewed','reference-only','provisional','replicated-description','rejected'].forEach(x=>status.add(new Option(x,x)));status.value=entry.name_status;status.setAttribute('aria-label','Human documentation status');
  note.value=entry.naming_notes;note.rows=3;note.maxLength=4000;note.setAttribute('aria-label','Naming evidence and limitations');
  const form=el('div','name-form');form.append(el('label','','Observed nickname'),name,el('label','','Documentation status (human annotation)'),status,el('label','','What changed, where, and what failed?'),note,save);
  save.onclick=async()=>{try{await api('/api/personas',{id:entry.id,observed_name:name.value,name_status:status.value,naming_notes:note.value});await refreshPersonas();message('Name saved locally. Exact recipe and raw evidence are unchanged.');}catch(e){message(e.message);}};panel.append(form);
  panel.append(el('h3','','Compared with default'));
  for(const o of entry.observations) {
    const block=el('div','observation');block.append(el('strong','',`${personaData.runs[o.run]?.experiment||o.run} · ${o.cohort}`),
      el('p','',`${o.changed_sequences}/${o.n} sequences changed · tokens ${o.default_mean_tokens.toFixed(1)} → ${o.mean_tokens.toFixed(1)} · EOS ${o.default_eos} → ${o.eos} / ${o.n}`),
      el('small','',o.exact_n?`Exact checks ${o.default_exact_pass} → ${o.exact_pass} / ${o.exact_n}`:'Exact checks not measured'));
    const open=el('button','secondary','Compare this evidence');open.onclick=async()=>{await loadRun(o.run);const x=current.outputs[o.output_indices[0]];$('promptFilter').value=x.prompt_id||x.id;renderOutputs();$('compareA').value=outputKey(x);renderOutputs();showTab('review');};block.append(open);panel.append(block);
  }
  for(const [title,value] of [['Exact change',entry.recipe],['Method derivation & sources',entry.methods],['Runtime and generation settings',entry.observations.map(o=>({run:o.run,cohort:o.cohort,system:o.system,default_system:o.default_system,generation:o.generation,environment:personaData.runs[o.run]?.environment}))]]) {
    const details=el('details','panel');details.append(el('summary','',title),el('pre','',JSON.stringify(value,null,2)));panel.append(details);
  }
}
async function refreshPerformance() {
  perfData=await api('/api/performance');const holder=$('performanceData');holder.replaceChildren();
  const identity=perfData.identity;
  $('identityStatus').textContent=selectedModelKey==='lfm3'&&identity&&Object.values(identity.checks).every(Boolean)?`Verified LiquidAI/LFM3-1.7B · pinned weights match Hub SHA-256 · declared window ${identity.architecture.max_position_embeddings.toLocaleString()} tokens`:'No long-context performance measurements for this selected model yet.';
  for(const run of perfData.runs.filter(x=>x.metadata.model?.model_id===selectedModelId())) {
    const section=el('article','panel');section.append(el('h3','',run.id),el('p','muted',`${run.metadata.runtime_attention} · ${run.metadata.experiment_config.prefill_chunk_tokens}-token prefill chunks · one timing per point · ${run.summary?'completed':'partial / interrupted'}`));
    if(run.interruption)section.append(el('p','notice',`${run.interruption.attempted_input_tokens?run.interruption.attempted_input_tokens.toLocaleString()+' input tokens: ':''}${run.interruption.reason}`));
    if(!run.rows.length) section.append(el('p','muted','No completed measurements. Inspect preflight/job logs.'));
    const table=el('table','data-table'),head=el('tr');['Input tokens','Change','Prefill','First token','Decode','GPU peak allocated / reserved','Host RSS end','Status'].forEach(x=>head.append(el('th','',x)));table.append(head);
    for(const row of run.rows) {const tr=el('tr');for(const value of [row.actual_input_tokens.toLocaleString(),row.condition,row.prefill_seconds?.toFixed(2)+'s',row.time_to_first_token_seconds?.toFixed(2)+'s',row.decode_tokens_per_second===null?'—':row.decode_tokens_per_second?.toFixed(1)+' tok/s',`${(row.peak_gpu_allocated_bytes/2**30).toFixed(2)} / ${(row.peak_gpu_reserved_bytes/2**30).toFixed(2)} GiB`,row.process_rss_end_bytes?`${(row.process_rss_end_bytes/2**30).toFixed(2)} GiB`:'—',row.status])tr.append(el('td','',value));table.append(tr);}section.append(table);holder.append(section);
  }
}
function initWorkflow() {
  document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));
  for(const id of ['compareA','compareB','showDiff','rawOutput'])$(id).onchange=renderOutputs;
  $('personaSearch').oninput=renderLibrary;$('refreshPersonas').onclick=()=>refreshPersonas().catch(e=>message(e.message));
  $('refreshPerformance').onclick=()=>refreshPerformance().catch(e=>message(e.message));
  refreshPersonas().catch(e=>message(e.message));
  initLive().catch(e=>message(e.message));showTab('chat');
}

function fillLiveRecipes() {
  const select=$('liveRecipe'),old=select.value;select.replaceChildren(new Option('Unchanged control (default twice)','default'));
  for(const entry of personaData.recipes.filter(x=>x.recipe.model.model_id===selectedModelId()).sort((a,b)=>Number(!!b.observed_name)-Number(!!a.observed_name)||(a.observed_name||a.codename).localeCompare(b.observed_name||b.codename))) select.add(new Option(`${entry.observed_name||entry.codename} · ${entry.method_label}`,entry.id));
  if([...select.options].some(x=>x.value===old))select.value=old;
  select.disabled=liveBusy;renderLive();
}
function renderLive() {
  $('chatColumns').replaceChildren();
  for(const branch of ['baseline','edited']) {
    const card=el('article','response-card');card.append(el('h3','response-head',branch==='baseline'?'Default':[...$('liveRecipe').options].find(x=>x.value===$('liveRecipe').value)?.textContent||'Edited'));
    const body=el('div','response-body');
    for(const item of liveHistory[branch])body.append(el('div','chat-'+item.role,item.content));
    if(livePrompt)body.append(el('div','chat-user',livePrompt));
    const answer=liveBranches[branch];
    if(answer)body.append(el('div','chat-assistant',answer.text.replace(/<\|im_end\|>|<\|endoftext\|>/g,'').trim()));
    if(!liveHistory[branch].length&&!livePrompt)body.append(el('p','muted','Your conversation will appear here.'));
    card.append(body);
    if(answer&&!answer.partial)card.append(el('p','live-metrics',`${answer.input_tokens} input · ${answer.output_tokens} output tokens · first token ${answer.time_to_first_token_seconds.toFixed(2)}s · decode ${answer.decode_tokens_per_second?.toFixed(1)??'—'} tok/s · GPU allocated ${(answer.peak_gpu_allocated_bytes/2**30).toFixed(2)} GiB · load ${answer.load_seconds.toFixed(2)}s · ${answer.terminated_with_eos?'EOS':'output cap'}`));
    $('chatColumns').append(card);
  }
}
function setLiveBusy(busy) {
  liveBusy=busy;
  for(const id of ['sendChat','newChat','modelSelect','liveRecipe','liveTokens','liveThinking','liveSystem','chatPrompt'])$(id).disabled=busy;
  if(state.read_only)$('sendChat').disabled=true;
  $('stopChat').disabled=!busy;
}
function clearChat() {liveHistory={baseline:[],edited:[]};liveBranches={};livePrompt='';renderLive();$('chatStatus').textContent='New conversation. Both histories cleared.';}
async function pollLive() {
  try {
    const result=await api('/api/live?id='+encodeURIComponent(liveJob));
    liveBranches=result.branches||{};renderLive();$('chatStatus').textContent=result.error||result.stage||result.status;
    if(['queued','running'].includes(result.status)){setTimeout(pollLive,800);return;}
    if(result.status==='complete') {
      for(const branch of ['baseline','edited'])liveHistory[branch].push({role:'user',content:livePrompt},{role:'assistant',content:liveBranches[branch].text.replace(/<\|im_end\|>|<\|endoftext\|>/g,'').trim()});
      livePrompt='';liveBranches={};renderLive();
      // Keep final per-branch metrics visible after saving the conversation.
      for(const [index,branch] of ['baseline','edited'].entries()) {
        const answer=result.branches[branch];$('chatColumns').children[index].append(el('p','live-metrics',`${answer.input_tokens} input · ${answer.output_tokens} output · first token ${answer.time_to_first_token_seconds.toFixed(2)}s · decode ${answer.decode_tokens_per_second?.toFixed(1)??'—'} tok/s · peak allocated ${(answer.peak_gpu_allocated_bytes/2**30).toFixed(2)} GiB · load ${answer.load_seconds.toFixed(2)}s · ${answer.terminated_with_eos?'EOS':'output cap'}`));
      }
      $('chatStatus').textContent='Both replies saved. Send another prompt or open Saved Comparisons.';await refresh();
    }
    setLiveBusy(false);
  }catch(e){setLiveBusy(false);message(e.message);}
}
async function initLive() {
  modelsData=await api('/api/models');const select=$('modelSelect');select.replaceChildren();
  for(const [key,spec]of Object.entries(modelsData))select.add(new Option(spec.label,key));select.value=selectedModelKey;select.disabled=false;
  select.onchange=async()=>{selectedModelKey=select.value;selected='';current=null;personaSelection='';$('comparisonColumns').replaceChildren();for(const id of ['promptFilter','cohortFilter','compareA','compareB'])fillSelect(id,[],null);$('runTitle').textContent='Select a '+modelsData[selectedModelKey].label+' run';$('modelBadge').textContent=modelsData[selectedModelKey].label+' · LOCAL LAB';clearChat();fillLiveRecipes();renderLibrary();await refresh();if(!$('performance').classList.contains('hidden'))await refreshPerformance();};
  fillLiveRecipes();$('newChat').onclick=clearChat;$('liveRecipe').onchange=clearChat;
  $('chatForm').onsubmit=async event=>{event.preventDefault();if(liveBusy)return;livePrompt=$('chatPrompt').value.trim();if(!livePrompt)return;setLiveBusy(true);liveBranches={};renderLive();$('chatStatus').textContent='Queued. First use loads the selected checkpoint.';
    try{const job=await api('/api/live',{model_key:selectedModelKey,recipe_id:$('liveRecipe').value,prompt:livePrompt,system:$('liveSystem').value,max_new_tokens:Number($('liveTokens').value),thinking:$('liveThinking').checked,history_default:liveHistory.baseline,history_edited:liveHistory.edited});liveJob=job.id;pollLive();}catch(e){setLiveBusy(false);message(e.message);}
  };
  $('stopChat').onclick=async()=>{try{await api('/api/live/stop',{});}catch(e){message(e.message);}};
}
