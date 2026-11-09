"use strict";
// Pure comparison helpers shared by the local browser and Node checks.
function wordDiff(before, after) {
  const a = before.match(/\s+|[^\s]+/g) || [], b = after.match(/\s+|[^\s]+/g) || [];
  if (a.length * b.length > 1000000) return [{kind:"removed", text:before}, {kind:"added", text:after}];
  const rows = Array.from({length:a.length+1}, () => new Uint16Array(b.length+1));
  for (let i=a.length-1;i>=0;i--) for (let j=b.length-1;j>=0;j--)
    rows[i][j] = a[i]===b[j] ? rows[i+1][j+1]+1 : Math.max(rows[i+1][j],rows[i][j+1]);
  const result = []; let i=0,j=0;
  const add = (kind,text) => {
    if (result.at(-1)?.kind===kind) result.at(-1).text += text;
    else result.push({kind,text});
  };
  while(i<a.length || j<b.length) {
    if(i<a.length && j<b.length && a[i]===b[j]) {add("same",a[i]);i++;j++;}
    else if(j<b.length && (i===a.length || rows[i][j+1]>=rows[i+1][j])) add("added",b[j++]);
    else add("removed",a[i++]);
  }
  return result;
}
function tokenComparison(before,after) {
  if (!Array.isArray(before) || !Array.isArray(after)) return null;
  let prefix=0;
  while(prefix<Math.min(before.length,after.length) && before[prefix]===after[prefix]) prefix++;
  return {identical: prefix===before.length && prefix===after.length, commonPrefix:prefix};
}
function offsetSetting(item) {
  return item.target_fraction === undefined ? item.sign : item.sign * item.target_fraction;
}
function runProgress(progress) {
  if(!progress?.planned)return '';
  const saved=`${progress.completed??0}/${progress.planned} trajectories saved`;
  return progress.active_trial?`${saved} · ${progress.active_trial.condition} · ${(progress.active_tokens??0).toLocaleString()} generated tokens`:saved;
}
function memoryLabel(answer) {
  if(Number.isFinite(answer.peak_gpu_allocated_bytes))
    return `GPU peak allocated ${(answer.peak_gpu_allocated_bytes/2**30).toFixed(2)} GiB`;
  if(Number.isFinite(answer.mps_current_allocated_bytes))
    return `MPS current allocated ${(answer.mps_current_allocated_bytes/2**30).toFixed(2)} GiB`;
  return 'GPU memory unavailable';
}
function liveMetrics(answer) {
  return `${answer.input_tokens} input · ${answer.output_tokens} output tokens · first token ${answer.time_to_first_token_seconds.toFixed(2)}s · decode ${answer.decode_tokens_per_second?.toFixed(1)??'—'} tok/s · ${memoryLabel(answer)} · load ${answer.load_seconds.toFixed(2)}s · ${answer.terminated_with_eos?'EOS':'output cap'}`;
}
globalThis.LabCompare = {wordDiff,tokenComparison,offsetSetting,runProgress,memoryLabel,liveMetrics};
if (typeof module !== "undefined") module.exports = globalThis.LabCompare;
