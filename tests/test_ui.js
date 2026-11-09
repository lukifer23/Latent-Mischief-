const assert = require('node:assert/strict');
const {wordDiff,tokenComparison,offsetSetting,runProgress,memoryLabel,liveMetrics} = require('../ui/compare.js');
for(const [before,after] of [['hello world','hello villain'],['','new'],['old',''],['a  b','a b'],['<script>x</script>','<img>']]) {
  const parts=wordDiff(before,after);
  assert.equal(parts.filter(x=>x.kind!=='added').map(x=>x.text).join(''),before);
  assert.equal(parts.filter(x=>x.kind!=='removed').map(x=>x.text).join(''),after);
}
assert.deepEqual(tokenComparison([1,2,3],[1,2,4]),{identical:false,commonPrefix:2});
assert.deepEqual(tokenComparison([1],[1]),{identical:true,commonPrefix:1});
assert.equal(tokenComparison(null,[1]),null);
assert.notEqual(offsetSetting({sign:1,target_fraction:.05}),offsetSetting({sign:1,target_fraction:.2}));
assert.notEqual(offsetSetting({sign:-1,target_fraction:.2}),offsetSetting({sign:1,target_fraction:.2}));
console.log('Comparison reconstruction and token-divergence checks pass.');
assert.equal(runProgress(null),'');
assert.match(runProgress({completed:2,planned:8,active_trial:{condition:'published-checkpoint'},active_tokens:4096}),/2\/8 trajectories saved.*4,096 generated tokens/);
assert.equal(memoryLabel({peak_gpu_allocated_bytes:null}),'GPU memory unavailable');
assert.equal(memoryLabel({peak_gpu_allocated_bytes:0}),'GPU peak allocated 0.00 GiB');
assert.equal(memoryLabel({peak_gpu_allocated_bytes:null,mps_current_allocated_bytes:2**30}),'MPS current allocated 1.00 GiB');
assert.match(liveMetrics({input_tokens:4,output_tokens:3,time_to_first_token_seconds:1,decode_tokens_per_second:null,load_seconds:2,terminated_with_eos:true}),/GPU memory unavailable.*EOS/);
