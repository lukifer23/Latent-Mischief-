const assert = require('node:assert/strict');
const {wordDiff,tokenComparison,offsetSetting} = require('../ui/compare.js');
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
