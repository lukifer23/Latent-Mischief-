const assert = require('node:assert/strict');
const {wordDiff,tokenComparison} = require('../ui/compare.js');
for(const [before,after] of [['hello world','hello villain'],['','new'],['old',''],['a  b','a b'],['<script>x</script>','<img>']]) {
  const parts=wordDiff(before,after);
  assert.equal(parts.filter(x=>x.kind!=='added').map(x=>x.text).join(''),before);
  assert.equal(parts.filter(x=>x.kind!=='removed').map(x=>x.text).join(''),after);
}
assert.deepEqual(tokenComparison([1,2,3],[1,2,4]),{identical:false,commonPrefix:2});
assert.deepEqual(tokenComparison([1],[1]),{identical:true,commonPrefix:1});
assert.equal(tokenComparison(null,[1]),null);
console.log('Comparison reconstruction and token-divergence checks pass.');
