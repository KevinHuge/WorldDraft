// Run with: node eval/tests/test_report_grid.cjs (from the repo root, after aggregate.py)
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const page = fs.readFileSync('eval/results/benchmark/report/tables/per_case.html','utf8');
const script = [...page.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).find(s=>s.includes('const GRID_DATA'));
assert.ok(script, 'Interactive grid must be embedded in the standalone report');
const context = vm.createContext({assert});
vm.runInContext(script, context);
vm.runInContext(`
const fixture = Object.fromEntries(GRID_CELLS.map((cell,i)=>[cell,{RF_total:i,SL_color:i}]));
assert.equal(aggregateGrid(fixture,'RF_total',GRID_CELLS).value,4);
assert.deepEqual(groupCells('merge-a').map(g=>aggregateGrid(fixture,'RF_total',g.cells).value),[1,4,7]);
assert.deepEqual(groupCells('merge-p').map(g=>aggregateGrid(fixture,'RF_total',g.cells).value),[3,4,5]);
assert.equal(aggregateGrid(fixture,'RF_total',['P0_AN']).value,0);
assert.equal(aggregateGrid(fixture,'SL_color',GRID_CELLS).value,5);
assert.equal(aggregateGrid(fixture,'SL_color',['P0_AN']).expected,0);
delete fixture.P0_AN.RF_total;
assert.equal(aggregateGrid(fixture,'RF_total',GRID_CELLS).value,null);
assert.equal(aggregateGrid(fixture,'RF_total',GRID_CELLS).available,8);
assert.equal(aggregateGrid({},'RF_total',GRID_CELLS).value,null);
let checked=0;
for(const cases of Object.values(GRID_DATA.values)) for(const scores of Object.values(cases)) for(const metric of GRID_DATA.metrics) {
 const total=aggregateGrid(scores,metric,GRID_CELLS);
 if(total.value===null) continue;
 for(const mode of ['merge-a','merge-p']) {
  const groups=groupCells(mode).map(g=>aggregateGrid(scores,metric,g.cells)).filter(r=>r.expected);
  assert.ok(groups.every(r=>r.value!==null));
  const merged=groups.reduce((sum,r)=>sum+r.value*r.expected,0)/groups.reduce((sum,r)=>sum+r.expected,0);
  assert.ok(Math.abs(merged-total.value)<1e-8);
  checked++;
 }
}
assert.ok(checked>10000);
`,context);
// Exercise control changes and click drill-down with a small DOM substitute.
const elements={};
for(const id of ['model','case','metric','mode','note','view','detail','detail-title','reset','export']) {
 const key=id==='detail-title'?id:'grid-'+id;
 elements[key]={value:id==='mode'?'grid':'',innerHTML:'',textContent:'',events:{},
  selectedOptions:[{text:'Grid'}],addEventListener(name,fn){this.events[name]=fn;}};
}
Object.defineProperty(elements['grid-case'],'innerHTML',{set(v){this.value='A01';}});
const browserContext=vm.createContext({document:{getElementById:id=>{assert.ok(elements[id],id);return elements[id];}},setTimeout});
vm.runInContext(script,browserContext);
assert.equal((elements['grid-view'].innerHTML.match(/data-cell=/g)||[]).length,9);
for(const mode of ['merge-a','merge-p','all','grid']) {
 elements['grid-mode'].value=mode;
 elements['grid-mode'].events.change();
 assert.ok(elements['grid-detail'].innerHTML.includes('RF↑'));
}
elements['grid-view'].events.click({target:{closest:()=>({dataset:{cell:'P1_AC'}})}});
assert.ok(elements['detail-title'].textContent.includes('P1_AC'));
elements['grid-reset'].events.click();
assert.ok(!elements['detail-title'].textContent.includes('P1_AC'));
console.log('Passed: zero/missing/SL, all published complete-case aggregates, controls and cell drill-down.');
