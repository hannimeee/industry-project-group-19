'use strict';
(() => {
 const short=['Technology','Home & garden','Fashion & care','Arts & gifts','Auto & outdoor'];
 function show(kind){
  const matrix=data.correlations[kind],revenue=kind==='revenue';
  const lo=Math.min(...matrix.values.flat()),hi=Math.max(...matrix.values.flat());
  for(const key of ['revenue','customers'])$('relationship-'+key).setAttribute('aria-pressed',String(key===kind));
  $('relationship-description').textContent=revenue?'Mean pairwise Pearson correlation of monthly merchant transaction revenue. Higher values indicate more similar revenue movements.':'Mean pairwise Jaccard similarity of merchant customer sets: shared customers divided by all distinct customers in the pair. Higher values indicate more overlap.';
  $('relationship-scale').textContent=revenue?'Pearson scale: −1 to +1. Undefined correlations (insufficient shared months or zero variance) are excluded. Hover over a cell for exact values and pair counts.':'Jaccard scale: 0 to 1. Self-comparisons are excluded. Hover over a cell for exact values and pair counts.';
  $('relationship-scale').textContent+=` Colour shading is scaled within this matrix: light ${lo.toFixed(4)} → dark ${hi.toFixed(4)}; compare numeric values across measures.`;
  $('relationship-matrix').innerHTML='<table><caption class="sr-only">'+(revenue?'Revenue correlation':'Customer overlap')+' by segment</caption><thead><tr><th scope="col">Segment</th>'+matrix.categories.map((c,i)=>`<th scope="col" title="${esc(c)}">${short[i]}</th>`).join('')+'</tr></thead><tbody>'+matrix.values.map((row,i)=>'<tr><th scope="row">'+esc(matrix.categories[i])+'</th>'+row.map((v,j)=>{
   const d=matrix.diagnostics.find(d=>(d.segment_a===matrix.categories[i]&&d.segment_b===matrix.categories[j])||(d.segment_b===matrix.categories[i]&&d.segment_a===matrix.categories[j]));
   const title=`${matrix.categories[i]} × ${matrix.categories[j]}: ${v}. ${d?`${d.valid_pairs.toLocaleString()} valid pairs; ${d.excluded_pairs.toLocaleString()} excluded.`:''}`;
   return `<td style="background:rgba(${v<0?'0,172,190':'79,47,152'},${.06+.76*((v-lo)/(hi-lo||1))})" title="${esc(title)}"><span style="color:${(v-lo)/(hi-lo||1)>.55?'white':'#271866'}">${v.toFixed(4)}</span></td>`;
  }).join('')+'</tr>').join('')+'</tbody></table>';
 }
 $('relationship-revenue').addEventListener('click',()=>show('revenue'));
 $('relationship-customers').addEventListener('click',()=>show('customers'));
 show('revenue');
})();
