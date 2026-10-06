'use strict';
(() => {
 const section=$('distributions'),host=$('distribution-charts'),popup=$('distribution-popup');
 const specs=[
  {key:'avg_consumer_district_income',name:'Consumer average district income',unit:'AUD / week',note:'Area-income estimate, not individual income.',format:v=>money(v),linear:true},
  {key:'avg_consumer_fraud_prob',name:'Average consumer fraud score',unit:'Percent',note:'Stored percentage points, averaged over merchant transactions.',format:v=>v.toFixed(2)+'%',linear:true},
  {key:'total_transactions',name:'Total transactions',unit:'Transactions',note:'Count over the curated transaction history, not per month.',format:v=>Math.round(v).toLocaleString('en-AU')},
  {key:'avg_value_per_transaction',name:'Average transaction value',unit:'AUD / transaction',note:'Curated total transaction value divided by transaction count.',format:v=>money(v)},
 ];
 let plotted=[],timer=null,returnFocus=null;
 function close(){clearTimeout(timer);popup.hidden=true;}
 function build(spec){
  const pool=data.portfolios.filter(p=>p.scope===portfolio().scope).flatMap(p=>p.merchants).map(m=>m[spec.key]).filter(v=>Number.isFinite(v)&&v>=0).sort((a,b)=>a-b);
  const valid=portfolio().merchants.filter(m=>Number.isFinite(m[spec.key])&&m[spec.key]>=0);
  if(!pool.length)return {spec,bins:[],valid,log:false};
  const min=pool[0],max=pool[pool.length-1],median=pool[Math.floor(pool.length/2)];
  const log=!spec.linear&&min>0&&max/min>30&&max/Math.max(median,1e-12)>8;
  const tx=v=>log?Math.log10(v):v,inv=v=>log?10**v:v;
  let low=tx(min),high=tx(max);const spread=high-low;
  if(!spread){low=log?low-.05:Math.max(0,low-Math.max(Math.abs(low)*.05,.5));high=log?high+.05:high+Math.max(Math.abs(high)*.05,.5);}
  else{low=log?low-spread*.025:Math.max(0,low-spread*.025);high+=spread*.025;}
  const bins=Array.from({length:10},(_,i)=>({low:inv(low+(high-low)*i/10),high:inv(low+(high-low)*(i+1)/10),members:[]}));
  for(const m of valid){const i=Math.max(0,Math.min(9,Math.floor((tx(m[spec.key])-low)/(high-low)*10)));bins[i].members.push(m);}
  return {spec,bins,valid,log,low,high,inv};
 }
 function tick(spec,v){if(spec.key==='avg_consumer_fraud_prob')return v.toFixed(1)+'%';return new Intl.NumberFormat('en-AU',{notation:'compact',maximumFractionDigits:1}).format(v);}
 function render(){
  close();if(!section.open)return;
  plotted=specs.map(build);
  $('distribution-scope').textContent=`${data.scopes.find(s=>s.id===portfolio().scope).name} · α = ${pct(portfolio().retention)} · ${portfolio().merchants.length} merchants`;
  host.innerHTML=plotted.map((d,k)=>{
   const {spec,bins,valid}=d;if(!bins.length)return `<article class="distribution-card"><h3>${esc(spec.name)}</h3><p>No available values.</p></article>`;
   const maxCount=Math.max(1,...bins.map(b=>b.members.length));
   const top=Math.max(2,Math.ceil(maxCount/4)*4),y=v=>174-v/top*128;
   let svg=`<svg viewBox="0 0 520 225" role="group" aria-label="${esc(spec.name)} histogram; ${d.log?'logarithmic':'linear'} horizontal scale"><text x="36" y="18">Merchants</text>`;
   for(let i=0;i<=2;i++){const n=top*i/2;svg+=`<line x1="38" y1="${y(n)}" x2="490" y2="${y(n)}" stroke="#e9e5f2"/><text x="28" y="${y(n)+4}" text-anchor="end">${Number(n.toFixed(1))}</text>`;}
   bins.forEach((bin,i)=>{const h=174-y(bin.members.length);svg+=`<rect class="histogram-bar" data-chart="${k}" data-bin="${i}" x="${40+i*45}" y="${174-Math.max(h,3)}" width="40" height="${Math.max(h,3)}" rx="2" fill="${bin.members.length?'#7a65bb':'#efecf6'}" tabindex="0" role="button" aria-label="${esc(spec.format(bin.low))} to ${esc(spec.format(bin.high))}: ${bin.members.length} merchants. Open merchant list."/>`;});
   for(let i=0;i<5;i++){const v=d.inv(d.low+(d.high-d.low)*i/4);svg+=`<text x="${40+450*i/4}" y="195" text-anchor="middle">${esc(tick(spec,v))}</text>`;}
   svg+=`<text x="265" y="218" text-anchor="middle">${esc(spec.unit)} · ${d.log?'log10 scale':'linear scale'}</text></svg>`;
   return `<article class="distribution-card"><div class="distribution-card-title"><h3>${esc(spec.name)}</h3><span>${valid.length} / ${portfolio().merchants.length} values</span></div>${svg}<p>${esc(spec.note)}</p></article>`;
  }).join('');
  host.querySelectorAll('.histogram-bar').forEach(bar=>{
   bar.addEventListener('pointerenter',()=>show(bar));bar.addEventListener('pointerleave',scheduleClose);
   bar.addEventListener('focus',()=>show(bar));
   bar.addEventListener('click',()=>{show(bar);$('distribution-popup-close').focus({preventScroll:true});});
   bar.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show(bar);$('distribution-popup-close').focus({preventScroll:true});}if(e.key==='Escape')close();});
  });
 }
 function scheduleClose(){timer=setTimeout(()=>{if(!popup.contains(document.activeElement))close();},220);}
 function show(bar){
  clearTimeout(timer);returnFocus=bar;
  const d=plotted[Number(bar.dataset.chart)],bin=d.bins[Number(bar.dataset.bin)];
  $('distribution-popup-title').textContent=`${d.spec.format(bin.low)}–${d.spec.format(bin.high)} · ${bin.members.length} merchants`;
  const sorted=bin.members.slice().sort((a,b)=>b[d.spec.key]-a[d.spec.key]);
  $('distribution-popup-list').innerHTML=sorted.length?sorted.map(m=>`<button type="button" data-rank="${m.rank}" data-abn="${esc(m.abn)}" title="Show ${esc(m.name)} in the ranking"><span>#${m.rank} ${esc(m.name)}</span><strong>${esc(d.spec.format(m[d.spec.key]))}</strong><small>Show in ranking ↗</small></button>`).join(''):'<p>No merchants in this interval.</p>';
  popup.hidden=false;
  const r=bar.getBoundingClientRect(),w=popup.offsetWidth,h=popup.offsetHeight;
  popup.style.left=Math.max(12,Math.min(r.left,window.innerWidth-w-12))+'px';
  popup.style.top=Math.max(12,Math.min(r.bottom+8,window.innerHeight-h-12))+'px';
 }
 popup.addEventListener('pointerenter',()=>clearTimeout(timer));popup.addEventListener('pointerleave',scheduleClose);
 popup.addEventListener('keydown',e=>{if(e.key==='Escape'){close();returnFocus?.focus({preventScroll:true});close();}});
 $('distribution-popup-close').addEventListener('click',()=>{close();returnFocus?.focus({preventScroll:true});close();});
 $('distribution-popup-list').addEventListener('click',e=>{
  const button=e.target.closest('button[data-rank]');if(!button)return;
  window.dispatchEvent(new Event('tour-show-merchant'));
  selectPage(Math.floor((Number(button.dataset.rank)-1)/pageSize));close();
  const row=document.querySelector(`[data-merchant-abn="${button.dataset.abn}"]`);
  if(row){row.classList.add('merchant-highlight');row.scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'center'});row.focus({preventScroll:true});}
 });
 document.addEventListener('pointerdown',e=>{if(!popup.hidden&&!popup.contains(e.target)&&!e.target.closest('.histogram-bar'))close();});
 window.addEventListener('resize',close);window.addEventListener('scroll',close,{passive:true});
 section.addEventListener('toggle',()=>section.open?render():close());
 window.closeDistributionPopup=close;
 window.renderDistributions=render;
 // Pure binning hook is used by local integrity checks as well as the renderer.
 window.distributionModel={specs,build};
 render();
})();
