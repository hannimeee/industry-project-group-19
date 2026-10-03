 'use strict';
const data=window.PORTFOLIO_DATA,$=id=>document.getElementById(id),pageSize=10;
let selectedId=(data.portfolios.find(p=>p.scope==='global'&&Math.abs(p.retention-.99)<1e-10)||data.portfolios[0]).id,page=0;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money=n=>new Intl.NumberFormat('en-AU',{style:'currency',currency:data.currency,minimumFractionDigits:2,maximumFractionDigits:2}).format(n);
const compact=n=>new Intl.NumberFormat('en-AU',{style:'currency',currency:data.currency,notation:'compact',maximumFractionDigits:3}).format(n);
const pct=(n,digits=1)=>new Intl.NumberFormat('en-AU',{style:'percent',maximumFractionDigits:digits}).format(n);
const date=s=>new Date(s+'T00:00:00').toLocaleDateString('en-AU',{day:'numeric',month:'short',year:'numeric'});
const portfolio=()=>data.portfolios.find(p=>p.id===selectedId);
$('source').textContent='Final results: '+new Date(data.results_date+'T00:00:00').toLocaleDateString('en-GB',{day:'2-digit',month:'long',year:'numeric'});
$('source').title='Results label date; underlying selection snapshot: '+date(data.as_of);
$('footnote').textContent='Snapshot '+date(data.as_of)+' · Final handoff';
$('revenue-note').textContent=data.currency+' per month · historical mean';
$('scope').innerHTML=data.scopes.map(s=>`<option value="${esc(s.id)}">${esc(s.name)}</option>`).join('');
$('scope').addEventListener('change',e=>{const current=portfolio();const next=data.portfolios.find(p=>p.scope===e.target.value&&p.retention===current.retention);if(next)selectPortfolio(next.id);});
function selectPortfolio(id){if(!data.portfolios.some(p=>p.id===id))throw Error('Portfolio unavailable');selectedId=id;page=Math.min(page,Math.max(0,Math.ceil(portfolio().merchants.length/pageSize)-1));render();}
$('retention').addEventListener('change',e=>selectPortfolio(e.target.value));
function selectPage(value){const count=Math.ceil(portfolio().merchants.length/pageSize);if(!Number.isInteger(value)||value<0||value>=count)throw Error('Page unavailable');page=value;renderTable();}
$('page-buttons').addEventListener('click',e=>{const button=e.target.closest('button[data-page]');if(!button)return;selectPage(Number(button.dataset.page));$('page-buttons').querySelector(`[data-page="${page}"]`).focus({preventScroll:true});});
function render(){
 const p=portfolio(),m=p.metrics,v=p.validation;
 $('scope').value=p.scope;
 $('retention').innerHTML=data.portfolios.filter(q=>q.scope===p.scope).map(q=>`<option value="${esc(q.id)}">${pct(q.retention)}${q.retention===1?' · baseline':''}</option>`).join('');
 $('retention').value=p.id;
 $('table-count').textContent=p.merchants.length;
 $('revenue').textContent=compact(m.expected_monthly_fees);$('revenue').title=money(m.expected_monthly_fees);
 $('risk-score').textContent=m.risk_score.toFixed(3);
 $('variability').textContent=pct(m.cv,3);$('variability').title=String(m.cv);
 $('concentration').textContent=m.hhi.toFixed(8);$('concentration').title=String(m.hhi);
 $('selection-title').textContent=`α = ${pct(p.retention)} · return-retention target`;
 $('selection-description').textContent=`This portfolio retains ${pct(m.achieved_retention,3)} of its scope’s baseline expected fees. Its combined risk score is ${m.risk_score.toFixed(3)}, against a baseline of 100.`;
 $('solution-status').textContent=`${v.certified?'Certified to solver tolerance':'Best found · not certified to tolerance'} · gap upper bound ${pct(v.gap,3)}`;
 $('solution-status').dataset.certified=String(v.certified);
 $('status').textContent=`Showing ${p.merchants.length} selected merchants at a ${pct(p.retention)} minimum fee-retention floor.`;
 renderTable();renderChart();renderSegmentMix();window.renderDistributions?.();
}
function renderTable(){window.closeMerchantPreview?.();const list=portfolio().merchants,start=page*pageSize;
 $('merchants').innerHTML=list.slice(start,start+pageSize).map(m=>`<tr data-merchant-abn="${esc(m.abn)}" tabindex="0" aria-label="View stats for ${esc(m.name)}"><td>${String(m.rank).padStart(2,'0')}</td><td class="merchant-name"><span>${esc(m.name)}</span><small>ABN ${esc(m.abn)} · ${esc(m.category)}</small></td><td class="num">${money(m.expected_monthly_fees)}</td><td class="num">${m.avg_consumer_fraud_prob==null?'—':m.avg_consumer_fraud_prob.toFixed(2)+'%'}</td></tr>`).join('');
 $('page-info').textContent=`Showing ${start+1}–${Math.min(start+pageSize,list.length)} of ${list.length} merchants`;
 $('page-buttons').innerHTML=Array.from({length:Math.ceil(list.length/pageSize)},(_,i)=>`<button type="button" data-page="${i}" aria-label="Page ${i+1}, merchants ${i*pageSize+1}–${Math.min((i+1)*pageSize,list.length)}" ${i===page?'aria-current="page"':''}>${i+1}</button>`).join('');
}
function renderChart() {
 const ps=data.portfolios.filter(p=>p.scope===portfolio().scope).sort((a,b)=>a.metrics.risk_score-b.metrics.risk_score);
 const xs=ps.map(p=>p.metrics.risk_score),ys=ps.map(p=>p.metrics.expected_monthly_fees);
 const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 const dx=xmax-xmin||Math.max(Math.abs(xmax)*.1,1),dy=ymax-ymin||Math.max(Math.abs(ymax)*.1,1);
 const x=v=>135+(v-xmin)/dx*395,y=v=>230-(v-ymin)/dy*135;
 let svg=`<svg viewBox="0 0 620 345" role="img" aria-label="Monthly expected revenue versus the combined CV and customer HHI risk index, using focused axis ranges. Lines connect supplied portfolios and do not estimate an efficient frontier.">
 <defs><marker id="axis-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 Z" fill="#271866"/></marker></defs>
 <rect x="12" y="12" width="83" height="29" rx="2" fill="#e7e3ff"/><text class="axis-title" x="53.5" y="32" text-anchor="middle">Return</text>
 <text x="108" y="32">Monthly expected revenue</text>
 <path d="M 94 269 V 57 M 94 269 H 578" fill="none" stroke="#271866" stroke-width="2.2" marker-end="url(#axis-arrow)"/>
 <path d="M 94 80 V 57" fill="none" stroke="#271866" stroke-width="2.2" marker-end="url(#axis-arrow)"/>`;
 for(let i=0;i<3;i++) {
  const v=ymin+dy*i/2,yy=y(v);
  svg+=`<line x1="94" y1="${yy}" x2="559" y2="${yy}" stroke="#eeebf6"/><line x1="89" y1="${yy}" x2="94" y2="${yy}" stroke="#8b80ab"/><text x="81" y="${yy+4}" text-anchor="end">${esc(compact(v))}</text>`;
 }
 for(let i=0;i<3;i++) {
  const v=xmin+dx*i/2,xx=x(v);
  svg+=`<line x1="${xx}" y1="269" x2="${xx}" y2="274" stroke="#8b80ab"/><text x="${xx}" y="291" text-anchor="middle">${v.toFixed(1)}</text>`;
 }
 if(ps.length>1)svg+=`<polyline points="${ps.map(p=>`${x(p.metrics.risk_score)},${y(p.metrics.expected_monthly_fees)}`).join(' ')}" fill="none" stroke="#271866" stroke-width="2.4" stroke-linejoin="round"/>`;
 ps.forEach((p,i)=>{
  const xx=x(p.metrics.risk_score),yy=y(p.metrics.expected_monthly_fees),sel=p.id===selectedId;
  svg+=`${sel?`<circle cx="${xx}" cy="${yy}" r="13" fill="#e7e3ff"/><circle cx="${xx}" cy="${yy}" r="8.5" fill="white" stroke="#271866" stroke-width="2"/>`:''}<circle class="point" data-portfolio="${esc(p.id)}" cx="${xx}" cy="${yy}" r="5.5" fill="#00bfd4"><title>Return-retention target α = ${pct(p.retention)} · revenue ${esc(money(p.metrics.expected_monthly_fees))} · risk score ${p.metrics.risk_score.toFixed(3)}${sel?' · selected':''}</title></circle><text class="${sel?'selected-point-label':''}" x="${xx}" y="${yy-20-(i%2?12:0)}" text-anchor="middle">α = ${pct(p.retention)}</text>`;
 });
 svg+='<rect x="110" y="311" width="83" height="29" rx="2" fill="#e7e3ff"/><text class="axis-title" x="151.5" y="331" text-anchor="middle">Risk</text><text x="207" y="331">Combined risk score · baseline = 100</text>';
 $('chart').innerHTML=svg+'</svg>';
}

render();
if(document.modelContext?.registerTool){try{Promise.resolve(document.modelContext.registerTool({name:'select_portfolio',description:'Display one of the supplied final global portfolios by portfolio ID.',inputSchema:{type:'object',properties:{portfolio_id:{type:'string'}},required:['portfolio_id'],additionalProperties:false},annotations:{readOnlyHint:false},execute(input){selectPortfolio(input.portfolio_id);return {portfolio_id:selectedId,metrics:portfolio().metrics,validation:portfolio().validation};}})).catch(()=>{});}catch{}}

function renderSegmentMix(){
 const p=portfolio(),panel=$('segment-mix');panel.hidden=p.scope!=='global';
 if(panel.hidden)return;
 const segments=data.scopes.filter(s=>s.id!=='global');
 $('segment-mix-rows').innerHTML=segments.map((s,i)=>{
  const count=p.merchants.filter(m=>m.segment===s.id).length,share=count/p.merchants.length;
  return `<div class="segment-mix-row"><span>${esc(s.name)}</span><strong>${pct(share)}</strong><div class="segment-mix-track"><i style="width:${share*100}%;--segment-color:${['#271866','#6554ad','#9884ce','#00a5bd','#6fc9d0'][i]}"></i></div></div>`;
 }).join('');
}
