'use strict';
(() => {
 const panel=$('merchant-preview'),table=$('merchants');let timer,anchor;
 function close(){clearTimeout(timer);panel.hidden=true;}
 function show(row){
  const m=portfolio().merchants.find(m=>m.abn===row.dataset.merchantAbn);if(!m)return;
  clearTimeout(timer);anchor=row;
  $('merchant-preview-name').textContent=m.name;
  $('merchant-preview-abn').textContent=`Rank ${m.rank} · ABN ${m.abn}`;
  $('merchant-preview-category').textContent=m.category;
  const stats=[['⌂','Consumer average district income',m.avg_consumer_district_income==null?'—':money(m.avg_consumer_district_income)+' / week'],['◇','Average consumer fraud score',m.avg_consumer_fraud_prob==null?'—':m.avg_consumer_fraud_prob.toFixed(2)+'%'],['≡','Total transactions',m.total_transactions==null?'—':m.total_transactions.toLocaleString('en-AU')],['↔','Average transaction value',m.avg_value_per_transaction==null?'—':money(m.avg_value_per_transaction)],['↗','Expected monthly return',money(m.expected_monthly_fees)]];
  $('merchant-preview-stats').innerHTML=stats.map(([icon,label,value])=>`<dt><span aria-hidden="true">${icon}</span>${esc(label)}</dt><dd>${esc(value)}</dd>`).join('');
  panel.hidden=false;const r=row.getBoundingClientRect(),w=panel.offsetWidth,h=panel.offsetHeight;
  const left=r.left>w+24?r.left-w-12:r.right+12;
  panel.style.left=Math.max(12,Math.min(left,window.innerWidth-w-12))+'px';panel.style.top=Math.max(12,Math.min(r.top,window.innerHeight-h-12))+'px';
 }
 function leave(){clearTimeout(timer);timer=setTimeout(()=>{if(!panel.contains(document.activeElement))close();},200);}
 table.addEventListener('pointerover',e=>{const row=e.target.closest('tr[data-merchant-abn]');if(row&&(!row.contains(e.relatedTarget)))show(row);});
 table.addEventListener('pointerout',e=>{const row=e.target.closest('tr[data-merchant-abn]');if(row&&!row.contains(e.relatedTarget))leave();});
 table.addEventListener('focusin',e=>{const row=e.target.closest('tr[data-merchant-abn]');if(row)show(row);});
 table.addEventListener('focusout',leave);
 table.addEventListener('click',e=>{const row=e.target.closest('tr[data-merchant-abn]');if(row){show(row);$('merchant-preview-close').focus({preventScroll:true});}});
 panel.addEventListener('pointerenter',()=>clearTimeout(timer));panel.addEventListener('pointerleave',leave);
 $('merchant-preview-close').addEventListener('click',()=>{anchor?.focus({preventScroll:true});close();});
 document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!panel.hidden){anchor?.focus({preventScroll:true});close();}});
 document.addEventListener('pointerdown',e=>{if(!panel.hidden&&!panel.contains(e.target)&&!table.contains(e.target))close();});
 window.addEventListener('resize',close);window.addEventListener('scroll',close,{passive:true});window.closeMerchantPreview=close;
})();
