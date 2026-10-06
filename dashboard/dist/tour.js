'use strict';
(() => {
  const key = 'merchant-dashboard-real.guide.v1';
  const byId = id => document.getElementById(id);
  const overlay = byId('tour-overlay'), card = byId('tour-card'), spotlight = byId('tour-spotlight');
  const help = byId('tour-help'), next = byId('tour-next'), back = byId('tour-back'), skip = byId('tour-skip');
  const covariance = byId('covariance');
  const distributions = byId('distributions');
  let distributionsWasOpen = false;
  const blockers=Array.from({length:4},()=>{const el=document.createElement('div');el.className='tour-blocker';overlay.prepend(el);return el;});
  const steps = [
    {target:'.scope',title:'Choose global or a sector',body:'Explore all eligible merchants or one of five sectors. Each scope has five return-retention settings and 100 merchants per selection. Switching scope preserves α and your merchant page. Risk scores are relative to each scope’s own baseline.'},
    {target:'.alpha-control',title:'Set return retention (α)',body:'Return retention (α) expresses risk tolerance by setting the minimum share of baseline expected return to keep. A lower target gives the model more room to reduce risk. At 99%, the model must retain at least 99% of baseline expected monthly return while reducing combined risk. Actual retention may be slightly higher.'},
    {target:'.metrics',title:'Inspect return and risk metrics',body:'Expected monthly return is adjusted for take rate and consumer fraud probability. Combined risk uses baseline-normalised weekly adjusted-return CV and shared-customer HHI with equal weights. The index has baseline 100; it is not a loss probability. Use each card’s ? button to read its definition. Small HHI values are displayed with eight decimal places.'},
    {target:'.tradeoff',title:'Compare the supplied selections',body:'The chart plots expected monthly return against the combined risk index. Dot labels show α, while the horizontal ticks show risk scores, not percentages. The purple ring marks the selected target. Some selections are best found rather than certified; their gap upper bounds appear below the chart.'},
    {target:'.merchant-section',title:'Review the selected merchants',body:'Use page buttons 1–10 to see all 100 merchants, ten per page. Your page stays selected when you change α, so you can compare the same ranks. They are ordered by expected monthly return within this portfolio. ABNs and categories identify the merchants. The consumer fraud score is shown for reference. Hover or focus a merchant to inspect its full category and other statistics. No individual merchant risk score is supplied.'},
    {target:'#distributions',title:'Inspect merchant distributions',body:'Expand this section to compare income, fraud scores, transaction counts, transaction values. Hover or select a bar to inspect its merchants, then jump directly to a ranking row. Axis labels identify any log scale.'},
    {target:'#covariance',title:'Explore category relationships',body:'Switch between mean monthly-revenue correlation and consumer overlap (Jaccard similarity). These describe the full merchant dataset and do not change with α or scope. Diagonal cells compare different merchants within one sector; they are not variances.'},
    {target:'#tour-help',title:'Return to the guide anytime',body:'Use this question mark to replay the guide. This separate dashboard contains the final model results; the original demonstration dashboard remains unchanged.'}
  ];
  let index = 0, active = false, frame = 0, restoreFocus, savedScroll, wasOpen;
  const clamp = (n, min, max) => Math.max(min, Math.min(n, Math.max(min, max)));
  function position() {
    if (!active) return;
    const target = document.querySelector(steps[index].target);
    const r = target.getBoundingClientRect(), width = window.innerWidth, height = window.innerHeight;
    const left = clamp(r.left - 5, 4, width - 4), top = clamp(r.top - 5, 4, height - 4);
    const right = clamp(r.right + 5, left, width - 4), bottom = clamp(r.bottom + 5, top, height - 4);
    Object.assign(spotlight.style, {left:`${left}px`,top:`${top}px`,width:`${right-left}px`,height:`${bottom-top}px`});
    [[0,0,width,top],[0,bottom,width,height-bottom],[0,top,left,bottom-top],[right,top,width-right,bottom-top]].forEach(([x,y,w,h],i)=>Object.assign(blockers[i].style,{left:x+'px',top:y+'px',width:w+'px',height:h+'px'}));
    const w = card.offsetWidth, h = card.offsetHeight, gap = 14;
    let x, y;
    if (width - right >= w + gap + 12) {x = right + gap; y = top;}
    else if (left >= w + gap + 12) {x = left - w - gap; y = top;}
    else if (height - bottom >= h + gap + 12) {x = left; y = bottom + gap;}
    else if (top >= h + gap + 12) {x = left; y = top - h - gap;}
    else {x = width - w - 12; y = height - h - 12;}
    Object.assign(card.style, {left:`${clamp(x,12,width-w-12)}px`,top:`${clamp(y,12,height-h-12)}px`});
  }
  function schedulePosition() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(position);
  }
  function showStep() {
    distributions.open = steps[index].target === '#distributions' ? true : distributionsWasOpen;
    covariance.open = steps[index].target === '#covariance' ? true : wasOpen;
    window.closeMerchantPreview?.();window.closeDistributionPopup?.();
    const step = steps[index];
    byId('tour-progress').textContent = `STEP ${index + 1} OF ${steps.length}`;
    byId('tour-title').textContent = step.title;
    byId('tour-body').textContent = step.body;
    byId('tour-dots').innerHTML = steps.map((_,i)=>`<i class="${i===index?'active':''}"></i>`).join('');
    back.disabled = index === 0;
    next.textContent = index === steps.length - 1 ? 'Finish' : 'Next →';
    document.querySelector(step.target).scrollIntoView({behavior:'instant',block:'center',inline:'nearest'});
    position();
    next.focus({preventScroll:true});
  }
  function remember() {try {localStorage.setItem(key,'seen');} catch { /* Storage may be disabled for local files. */ }}
  function start() {
    if (active) return;
    restoreFocus = document.activeElement;
    savedScroll = {left:window.scrollX,top:window.scrollY};
    wasOpen = covariance.open;
    distributionsWasOpen = distributions.open;

    index=0;active=true;overlay.hidden=false;
    document.body.classList.add('tour-active');
    showStep();
  }
  function close() {
    if (!active) return;
    active=false;cancelAnimationFrame(frame);remember();window.closeMerchantPreview?.();window.closeDistributionPopup?.();
    covariance.open=wasOpen;distributions.open=distributionsWasOpen;
    overlay.hidden=true;document.body.classList.remove('tour-active');

    window.scrollTo({...savedScroll,behavior:'instant'});
    (restoreFocus && restoreFocus !== document.body ? restoreFocus : help).focus({preventScroll:true});
  }
  help.addEventListener('click',start);
  skip.addEventListener('click',close);
  next.addEventListener('click',()=>{if(index===steps.length-1)close();else{index++;showStep();}});
  back.addEventListener('click',()=>{if(index>0){index--;showStep();}});
  document.addEventListener('keydown',e=>{
    if(!active || document.querySelector('dialog[open]'))return;
    const popups=[byId('merchant-preview'),byId('distribution-popup')].filter(el=>el&&!el.hidden);
    if(e.key==='Escape'){if(popups.length)return;e.preventDefault();close();return;}
    if(e.key==='Tab'){
      const roots=[document.querySelector(steps[index].target),...popups,card];
      const controls=[...new Set(roots.flatMap(root=>[...(root.matches('button, [tabindex]')?[root]:[]),...root.querySelectorAll('button, select, summary, a[href], [tabindex]')]))].filter(el=>!el.disabled&&el.tabIndex>=0&&el.getClientRects().length);
      const current=controls.indexOf(document.activeElement);
      e.preventDefault();controls[(current+(e.shiftKey?-1:1)+controls.length)%controls.length]?.focus();
    }
  });
  window.addEventListener('tour-show-merchant',()=>{if(active){index=4;showStep();}});
  window.addEventListener('resize',schedulePosition);
  window.addEventListener('scroll',schedulePosition,{passive:true});
  let seen=false;
  try {seen=localStorage.getItem(key)==='seen';} catch { /* Show on entry if preferences cannot be stored. */ }
  if(!seen) requestAnimationFrame(()=>{if(!active)start();});
})();
