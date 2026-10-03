'use strict';
(() => {
 const dialog=$('metric-dialog');
 let trigger=null;
 const explanation={
  revenue:()=>({title:'Monthly expected revenue',paragraphs:[
   'Monthly expected revenue here means the BNPL firm’s historical mean monthly gross fees, summed across the selected merchants. Monthly transaction sales × take rate / 100 gives monthly gross fees; the take rate is already applied and must not be applied again.',
   'This differs from the curated expected_revenue field, which is fraud-adjusted over the stored history. The dashboard consistently uses the final ranking’s monthly gross-fee estimate. The estimate uses 19 complete months, March 2021–September 2022, in AUD. It is gross fee income, not net profit: fraud losses, unpaid principal, funding and operating costs, and settlement timing are excluded.',
   `For the current selection: ${money(portfolio().metrics.expected_monthly_fees)} per month. Alpha (α) sets the minimum proportion of baseline expected fees that must be retained; it is not a risk score.`]}),
  risk:()=>({title:'Combined risk score',paragraphs:[
   'The model minimises this score while selecting exactly 100 merchants and meeting the return-retention target (α). Lower scores mean less estimated risk under the model.',
   `Score = 100 × [${pct(data.variability_weight)} × (CV / baseline CV) + ${pct(data.concentration_weight)} × (HHI / baseline HHI)]. The baseline is the 100 eligible merchants with the highest expected fees within the selected scope, and its score is 100. Scores from different scopes use different baselines.`,
   'CV measures weekly fee volatility; HHI measures shared-customer concentration. The weights express a modelling preference. This index is not a probability, and lower estimated risk does not guarantee future protection.',
   `Current score: ${portfolio().metrics.risk_score.toFixed(3)}. ${portfolio().validation.certified?'Certified to the requested solver tolerance':'Best found; not certified to the requested solver tolerance'}. Gap upper bound: ${pct(portfolio().validation.gap,3)}; score lower bound: ${portfolio().validation.lower_bound.toFixed(6)}.`,
   'Certification means the 0.1% relative gap or 0.0001 score-point tolerance was met, not an exact-optimum or future-performance guarantee. The source status records how this particular search ended.',
   `Source status: ${portfolio().validation.status}. Portfolio: ${portfolio().id}.`]}),
  volatility:()=>({title:'Weekly fee volatility (CV)',paragraphs:[
   'Coefficient of variation is portfolio weekly fee standard deviation divided by historical mean weekly fees. It measures variability relative to the size of the portfolio, so it is displayed as a percentage.',
   'Portfolio standard deviation uses the shrunk merchant covariance matrix, accounting for how merchants move together. It is not the sum or simple average of individual volatilities.',
   `The risk history covers 26 complete weeks, ${date(data.window_start)}–${date(data.window_end)}, with a selection snapshot of ${date(data.as_of)}. Current CV: ${pct(portfolio().metrics.cv,3)}.`,
   'Lower CV indicates more stable historical weekly fees. CV has no inherent upper bound of 100%, and it is not a probability of loss.' ]}),
  concentration:()=>({title:'Customer concentration (HHI)',paragraphs:[
   'The Herfindahl–Hirschman Index is the sum of squared customer shares of expected portfolio fee exposure. Each merchant’s expected fees are allocated to its customers according to their historical fee shares.',
   'Exposure to the same customer is combined across merchants before calculating the index. Lower HHI means expected fees are spread more broadly across customers; higher HHI means greater concentration.',
   `HHI ranges from 0 to 1. The current value is ${portfolio().metrics.hhi.toFixed(8)}. Eight decimal places preserve the small differences between these large customer bases.`,
   'This measures concentration, not fraud probability. Shared customer identifiers must represent the same person across merchants.' ]})
 };
 for(const key of Object.keys(explanation)){
  $('help-'+key).addEventListener('click',()=>{
   trigger=$('help-'+key);
   const content=explanation[key]();
   $('metric-help-title').textContent=content.title;
   $('metric-help-body').innerHTML=content.paragraphs.map(p=>`<p>${esc(p)}</p>`).join('');
   dialog.showModal();$('metric-help-close').focus();
  });
 }
 $('metric-help-close').addEventListener('click',()=>dialog.close());
 dialog.addEventListener('click',e=>{if(e.target!==dialog)return;const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();});
 dialog.addEventListener('close',()=>trigger?.focus({preventScroll:true}));
})();
