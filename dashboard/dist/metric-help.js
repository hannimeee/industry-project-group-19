'use strict';
(() => {
 const dialog=$('metric-dialog');
 let trigger=null;
 const explanation={
  revenue:()=>({title:'Expected monthly return',paragraphs:[
   'Each transaction is adjusted by the merchant take rate and consumer fraud probability: transaction value × take rate / 100 × (1 − consumer fraud probability / 100). Monthly amounts are averaged over 19 complete months, March 2021–September 2022, then summed across selected merchants.',
   'Return, weekly covariance and consumer exposures all use the adjusted basis. This is a retrospective modelled return proxy, not net profit: it does not separately deduct unrecovered principal, funding or operating costs. Missing fraud scores were reconstructed using the existing consumer model; most transactions are outside labelled predictor support, so calibration remains uncertain.',
   `Current expected monthly return: ${money(portfolio().metrics.expected_monthly_fees)}. Return retention (α) is relative to the maximum adjusted-return portfolio in this scope.`]}),
  risk:()=>({title:'Combined risk score',paragraphs:[
   'The model minimises this score while selecting exactly 100 merchants and meeting the return-retention target (α). Lower scores mean less estimated risk under the model.',
   `Score = 100 × [${pct(data.variability_weight)} × (CV / baseline CV) + ${pct(data.concentration_weight)} × (HHI / baseline HHI)]. The baseline is the 100 eligible merchants with the highest expected return within the selected scope, and its score is 100. Scores from different scopes use different baselines.`,
   'Strategy labels describe the selected alpha: 100% High Growth; 99.5% Growth; 99% Balanced; 98% and 95% Conservative Balanced. They are not risk-score bands.',
   'CV measures weekly revenue volatility; HHI measures shared-customer concentration. The weights express a modelling preference. This index is not a probability, and lower estimated risk does not guarantee future protection.',
   `Current score: ${portfolio().metrics.risk_score.toFixed(3)}. ${portfolio().validation.certified?'Certified to the requested solver tolerance':'Best found; not certified to the requested solver tolerance'}. Gap upper bound: ${pct(portfolio().validation.gap,3)}; score lower bound: ${portfolio().validation.lower_bound.toFixed(6)}.`,
   'Certification means the 0.1% relative gap or 0.0001 score-point tolerance was met, not an exact-optimum or future-performance guarantee. The source status records how this particular search ended.',
   `Source status: ${portfolio().validation.status}. Portfolio: ${portfolio().id}.`]}),
  volatility:()=>({title:'Weekly revenue volatility (CV)',paragraphs:[
   'Coefficient of variation is portfolio weekly adjusted return standard deviation divided by historical mean weekly adjusted returns. It measures variability relative to the size of the portfolio, so it is displayed as a percentage.',
   'Portfolio standard deviation uses the shrunk merchant covariance matrix, accounting for how merchants move together. It is not the sum or simple average of individual volatilities.',
   `The risk history covers 26 complete weeks, ${date(data.window_start)}–${date(data.window_end)}, with a selection snapshot of ${date(data.as_of)}. Current CV: ${pct(portfolio().metrics.cv,3)}.`,
   'Lower CV indicates more stable historical weekly adjusted returns. CV has no inherent upper bound of 100%, and it is not a probability of loss.' ]}),
  concentration:()=>({title:'Customer concentration (HHI)',paragraphs:[
   'The Herfindahl–Hirschman Index is the sum of squared customer shares of expected portfolio return exposure. Each merchant’s expected return are allocated to its customers according to their historical adjusted-return shares.',
   'Exposure to the same customer is combined across merchants before calculating the index. Lower HHI means expected return is spread more broadly across customers; higher HHI means greater concentration.',
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
