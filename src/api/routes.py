"""FastAPI application — REST endpoints + browser UI."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from config.settings import get_settings
from src.agent.react_agent import chat, create_agent, setup
from src.api.cache import get_cache
from src.utils.logger import get_logger
from src.utils.schemas import AnalysisRequest, AnalysisResponse, AnalysisStatus

logger = get_logger(__name__)
_GRAPH = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _GRAPH
    try:
        setup()
    except EnvironmentError as e:
        logger.warning("startup.config_warning", detail=str(e))
    _GRAPH = create_agent()
    logger.info("startup.complete", model=get_settings().groq_model)
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Market Analysis Agent",
    description="AI-powered e-commerce market intelligence — LangGraph + Groq + DuckDuckGo",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Ops ───────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "model": get_settings().groq_model}


# ── Analysis ──────────────────────────────────────────────────────────────────

import re as _re


def _parse_rate_limit_error(exc: Exception) -> str | None:
    """Extract a human-readable message from a Groq 429 error, or return None."""
    msg = str(exc)
    if "429" not in msg and "rate_limit_exceeded" not in msg:
        return None
    match = _re.search(r"Please try again in ([\d\.]+m[\d\.]+s|[\d\.]+s|[\d\.]+m)", msg)
    retry = match.group(1) if match else None
    base = "Groq rate limit reached — daily token quota exceeded."
    return (base + " Resets in " + retry + ".") if retry else (base + " Please try again shortly.")


@app.post("/api/v1/analyze", response_model=AnalysisResponse, response_model_exclude_none=False, tags=["analysis"])
async def analyze(request: AnalysisRequest):
    """Run the full market analysis pipeline for a product query."""
    cache   = get_cache()
    cache_key = f"report:{request.product_query.lower().strip()}:sample={request.use_sample_data}:lang={request.language}"
    cached  = cache.get(cache_key)
    if cached:
        logger.info("api.cache_hit", query=request.product_query)
        return cached

    request_id = uuid.uuid4()
    t0 = time.perf_counter()

    try:
        report = await chat(request, graph=_GRAPH)
        duration = round(time.perf_counter() - t0, 2)
        response = AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.COMPLETED,
            product_query=request.product_query,
            report=report,
            duration_seconds=duration,
        )
        cache.set(cache_key, response)
        logger.info("api.done", query=request.product_query, duration=duration)
        return response

    except Exception as exc:
        duration = round(time.perf_counter() - t0, 2)
        error_msg = _parse_rate_limit_error(exc) or str(exc)
        logger.error("api.failed", query=request.product_query, error=error_msg)
        return AnalysisResponse(
            request_id=request_id,
            status=AnalysisStatus.FAILED,
            product_query=request.product_query,
            error=error_msg,
            duration_seconds=duration,
        )


# ── Browser UI ────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Market Analysis Agent</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:system-ui,-apple-system,sans-serif;background:#f4f5f7;color:#1a1d23;min-height:100vh}
    header{background:#fff;border-bottom:1px solid #e0e3e8;padding:1rem 2rem;display:flex;align-items:center;gap:1rem;box-shadow:0 1px 3px rgba(0,0,0,.06)}
    header h1{font-size:1.25rem;font-weight:700;color:#1a1d23;letter-spacing:-.01em}
    header span{color:#6b7280;font-size:.85rem}
    main{max-width:860px;margin:2rem auto;padding:0 1.25rem}
    .card{background:#fff;border:1px solid #e0e3e8;border-radius:10px;padding:1.5rem;margin-bottom:1.25rem;box-shadow:0 1px 3px rgba(0,0,0,.04)}
    .card-label{font-size:.75rem;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.9rem}
    .row{display:flex;gap:.75rem;align-items:center}
    input[type=text]{flex:1;padding:.7rem 1rem;border-radius:7px;border:1.5px solid #d1d5db;background:#f9fafb;color:#1a1d23;font-size:.95rem;outline:none;transition:border-color .15s}
    input[type=text]:focus{border-color:#374151;background:#fff}
    label.toggle{display:flex;align-items:center;gap:.4rem;font-size:.85rem;color:#6b7280;white-space:nowrap;cursor:pointer;user-select:none}
    input[type=checkbox]{accent-color:#374151;width:15px;height:15px}
    button{padding:.7rem 1.4rem;border-radius:7px;border:none;background:#1a1d23;color:#fff;font-size:.95rem;cursor:pointer;font-weight:600;transition:background .15s;white-space:nowrap}
    button:hover{background:#374151}
    button:disabled{background:#9ca3af;cursor:not-allowed}
    .hint{color:#9ca3af;font-size:.78rem;margin-top:.6rem}
    .badge{display:inline-block;padding:.2rem .55rem;border-radius:5px;font-size:.72rem;font-weight:600;letter-spacing:.02em}
    .completed{background:#dcfce7;color:#166534}
    .failed{background:#fee2e2;color:#991b1b}
    .sample-badge{background:#f0f9ff;color:#0369a1;border:1px solid #bae6fd;font-size:.7rem;padding:.15rem .5rem;border-radius:4px;margin-left:.4rem;font-weight:500}
    .section{margin-bottom:1.4rem}
    .section h3{font-size:.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem}
    .section p{line-height:1.75;color:#374151;font-size:.93rem}
    ul.recs{padding-left:1.1rem}
    ul.recs li{line-height:1.75;color:#374151;font-size:.93rem;margin-bottom:.2rem}
    .meta{color:#9ca3af;font-size:.78rem;margin-top:.35rem}
    hr{border:none;border-top:1px solid #e0e3e8;margin:1.1rem 0}
    .err{background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:1rem;color:#dc2626;margin-bottom:1rem;display:none;font-size:.9rem}
    #result{display:none}
    .spin{display:inline-block;width:14px;height:14px;border:2.5px solid #d1d5db;border-top-color:#374151;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:.4rem}
    @keyframes spin{to{transform:rotate(360deg)}}
    .report-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.5rem}
    .report-title{font-size:1.1rem;font-weight:700;color:#1a1d23;letter-spacing:-.01em}
    /* Sentiment chart */
    #chart-wrap{display:none;margin-bottom:1.4rem;flex-direction:column;align-items:center;gap:.75rem;background:#f9fafb;border:1px solid #e0e3e8;border-radius:8px;padding:1.5rem;text-align:center}
    #chart-wrap h4{font-size:.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.08em}
    #chartSent{max-height:200px;max-width:200px}
    .sent-score .score-num{font-size:2.25rem;font-weight:700;color:#1a1d23;line-height:1}
    .sent-score .score-label{font-size:.75rem;color:#6b7280;margin-top:.3rem}
  </style>
</head>
<body>
<header>
  <h1>&#x1F4CA; Market Analysis Agent</h1>
  <span>LangGraph &middot; Groq &middot; DuckDuckGo</span>
</header>
<main>
  <div class="card">
    <div class="card-label">New Analysis</div>
    <div class="row">
      <input id="q" type="text" placeholder="e.g. iPhone 16 Pro, Sony WH-1000XM6, Nike Air Max 2025"/>
      <label class="toggle"><input type="checkbox" id="sample"/> Sample data</label>
      <label class="toggle"><input type="checkbox" id="french"/> Fran&ccedil;ais</label>
      <button id="btn" onclick="run()">Analyse</button>
    </div>
    <p class="hint">Toggle &ldquo;Sample data&rdquo; to use fixture files instead of live web scraping.</p>
  </div>

  <div class="err" id="err"></div>

  <div id="result" class="card">
    <div class="report-header">
      <span class="report-title" id="rtitle"></span>
      <div style="display:flex;align-items:center;gap:.4rem;flex-shrink:0">
        <span class="badge" id="badge"></span>
        <span id="sample-ind"></span>
      </div>
    </div>
    <p class="meta" id="rmeta"></p>
    <hr/>
    <div id="chart-wrap">
      <h4 id="h-chart-sent">Customer Sentiment</h4>
      <canvas id="chartSent"></canvas>
      <div class="sent-score">
        <div class="score-num" id="sent-score-num"></div>
        <div class="score-label" id="sent-score-label">sentiment score</div>
      </div>
    </div>
    <div class="section"><h3 id="h-exec">Executive Summary</h3><p id="exec"></p></div>
    <div class="section"><h3 id="h-price">Pricing Analysis</h3><p id="price"></p></div>
    <div class="section"><h3 id="h-sent">Customer Sentiment</h3><p id="sent"></p></div>
    <div class="section"><h3 id="h-trend">Market Trends</h3><p id="trend"></p></div>
    <div class="section">
      <h3 id="h-recs">Strategic Recommendations</h3>
      <ul class="recs" id="recs"></ul>
    </div>
  </div>
</main>
<script>
let _sentChart = null;

async function run(){
  const q=document.getElementById('q').value.trim();
  if(!q)return;
  const sample=document.getElementById('sample').checked;
  const french=document.getElementById('french').checked;
  const btn=document.getElementById('btn');
  const err=document.getElementById('err');
  const res=document.getElementById('result');
  btn.disabled=true;
  btn.innerHTML='<span class="spin"></span>Analysing\u2026';
  err.style.display='none';
  res.style.display='none';
  try{
    const r=await fetch('/api/v1/analyze',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({product_query:q,use_sample_data:sample,language:french?'fr':'en'})
    });
    const d=await r.json();
    if(d.status==='failed'||!d.report){
      err.textContent='\u26a0 '+(d.error||'Analysis failed.');
      err.style.display='block';
    } else {
      const rp=d.report;
      const fr=document.getElementById('french').checked;
      document.getElementById('rtitle').textContent=rp.product_query;
      const b=document.getElementById('badge');
      b.textContent=d.status; b.className='badge '+d.status;
      document.getElementById('sample-ind').innerHTML=
        rp.raw_data&&rp.raw_data.source==='sample'?'<span class="sample-badge">sample data</span>':'';
      document.getElementById('rmeta').textContent='ID '+d.request_id+' \u00b7 '+d.duration_seconds+'s';
      document.getElementById('exec').textContent=rp.executive_summary||'';
      document.getElementById('price').textContent=rp.pricing_analysis||'';
      document.getElementById('sent').textContent=rp.sentiment_analysis||'';
      document.getElementById('trend').textContent=rp.market_trends||'';
      const ul=document.getElementById('recs');
      ul.innerHTML='';
      (rp.strategic_recommendations||[]).forEach(function(rec,i){
        const li=document.createElement('li');
        li.textContent=(i+1)+'. '+rec;
        ul.appendChild(li);
      });
      res.style.display='block';
      // labels
      var sl=function(id,en,fr_){var el=document.getElementById(id);if(el)el.textContent=fr?fr_:en;};
      sl('h-exec','Executive Summary','R\u00e9sum\u00e9 ex\u00e9cutif');
      sl('h-price','Pricing Analysis','Analyse des prix');
      sl('h-sent','Customer Sentiment','Sentiment client');
      sl('h-trend','Market Trends','Tendances du march\u00e9');
      sl('h-recs','Strategic Recommendations','Recommandations strat\u00e9giques');
      sl('h-chart-sent','Customer Sentiment','Sentiment client');
      sl('sent-score-label','sentiment score','score de sentiment');
      // sentiment doughnut: use score + sentiment when key_positives/key_negatives are empty (e.g. fast model)
      var wrap=document.getElementById('chart-wrap');
      var ins=rp.raw_sentiment&&rp.raw_sentiment.insights?rp.raw_sentiment.insights:null;
      if(ins){
        var score=Math.min(1,Math.max(0,Number(ins.score)||0));
        var scorePct=Math.round(score*100);
        document.getElementById('sent-score-num').textContent=scorePct+'%';
        wrap.style.display='flex';
        var pos=ins.key_positives?ins.key_positives.length:0;
        var neg=ins.key_negatives?ins.key_negatives.length:0;
        var hasCounts=pos+neg>0;
        var sent=(ins.sentiment||'').toLowerCase();
        var data;
        if(hasCounts){
          var neu=Math.max(0,5-pos-neg);
          data=[pos,neg,neu];
        } else {
          var rest=(1-score)/2;
          if(sent==='positive') data=[score,rest,rest];
          else if(sent==='negative') data=[rest,score,rest];
          else data=[rest,rest,score];
          data=data.map(function(v){return Math.round(v*100);});
        }
        if(_sentChart){_sentChart.destroy();_sentChart=null;}
        _sentChart=new Chart(document.getElementById('chartSent'),{
          type:'doughnut',
          data:{
            labels:fr?['Positif','N\u00e9gatif','Neutre']:['Positive','Negative','Neutral'],
            datasets:[{
              data:data,
              backgroundColor:['#16a34a','#dc2626','#d1d5db'],
              borderWidth:0,
            }]
          },
          options:{
            responsive:true,
            cutout:'70%',
            plugins:{
              legend:{position:'bottom',labels:{font:{size:11},padding:10,boxWidth:12}}
            }
          }
        });
      } else {
        wrap.style.display='none';
      }
    }
  } catch(e){
    err.textContent='\u26a0 Error: '+e.message;
    err.style.display='block';
  } finally{
    btn.disabled=false;
    btn.textContent='Analyse';
  }
}
document.addEventListener('DOMContentLoaded',function(){
  document.getElementById('q').addEventListener('keydown',function(e){if(e.key==='Enter')run();});
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return HTMLResponse(_HTML)