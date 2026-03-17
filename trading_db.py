#!/usr/bin/env python3
"""trading_db.py — rewrote with real learning + transparency"""
import json, os, math, datetime
from collections import defaultdict

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_data")
os.makedirs(DB_DIR, exist_ok=True)

def _db_path(t): return os.path.join(DB_DIR, f"{t.upper()}_db.json")

def load_db(ticker):
    p = _db_path(ticker)
    if not os.path.exists(p): return {"ticker":ticker,"v":2,"days":{}}
    try:
        with open(p) as f: return json.load(f)
    except: return {"ticker":ticker,"v":2,"days":{}}

def _save_db(ticker, db):
    try:
        with open(_db_path(ticker),"w") as f: json.dump(db,f,indent=2,default=str)
    except Exception as e: print(f"[db] {e}")

def save_day(ticker, date_str, snap):
    db = load_db(ticker)
    db["days"][date_str] = snap
    db["updated"] = str(datetime.datetime.now())[:16]
    _save_db(ticker, db)

def build_snapshot(bt):
    if not bt or "error" in bt: return {}
    ao=bt.get("actual_open",0); ah=bt.get("actual_high",0)
    al=bt.get("actual_low",0);  ac=bt.get("actual_close",0)
    pc=bt.get("prev_close",0);  gp=bt.get("gap_pct",0)
    gd=bt.get("gap_dir","flat"); gs=bt.get("gap_size","tiny")
    fp=bt.get("fill_prob",50);  gf=bt.get("gap_filled",False)
    orb_data={}
    for lbl,o in bt.get("orb_results",{}).items():
        if not o: continue
        orb_data[lbl]={"high":o.get("oh",0),"low":o.get("ol",0),
                        "range_pct":o.get("rng",0),"size":o.get("size_cat",""),
                        "broke_up":o.get("actual_bu",False),"broke_down":o.get("actual_bd",False),
                        "broke_both":o.get("actual_both",False),
                        "ext_up":o.get("ext_up_actual",0),"ext_down":o.get("ext_dn_actual",0)}
    signals=[]
    for s in bt.get("signals",[]):
        signals.append({"type":s.get("type",""),"entry":s.get("price",0),
                         "target":s.get("target_price",0),"stop":s.get("stop_price",0),
                         "rr":s.get("rr",0),"touches":s.get("touches",1),
                         "dist_abs":s.get("dist_abs",0),"label":s.get("label",""),
                         "reached":s.get("reached",False),"tgt_hit":s.get("target_hit",False),
                         "move_act":s.get("move_actual",0),
                         "outcome":("TARGET" if s.get("target_hit") else
                                    "HIT"    if s.get("reached")    else "MISS")})
    return {"date":bt.get("date",""),"gap_dir":gd,"gap_pct":round(gp,3),
            "gap_size":gs,"fill_prob":fp,"prev_close":pc,
            "day_dir":"bull" if ac>ao else "bear",
            "day_range":round(ah-al,2),"day_body":round(abs(ac-ao),2),
            "gap_filled":gf,"orb":orb_data,"signals":signals}

def _features(snap):
    gd=snap.get("gap_dir","flat"); gp=abs(snap.get("gap_pct",0))
    gs=snap.get("gap_size","tiny"); fp=snap.get("fill_prob",50)
    o5=snap.get("orb",{}).get("5m",{}); o15=snap.get("orb",{}).get("15m",{})
    sm={"tiny":0,"small":1,"medium":2,"large":3,"huge":4}
    return {"gap_up":1.0 if gd=="up" else 0.0,"gap_down":1.0 if gd=="down" else 0.0,
            "gap_flat":1.0 if gd=="flat" else 0.0,"gap_pct":min(gp/2.0,1.0),
            "gap_size":sm.get(gs,0)/4.0,"fill_prob":fp/100.0,
            "orb5_rng":min(o5.get("range_pct",0)/1.5,1.0),
            "orb5_up":1.0 if o5.get("broke_up") else 0.0,
            "orb5_dn":1.0 if o5.get("broke_down") else 0.0,
            "orb5_both":1.0 if o5.get("broke_both") else 0.0,
            "orb15_up":1.0 if o15.get("broke_up") else 0.0,
            "orb15_dn":1.0 if o15.get("broke_down") else 0.0,
            "orb15_ext_up":min(o15.get("ext_up",0)/0.5,1.0),
            "orb15_ext_dn":min(o15.get("ext_down",0)/0.5,1.0)}

def _features_from_ctx(ctx,gap_dir,gap_pct,gap_size,fill_prob):
    orb_today=ctx.get("orb_today",{}); o5=orb_today.get("5m",{}); o15=orb_today.get("15m",{})
    gp=abs(gap_pct); sm={"tiny":0,"small":1,"medium":2,"large":3,"huge":4}
    return {"gap_up":1.0 if gap_dir=="up" else 0.0,"gap_down":1.0 if gap_dir=="down" else 0.0,
            "gap_flat":1.0 if gap_dir=="flat" else 0.0,"gap_pct":min(gp/2.0,1.0),
            "gap_size":sm.get(gap_size,0)/4.0,"fill_prob":fill_prob/100.0,
            "orb5_rng":min(o5.get("range_pct",0)/1.5,1.0),
            "orb5_up":1.0 if o5.get("broke_up") else 0.0,
            "orb5_dn":1.0 if o5.get("broke_down") else 0.0,
            "orb5_both":1.0 if o5.get("broke_both") else 0.0,
            "orb15_up":1.0 if o15.get("broke_up") else 0.0,
            "orb15_dn":1.0 if o15.get("broke_down") else 0.0,
            "orb15_ext_up":min(o15.get("ext_now_u",0)/0.5,1.0),
            "orb15_ext_dn":min(o15.get("ext_now_d",0)/0.5,1.0)}

def _cos(a,b):
    k=set(a)|set(b)
    dot=sum(a.get(x,0)*b.get(x,0) for x in k)
    na=math.sqrt(sum(a.get(x,0)**2 for x in k))
    nb=math.sqrt(sum(b.get(x,0)**2 for x in k))
    return dot/(na*nb+1e-9)

def train_from_daily(ticker, daily_df, backtest_fn, months=12, min_move=2.0):
    import pandas as pd
    db=load_db(ticker); existing=set(db.get("days",{}).keys())
    cutoff=datetime.date.today()-datetime.timedelta(days=months*30)
    all_dates=[str(d)[:10] for d in daily_df.index if pd.Timestamp(d).date()>=cutoff]
    new_dates=[d for d in all_dates if d not in existing]
    summary={"ticker":ticker,"total":len(all_dates),"already":len(existing),
              "new":len(new_dates),"saved":0,"errors":0,"log":[]}
    if not new_dates:
        summary["log"].append(f"Todo al día — {len(existing)} días ya en la base de datos.")
        return summary
    for date_str in new_dates:
        try:
            bt=backtest_fn(ticker,date_str,min_move)
            if not bt or "error" in bt: summary["errors"]+=1; continue
            snap=build_snapshot(bt)
            if snap:
                db.setdefault("days",{})[date_str]=snap; summary["saved"]+=1
                for s in snap.get("signals",[]):
                    summary["log"].append(f"{date_str} {s['type']} @${s['entry']} → {s['outcome']} mov${s['move_act']}")
            summary["saved"]+=0
        except Exception as e:
            summary["errors"]+=1; summary["log"].append(f"{date_str} ERR: {e}")
    db["updated"]=str(datetime.datetime.now())[:16]
    _save_db(ticker,db)
    summary["db_stats"]=get_stats(ticker)
    summary["log"]=summary["log"][-60:]
    return summary

def pattern_score(ticker, ctx, rec, gap_dir="flat", gap_pct=0,
                  gap_size="tiny", fill_prob=50, top_k=30, min_sim=0.55):
    db=load_db(ticker); days=db.get("days",{})
    if len(days)<5:
        return {"similar_n":0,"quality":0,"recommendation":"TRAIN",
                "rec_label":"⚡ Entrenar primero","rec_color":"var(--text3)",
                "learning_notes":[
                    f"Solo hay {len(days)} días en la base de datos.",
                    "Ve al panel Backtest → botón ⚡ Entrenar (6-12 meses recomendado).",
                    "El sistema necesita ≥5 días similares para dar recomendaciones."],
                "call_win_rate":0,"put_win_rate":0,"call_n":0,"put_n":0,
                "avg_move_call":0,"avg_move_put":0,"day_bull_rate":0,"avg_range":0}
    fv=_features_from_ctx(ctx,gap_dir,gap_pct,gap_size,fill_prob)
    scored=[(sim:=_cos(fv,_features(snap)),d,snap)
            for d,snap in days.items() if (sim:=_cos(fv,_features(snap)))>=min_sim]
    scored.sort(key=lambda x:x[0],reverse=True)
    similar=scored[:top_k]; n=len(similar)
    if n<3:
        return {"similar_n":n,"quality":30,"recommendation":"NEUTRAL",
                "rec_label":"◆ NEUTRAL","rec_color":"var(--both)",
                "learning_notes":[
                    f"Solo {n} días similares encontrados (necesario ≥3).",
                    "Las condiciones de hoy son poco frecuentes en el historial.",
                    "Sin suficiente evidencia — esperar confirmación del ORB."],
                "call_win_rate":0,"put_win_rate":0,"call_n":0,"put_n":0,
                "avg_move_call":0,"avg_move_put":0,"day_bull_rate":0,"avg_range":0}
    bull=0; ranges=[]; call_s=[]; put_s=[]; fills=0; o5u=0; o5d=0
    for sim,d,snap in similar:
        if snap.get("day_dir")=="bull": bull+=1
        ranges.append(snap.get("day_range",0))
        if snap.get("gap_filled"): fills+=1
        o5=snap.get("orb",{}).get("5m",{})
        if o5.get("broke_up"):   o5u+=1
        if o5.get("broke_down"): o5d+=1
        for s in snap.get("signals",[]):
            (call_s if s["type"]=="CALL" else put_s).append(s)
    def ss(sigs):
        if not sigs: return {"n":0,"win":0,"hit":0,"avg_mv":0,"wins":[]}
        wins=[s for s in sigs if s["outcome"]=="TARGET"]
        hits=[s for s in sigs if s["outcome"] in ("TARGET","HIT")]
        return {"n":len(sigs),"win":round(len(wins)/len(sigs)*100,1),
                "hit":round(len(hits)/len(sigs)*100,1),
                "avg_mv":round(sum(s["move_act"] for s in sigs)/len(sigs),2),"wins":wins}
    cs=ss(call_s); ps=ss(put_s)
    bull_r=round(bull/n*100,1); avg_r=round(sum(ranges)/n,2) if ranges else 0
    fill_r=round(fills/n*100,1); o5u_r=round(o5u/n*100,1); o5d_r=round(o5d/n*100,1)
    bias=rec.get("bias",50) if rec else 50
    # Quality
    if bias>=60: ds=cs["win"]*0.7+bull_r*0.3
    elif bias<=40: ds=ps["win"]*0.7+(100-bull_r)*0.3
    else: ds=max(cs["win"],ps["win"])*0.5+50*0.5
    quality=min(95,int(ds*min(n/15,1.0)))
    # Recommendation
    if cs["n"]>=3 and cs["win"]>=60 and bias>=55: rk="STRONG_CALL"; rl=f"▲ CALL FUERTE  {cs['win']:.0f}%"; rc="var(--up)"
    elif cs["n"]>=3 and cs["win"]>=45 and bias>=50: rk="CALL"; rl=f"▲ CALL  {cs['win']:.0f}%"; rc="var(--up)"
    elif ps["n"]>=3 and ps["win"]>=60 and bias<=45: rk="STRONG_PUT"; rl=f"▼ PUT FUERTE  {ps['win']:.0f}%"; rc="var(--down)"
    elif ps["n"]>=3 and ps["win"]>=45 and bias<=50: rk="PUT"; rl=f"▼ PUT  {ps['win']:.0f}%"; rc="var(--down)"
    elif abs(bull_r-50)<10: rk="NEUTRAL"; rl="◆ NEUTRAL"; rc="var(--both)"
    else: rk="WAIT"; rl="⏸ ESPERAR"; rc="var(--text3)"
    # Notes
    notes=[]
    top_d=[d for _,d,_ in similar[:5]]
    notes.append(f"Análisis de {n} días similares (similitud ≥{int(min_sim*100)}%). Más parecidos: {', '.join(top_d[:3])}.")
    if bull_r>=65: notes.append(f"📈 {bull_r}% de esos días cerraron ALCISTAS — historia favorece suba.")
    elif bull_r<=35: notes.append(f"📉 Solo {bull_r}% cerraron alcistas ({100-bull_r:.0f}% bajistas) — historia favorece baja.")
    else: notes.append(f"↔ Días similares: {bull_r}% alcistas — sin sesgo claro de dirección.")
    notes.append(f"📏 Rango promedio en días similares: ${avg_r}. Target máximo realista: ${round(avg_r*0.65,2)}.")
    if o5u_r>=60: notes.append(f"🔺 ORB 5m rompió arriba en el {o5u_r}% de los días similares.")
    if o5d_r>=60: notes.append(f"🔻 ORB 5m rompió abajo en el {o5d_r}% de los días similares.")
    if gap_dir!="flat": notes.append(f"{'📭' if fill_r>=60 else '📮'} Gap {'llenado' if fill_r>=60 else 'NO llenado'} en {fill_r}% de días similares.")
    if cs["n"]>=3:
        if cs["win"]>=55:
            ex=cs["wins"][0] if cs["wins"] else {}
            notes.append(f"✅ Señales CALL: {cs['win']}% éxito (n={cs['n']}, mov prom +${cs['avg_mv']})."+(" Ej: "+str(ex.get('entry','?'))+" → +$"+str(ex.get('move_act','?')) if ex else ""))
        else: notes.append(f"⚠ Señales CALL: solo {cs['win']}% éxito en días similares (n={cs['n']}).")
    if ps["n"]>=3:
        if ps["win"]>=55:
            ex=ps["wins"][0] if ps["wins"] else {}
            notes.append(f"✅ Señales PUT: {ps['win']}% éxito (n={ps['n']}, mov prom -${ps['avg_mv']})."+(" Ej: "+str(ex.get('entry','?'))+" → -$"+str(ex.get('move_act','?')) if ex else ""))
        else: notes.append(f"⚠ Señales PUT: solo {ps['win']}% éxito en días similares (n={ps['n']}).")
    if "CALL" in rk: notes.append(f"🎯 RECOMIENDO CALL: {cs['win']}% win histórico + {bull_r}% días alcistas + sesgo {bias}/100.")
    elif "PUT" in rk: notes.append(f"🎯 RECOMIENDO PUT: {ps['win']}% win histórico + {100-bull_r:.0f}% días bajistas + sesgo {bias}/100.")
    elif rk=="WAIT": notes.append("⏸ ESPERAR: sin ventaja estadística clara en ninguna dirección.")
    return {"similar_n":n,"quality":quality,"recommendation":rk,"rec_label":rl,"rec_color":rc,
            "learning_notes":notes,"call_win_rate":cs["win"],"call_hit_rate":cs["hit"],
            "call_n":cs["n"],"avg_move_call":cs["avg_mv"],"put_win_rate":ps["win"],
            "put_hit_rate":ps["hit"],"put_n":ps["n"],"avg_move_put":ps["avg_mv"],
            "day_bull_rate":bull_r,"avg_range":avg_r,"fill_rate":fill_r,
            "orb5_up_rate":o5u_r,"orb5_dn_rate":o5d_r,
            "top_matches":[(round(s,2),d) for s,d,_ in similar[:5]]}

def get_stats(ticker):
    db=load_db(ticker); days=db.get("days",{}); n=len(days)
    if n==0: return {"ticker":ticker,"n":0,"message":"Sin datos. Usa ⚡ Entrenar en Backtest."}
    bull=0; ranges=[]; gap_st=defaultdict(lambda:{"n":0,"filled":0})
    o5s={"up":0,"dn":0,"both":0,"tot":0}
    so=defaultdict(lambda:{"tot":0,"hits":0,"tgts":0,"mvs":[]})
    for snap in days.values():
        if snap.get("day_dir")=="bull": bull+=1
        ranges.append(snap.get("day_range",0))
        gd=snap.get("gap_dir","flat"); gap_st[gd]["n"]+=1
        if snap.get("gap_filled"): gap_st[gd]["filled"]+=1
        o5=snap.get("orb",{}).get("5m",{})
        if o5:
            o5s["tot"]+=1
            if o5.get("broke_up"):   o5s["up"]+=1
            if o5.get("broke_down"): o5s["dn"]+=1
            if o5.get("broke_both"): o5s["both"]+=1
        for s in snap.get("signals",[]):
            t=s["type"]; so[t]["tot"]+=1
            if s["outcome"] in ("TARGET","HIT"): so[t]["hits"]+=1
            if s["outcome"]=="TARGET": so[t]["tgts"]+=1
            so[t]["mvs"].append(s["move_act"])
    def p(a,b): return round(a/b*100,1) if b else 0
    def ss(t):
        d=so[t]; mv=d["mvs"]
        return {"n":d["tot"],"win_rate":p(d["tgts"],d["tot"]),"hit_rate":p(d["hits"],d["tot"]),
                "avg_move":round(sum(mv)/len(mv),2) if mv else 0}
    ot=o5s["tot"]
    return {"ticker":ticker,"n":n,"updated":db.get("updated",""),
            "bull_rate":p(bull,n),"avg_range":round(sum(ranges)/n,2) if ranges else 0,
            "gap":{gd:{"n":v["n"],"fill_rate":p(v["filled"],v["n"])} for gd,v in gap_st.items()},
            "orb5":{"up_rate":p(o5s["up"],ot),"dn_rate":p(o5s["dn"],ot),"both_rate":p(o5s["both"],ot)},
            "calls":ss("CALL"),"puts":ss("PUT"),"db_path":_db_path(ticker)}

def atr_quality_check(entry,target,stop,atr,**_):
    if atr<=0: return {"valid":True,"notes":[],"quality_score":50}
    move=abs(target-entry); risk=abs(entry-stop)
    rr=round(move/risk,2) if risk>0 else 0
    mpa=move/atr; notes=[]; score=60
    if mpa>1.2: notes.append(f"⚠ Target supera 1.2x ATR ({atr:.2f}) — poco probable en 1 sesión."); score-=20
    elif mpa<0.3: notes.append(f"Target conservador (0.3x ATR) — alta prob."); score+=12
    else: score+=8
    if rr>=2: notes.append(f"R:R excelente ({rr}x)"); score+=15
    elif rr<1: notes.append(f"R:R desfavorable ({rr}x)"); score-=20
    return {"valid":score>=40,"notes":notes,"quality_score":max(0,min(100,score)),"rr":rr}

def build_snapshot_from_backtest(bt): return build_snapshot(bt)
def record_backtest_batch(ticker, backtest_fn, date_list, deep_fn=None, min_move=2.0):
    db=load_db(ticker); existing=set(db.get("days",{}).keys())
    saved=0; skipped=0
    for d in date_list:
        if d in existing: skipped+=1; continue
        try:
            bt=backtest_fn(ticker,d,min_move)
            if bt and "error" not in bt:
                snap=build_snapshot(bt)
                if snap: db.setdefault("days",{})[d]=snap; saved+=1
        except: pass
    db["updated"]=str(datetime.datetime.now())[:16]
    _save_db(ticker,db)
    return {"saved":saved,"skipped":skipped,"total":len(db.get("days",{}))}

if __name__=="__main__":
    print("trading_db OK"); print("DB dir:", DB_DIR)
    s=get_stats("QQQ"); print(s.get("n",0),"días ·",s.get("message",""))