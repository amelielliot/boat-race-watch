import copy
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import requests

from test_validation import race, model
from boat_watch.learned_decision import decide_learned, load_model
from boat_watch.validation_model import (PAIRS, TRIPLES, TRIFECTA_MODEL_VERSION,
    learned_probabilities, learned_trifecta_probabilities, triple_outcome, fit_weights)
from boat_watch.ledger import Ledger
from boat_watch.main import _format
from boat_watch.comparison import ComparisonBook

JST=ZoneInfo("Asia/Tokyo")


def trained():
    m = model()
    m.update(version=TRIFECTA_MODEL_VERSION,third_weights=[0.]*len(m["features"]),
             third_temperature=1.,notification_id="conditional-logit-v2-3t@test")
    m["first_weights"][0]=3.
    m["second_weights"][1]=3.
    m["third_weights"][2]=3.
    counts=np.zeros((6,6,6))
    for t in TRIPLES:
        counts[tuple(int(i)-1 for i in t.split("-"))]=1
    m["course_triple_counts"]=counts.tolist()
    return m


def upcoming():
    r=race()
    r["result"]={"racers":{str(i):{"place_number":None,"start_timing":None} for i in range(1,7)},
                 "payouts":{"exacta":[],"trifecta":[]},"refunds":[]}
    return r


def config():
    return SimpleNamespace(max_per_race=2000,min_expected_value=1.15,max_wind_speed=6,max_wave_height=10)


def decision(pair_quote=1., triple_quote=10., m=None, budget=2000):
    return decide_learned(upcoming(),dict.fromkeys(PAIRS,pair_quote),budget,config(),
                          m or trained(),datetime(2026,10,30,12,tzinfo=JST),
                          trifecta_odds=dict.fromkeys(TRIPLES,triple_quote))


def test_120_probabilities_sum_to_one_and_marginalize_to_exacta():
    m=trained()
    pairs=learned_probabilities(upcoming(),m)
    triples=learned_trifecta_probabilities(upcoming(),m)
    assert set(triples)==set(TRIPLES)
    assert sum(triples.values())==pytest.approx(1.)
    for pair,p in pairs.items():
        assert sum(v for t,v in triples.items() if t.startswith(pair+"-"))==pytest.approx(p)


def test_selects_learned_trifecta_without_increasing_total_budget():
    d=decision()
    assert d.action=="BUY"
    assert all(b.bet_type=="3連単" for b in d.bets)
    assert sum(b.stake for b in d.bets)==2000
    assert len(d.bets)<=3
    assert d.model_version.endswith("@test+trifecta-hit-v1")


def test_skips_instead_of_falling_back_to_exacta():
    d=decision(pair_quote=10.,triple_quote=1.)
    assert d.action=="SKIP"
    assert not d.bets


def test_hit_first_policy_rejects_low_probability_longshots():
    triple_odds = dict.fromkeys(TRIPLES, 1.)
    triple_odds["6-5-4"] = 80.
    d=decide_learned(upcoming(),None,2000,config(),trained(),
                     datetime(2026,10,30,12,tzinfo=JST),trifecta_odds=triple_odds)
    assert d.action=="SKIP"


def test_hit_first_policy_orders_bets_by_probability_not_raw_ev():
    m=trained()
    m["third_weights"]=[0.]*len(m["third_weights"])
    triple_odds = dict.fromkeys(TRIPLES, 1.)
    triple_odds["1-2-3"] = 10.
    triple_odds["1-2-4"] = 30.
    triple_odds["1-2-5"] = 20.
    d=decide_learned(upcoming(),None,2000,config(),m,
                     datetime(2026,10,30,12,tzinfo=JST),trifecta_odds=triple_odds)
    assert d.action=="BUY"
    assert d.bets[0].combination=="1-2-3"
    assert d.bets[0].probability > d.bets[-1].probability
    assert d.bets[0].expected_value < d.bets[-1].expected_value


def test_missing_trifecta_odds_does_not_fall_back_to_exacta():
    d=decide_learned(upcoming(),dict.fromkeys(PAIRS,10.),2000,config(),trained(),
                     datetime(2026,10,30,12,tzinfo=JST),trifecta_odds=None)
    assert d.action=="SKIP" and not d.bets


def test_missing_pair_odds_still_allows_valid_trifecta():
    d=decide_learned(upcoming(),None,2000,config(),trained(),
                     datetime(2026,10,30,12,tzinfo=JST),trifecta_odds=dict.fromkeys(TRIPLES,10.))
    assert d.action=="BUY" and all(b.bet_type=="3連単" for b in d.bets)


@pytest.mark.parametrize("bad", [None,"old","missing_weights"])
def test_unavailable_or_old_model_never_silently_uses_heuristic(bad):
    m=None if bad is None else model() if bad=="old" else trained()
    if bad=="missing_weights":
        del m["third_weights"]
    d=decide_learned(upcoming(),dict.fromkeys(PAIRS,20.),2000,config(),m,
                     datetime(2026,10,30,12,tzinfo=JST),dict.fromkeys(TRIPLES,50.))
    assert d.action=="SKIP" and not d.bets


@pytest.mark.parametrize("kind",["wind","missing_preview","result_known","calibration_period"])
def test_safety_gates_and_temporal_boundary(kind):
    r=upcoming()
    now=datetime(2026,10,30,12,tzinfo=JST)
    if kind=="wind":r["preview"]["wind_speed"]=6
    elif kind=="missing_preview":r["preview"]={}
    elif kind=="result_known":r["result"]["racers"]["1"]["start_timing"]=.1
    else:now=now.replace(day=28)
    d=decide_learned(r,dict.fromkeys(PAIRS,20.),2000,config(),trained(),now,dict.fromkeys(TRIPLES,50.))
    assert d.action=="SKIP"


def test_budget_below_three_units_does_not_overallocate():
    d=decision(pair_quote=10.,triple_quote=10.,budget=100)
    assert sum(b.stake for b in d.bets)==100
    assert len(d.bets)==1


def test_notification_and_ledger_recovery_preserve_model_identity(tmp_path):
    d=decision()
    now=datetime(2026,10,30,12,tzinfo=JST)
    title,message,_=_format(upcoming(),d,"20261030-01-01",True)
    assert "検証中・買わない" in title
    assert "モデル：conditional-logit-v2-3t@test" in message
    ledger=Ledger(tmp_path/"ledger.json",tmp_path/"PERFORMANCE.md")
    ledger.reconcile_history([{"message":message,"time":int(now.timestamp())}],now)
    assert ledger.records["20261030-01-01"]["model_version"]==d.model_version
    ledger.save()
    assert "モデル別成績" in (tmp_path/"PERFORMANCE.md").read_text()


def test_trifecta_book_is_separate_and_settles_actual_payout(tmp_path):
    root=tmp_path/"data"
    root.mkdir()
    (root/"notification-model.json").write_text(json.dumps(trained()),encoding="utf-8")
    b=ComparisonBook(root,tmp_path/"live3.md",ticket_code="3T")
    now=datetime(2026,10,30,12,tzinfo=JST)
    assert b.capture(upcoming(),dict.fromkeys(TRIPLES,10.),now)
    r=race()
    r["result"]["payouts"]["trifecta"]=[{"combination":"1-2-3","amount":2500}]
    assert triple_outcome(r)==(0,1,2)
    b.settle({"programs":{"stadiums":{"1":{"races":{"1":r}}}}},now)
    b.save()
    assert b.path.name=="live-3t.json"
    assert b.summary()["学習版3着対応"]["returned"]==2500
    assert b.summary()["学習版3着対応"]["profit"]==2400


def test_fit_can_exclude_two_finishers():
    x=np.zeros((2,6,3))
    x[:,2,0]=1.
    w=fit_weights(x,np.array([2,2]),np.array([[0,1],[0,1]]))
    assert w[0]>0


def test_model_identifier_matches_file_on_reload(tmp_path):
    path=tmp_path/"m.json"
    path.write_text(json.dumps({k:v for k,v in trained().items() if k!="notification_id"}))
    a,b=load_model(path),load_model(path)
    assert a["notification_id"]==b["notification_id"]


def test_notification_model_preserves_frozen_first_and_second_place_models():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/"data"/"validation"
    base=json.loads((root/"model.json").read_text())
    active=load_model(root/"notification-model.json")
    for key in ("first_weights","second_weights","first_temperature","second_temperature"):
        assert base[key]==active[key]


@pytest.mark.parametrize("publish_fails", [False, True])
def test_main_routes_notifications_to_learned_model(tmp_path,monkeypatch,publish_fails):
    from boat_watch import main
    now=datetime.now(JST)
    r=upcoming()
    r["date"]=now.date().isoformat()
    r["closed_at"]=(now+timedelta(minutes=10)).isoformat()
    sent=[]
    class FakeNotifier:
        def __init__(self,*args):pass
        def recent(self):return []
        def used_amounts(self,*args):return {}
        def recorded_bets(self,*args):return {}
        def settled_results(self,*args):return set(),0
        def publish(self,*args):
            if publish_fails:
                raise requests.ConnectionError("ntfy unavailable")
            sent.append(args)
    def disabled_book(*args,**kwargs):raise ValueError("isolated test")
    monkeypatch.setattr(main,"ComparisonBook",disabled_book)
    monkeypatch.setattr(main,"Ntfy",FakeNotifier)
    monkeypatch.setattr(main,"load_model",trained)
    monkeypatch.setattr(main,"Ledger",lambda:Ledger(tmp_path/"ledger.json",tmp_path/"report.md"))
    monkeypatch.setattr(main,"fetch_programs",lambda *args:{"programs":{"stadiums":{"1":{"races":{"1":r}}}}})
    monkeypatch.setattr(main,"fetch_exacta_odds",lambda *args:dict.fromkeys(PAIRS,1.))
    monkeypatch.setattr(main,"fetch_trifecta_odds",lambda *args:dict.fromkeys(TRIPLES,10.))
    assert main.run(now)==0
    if publish_fails:
        assert sent == []
    else:
        assert len(sent)==1 and "3連単" in sent[0][1] and "conditional-logit-v2-3t@test+trifecta-hit-v1" in sent[0][1]
