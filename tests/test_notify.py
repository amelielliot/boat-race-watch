from datetime import datetime

from boat_watch.notify import Ntfy


class Response:
    text = ""

    def raise_for_status(self):
        return None


class Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response()


def test_publish_uses_utf8_safe_json():
    session = Session()
    notifier = Ntfy("hard-to-guess-topic", session)
    notifier.publish("【見送り】桐生1R", "理由：強風")

    url, kwargs = session.calls[0]
    assert url == "https://ntfy.sh"
    assert kwargs["json"]["title"] == "【見送り】桐生1R"
    assert kwargs["json"]["message"] == "理由：強風"
    assert kwargs["json"]["topic"] == "hard-to-guess-topic"


def test_history_deduplicates_and_totals_today():
    notifier = Ntfy("topic")
    notifier.recent = lambda: [
        {"message": "管理ID:20260921-01-01;予定額:1200"},
        {"message": "管理ID:20260921-02-02;見送り"},
        {"message": "管理ID:20260920-03-03;予定額:2000"},
    ]

    keys, total = notifier.used_keys_and_budget(datetime(2026, 9, 21, 12, 0))
    assert keys == {"20260921-01-01", "20260921-02-02"}
    assert total == 1200


def test_history_parses_bets_and_settled_profit():
    items = [
        {"message": "管理ID:20260921-01-01;予定額:1200;買い目:1-2-3@500,1-3-2@700"},
        {"message": "結果ID:20260921-01-01;収支:+5000"},
        {"message": "結果ID:20260921-02-02;収支:-1200"},
        {"message": "結果ID:20260920-03-03;収支:9999"},
    ]
    notifier = Ntfy("topic")

    records = notifier.recorded_bets(datetime(2026, 9, 21, 12, 0), items)
    settled, profit = notifier.settled_results(datetime(2026, 9, 21, 12, 0), items)

    assert records["20260921-01-01"].bets == {"1-2-3": 500, "1-3-2": 700}
    assert settled == {"20260921-01-01", "20260921-02-02"}
    assert profit == 3800
