from boat_watch.data import parse_trifecta_odds


def test_parse_six_block_table():
    rows = []
    for second_index in range(5):
        second_by_first = []
        for first in range(1, 7):
            seconds = [n for n in range(1, 7) if n != first]
            second = seconds[second_index]
            thirds = [n for n in range(1, 7) if n not in {first, second}]
            second_by_first.append((second, thirds))
        for third_index in range(4):
            cells = []
            for first, (second, thirds) in enumerate(second_by_first, start=1):
                cells.extend([str(second), str(thirds[third_index]), f"{first + second + thirds[third_index]}.1"])
            rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in cells) + "</tr>")
    html = "<table>" + "".join(rows) + "</table>"
    odds = parse_trifecta_odds(html)
    assert len(odds) == 120
    assert odds["1-2-3"] == 6.1
