from __future__ import annotations

from backend import storage
from backend.football_api import FootballAPI


def reconcile(api: FootballAPI, conn) -> int:
    pending = set(storage.predicted_match_ids_without_result(conn))
    if not pending:
        return 0
    count = 0
    for match in api.get_finished_matches():
        if match["id"] in pending and match["home_goals"] is not None:
            storage.save_result(
                conn, match["id"], match["home_goals"], match["away_goals"], match["status"]
            )
            count += 1
    return count
