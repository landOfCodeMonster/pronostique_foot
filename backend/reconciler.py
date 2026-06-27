from __future__ import annotations

from backend import storage
from backend.football_api import FootballAPI


def reconcile(api: FootballAPI, conn) -> int:
    """Attach real scores to predicted matches once finished.

    Saves a result for any predicted match that is now finished, and also
    updates a previously stored result whose score was later corrected
    (e.g. a goal disallowed by VAR, or a delayed update from the API).
    Returns the number of results created or updated.
    """
    count = 0
    for match in api.get_finished_matches():
        if match["home_goals"] is None:
            continue
        if storage.get_prediction(conn, match["id"]) is None:
            continue
        existing = storage.get_result(conn, match["id"])
        if (existing is None
                or existing["actual_home"] != match["home_goals"]
                or existing["actual_away"] != match["away_goals"]):
            storage.save_result(
                conn, match["id"], match["home_goals"], match["away_goals"], match["status"]
            )
            count += 1
    return count
